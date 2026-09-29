// =============================================================================
// fa_attention_kernel.cc — Hygon FlashAttention C 库 → XLA FFI/legacy custom call 胶水
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <cmath>
#include <cstddef>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"   // 类型化 FFI (XLA_FFI_DEFINE_HANDLER)
#include "flash.h"

static_assert(sizeof(Flash_fwd_params) == 0x2a0, "flash.h 与 libflash_attention.so.1 (09/21) 不匹配");

static void ResolveHipMem();
static int32_t* acquire_workspace(hipStream_t stream);
static thread_local const char* fa_native_error = nullptr;

// --------------------------- 算子库句柄 (运行时 dlopen) ----------------------
// 按 FA_LIB_PATH 或 /opt/libflash_attention.so 延迟加载算子库。
// 首次加载结果（包括失败）会被缓存，句柄不卸载，以保持函数指针有效。
// fwd/bwd 为必需符号；旧式入口不通过 status 回传错误。
// Flash_*_params 的字段布局须与运行时库一致；大小断言仅检查当前头文件，
// 不能据此保证动态库 ABI 兼容。
struct FaLib {
  void (*fwd)(Flash_fwd_params&, hipStream_t, bool) = nullptr;
  void (*bwd)(Flash_bwd_params&, hipStream_t, bool) = nullptr;
  void (*kvcache)(Flash_fwd_params&, hipStream_t, bool) = nullptr;
  void (*int8_kvcache)(Flash_fwd_params&, hipStream_t, bool) = nullptr;
  void (*prefix_mla)(Flash_fwd_mla_params&, hipStream_t) = nullptr;
  int (*get_arch)() = nullptr;
  bool loaded = false;      // dlopen 本身成功
  bool ok = false;          // 必需符号 (fwd/bwd) 齐
  char path[512] = {};
  char missing[256] = {};   // 缺失的可选符号名 (逗号分隔, 诊断用)
};

static FaLib& fa_lib() {
  static FaLib lib = [] {
    FaLib l;
    const char* env = getenv("FA_LIB_PATH");
    const char* path = (env && env[0]) ? env : "/opt/libflash_attention.so";
    snprintf(l.path, sizeof(l.path), "%s", path);
    void* h = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (!h) {
      fprintf(stderr, "[fa_attention] 无法加载算子库 %s: %s\n", path, dlerror());
      return l;   // loaded=false: 后续所有 handler 报错而非静默
    }
    l.loaded = true;
    auto sym = [&](const char* n) { return (void*)dlsym(h, n); };
    auto need = [&](void* p, const char* show) {
      if (!p) {
        size_t n = strlen(l.missing);
        snprintf(l.missing + n, sizeof(l.missing) - n, "%s%s", n ? ", " : "", show);
      }
    };
    l.fwd = (decltype(l.fwd))sym("_Z11run_mha_fwdR16Flash_fwd_paramsP12ihipStream_tb");
    l.bwd = (decltype(l.bwd))sym("_Z11run_mha_bwdR16Flash_bwd_paramsP12ihipStream_tb");
    l.kvcache = (decltype(l.kvcache))sym("_Z19run_mha_fwd_kvcacheR16Flash_fwd_paramsP12ihipStream_tb");
    l.int8_kvcache = (decltype(l.int8_kvcache))sym("_Z20run_int8_fwd_kvcacheR16Flash_fwd_paramsP12ihipStream_tb");
    l.prefix_mla = (decltype(l.prefix_mla))sym("_Z26run_fwd_prefix_prefill_mlaR20Flash_fwd_mla_paramsP12ihipStream_t");
    l.get_arch = (decltype(l.get_arch))sym("_Z7getArchv");
    need((void*)l.kvcache, "kvcache");
    need((void*)l.prefix_mla, "prefix_mla"); need((void*)l.get_arch, "getArch");
    l.ok = l.fwd && l.bwd;
    if (!l.ok)
      fprintf(stderr, "[fa_attention] 算子库 %s 缺少必需符号 (run_mha_fwd/run_mha_bwd), "
                      "所有入口将拒绝调用\n", path);
    else if (l.missing[0])
      fprintf(stderr, "[fa_attention] 算子库 %s 不含可选接口: %s "
                      "(对应 API 调用时会显式报错)\n", path, l.missing);
    return l;
  }();
  return lib;
}

// handler 公用的前置检查: 库没装好或缺该符号 → 返回可读错误信息 (不发射)。
static const char* fa_lib_check(const char* what, const void* sym) {
  const FaLib& l = fa_lib();
  if (!l.loaded)
    return "算子库未加载 (dlopen 失败; 请检查 /opt/libflash_attention.so 或 FA_LIB_PATH)";
  if (!l.ok)
    return "算子库缺少必需符号 (run_mha_fwd/run_mha_bwd), 版本可能不匹配";
  if (!sym) {
    static thread_local char buf[256];
    snprintf(buf, sizeof(buf), "当前算子库 (%s) 不含接口 %s —— 该入口在此版本已被移除, "
                               "请升级/更换算子库", l.path, what);
    return buf;
  }
  return nullptr;
}



// ---------------------------- 小工具 ----------------------------------------
static XLA_FFI_Error* MakeError(const XLA_FFI_CallFrame* f, const char* msg) {
  XLA_FFI_Error_Create_Args a;
  memset(&a, 0, sizeof(a));
  a.message = msg;
  a.errc = XLA_FFI_Error_Code_INVALID_ARGUMENT;
  return f->api->XLA_FFI_Error_Create(&a);
}

// 标量属性: attrs[i] → XLA_FFI_Scalar{dtype, value*}
static const void* AttrByName(const XLA_FFI_CallFrame* f, const char* name) {
  for (int64_t i = 0; i < f->attrs.size; ++i) {
    const XLA_FFI_ByteSpan& n = *f->attrs.names[i];
    if (strncmp(n.ptr, name, n.len) == 0 && name[n.len] == '\0') {
      const XLA_FFI_Scalar* s = (const XLA_FFI_Scalar*)f->attrs.attrs[i];
      return s->value;
    }
  }
  return nullptr;
}

static XLA_FFI_Error* FaFwdImpl(XLA_FFI_CallFrame* frame, bool causal_override) {
  if (frame->args.size != 6 || frame->rets.size != 2) {
    return MakeError(frame, "fa_fwd: 期望 args=q,k,v,rng,dbg,sem; rets=o,lse");
  }
  const XLA_FFI_Buffer* q = (const XLA_FFI_Buffer*)frame->args.args[0];
  const XLA_FFI_Buffer* k = (const XLA_FFI_Buffer*)frame->args.args[1];
  const XLA_FFI_Buffer* v = (const XLA_FFI_Buffer*)frame->args.args[2];

  const float* scale_p = (const float*)AttrByName(frame, "softmax_scale");
  const int32_t* causal_p = (const int32_t*)AttrByName(frame, "causal");

  const int64_t B = q->dims[0], H = q->dims[1], SQ = q->dims[2], D = q->dims[3];
  const float scale = scale_p ? *scale_p : (float)(1.0 / std::sqrt((double)D));
  const bool causal = causal_override;
  const int64_t HK = k->dims[1], SK = k->dims[2], DV = v->dims[3];

  auto rm = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
  const int64_t sq_r = rm(SQ, 32), sk_r = rm(SK, 32);
  const int64_t d = rm(D, 8), d_r = rm(d, 32);
  const int64_t dv = rm(DV, 8), dv_r = rm(dv, 32);

  Flash_fwd_params p;
  memset(&p, 0, sizeof(p));   // 与 set_params_fprop 一致: 未列字段全 0

  p.q_ptr = q->data; p.k_ptr = k->data; p.v_ptr = v->data;
  p.q_batch_stride = (int32_t)(H * SQ * D);
  p.k_batch_stride = (int32_t)(HK * SK * D);
  p.v_batch_stride = (int32_t)(HK * SK * DV);
  p.q_row_stride = (int32_t)D;  p.k_row_stride = (int32_t)D;
  p.v_row_stride = (int32_t)DV; p.v_dim_stride = 0;
  p.q_head_stride = (int32_t)(SQ * D); p.k_head_stride = (int32_t)(SK * D);
  p.v_head_stride = (int32_t)(SK * DV);
  p.h = (int32_t)H; p.h_k = (int32_t)HK; p.h_h_k_ratio = (int32_t)(H / HK);

  const XLA_FFI_Buffer* o_buf = (const XLA_FFI_Buffer*)frame->rets.rets[0];
  const XLA_FFI_Buffer* lse_buf = (const XLA_FFI_Buffer*)frame->rets.rets[1];
  p.o_ptr = o_buf->data;
  p.o_batch_stride = (int32_t)(H * SQ * DV);
  p.o_row_stride = (int32_t)DV;
  p.o_head_stride = (int32_t)(SQ * DV);
  p.softmax_lse_ptr = lse_buf->data;

  // 运行期契约: 仅 sem 必须分配+清零; 其余 NULL (内核有守卫)。
  p.p_ptr = nullptr;
  p.scores_sum_ptr = nullptr;
  p.scores_max_ptr = nullptr;
  p.rng_state = (uint64_t*)((const XLA_FFI_Buffer*)frame->args.args[3])->data;
  p.dropout_debug_count = (uint32_t*)((const XLA_FFI_Buffer*)frame->args.args[4])->data;
  p.tile_count_semaphore = (int32_t*)((const XLA_FFI_Buffer*)frame->args.args[5])->data;

  p.b = (int32_t)B; p.seqlen_q = (int32_t)SQ; p.seqlen_k = (int32_t)SK;
  p.d = (int32_t)d; p.d_value = (int32_t)dv;
  p.seqlen_q_rounded = (int32_t)sq_r;
  p.seqlen_k_rounded = (int32_t)sk_r;
  p.d_rounded = (int32_t)d_r; p.d_value_rounded = (int32_t)dv_r;
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * 1.4426950408889634f;
  p.rp_dropout = 1.0f; p.scale_softmax_rp_dropout = scale;
  p.window_size_left = -1;
  p.window_size_right = causal ? 0 : -1;
  p.num_splits = 1;      // 本构建 fwd 启发式 num_SMs 恒为 1
  p.partition_size = 0;
  p.arch = 936;
  p.layout = 0;          // 0=(b,h,s,d); bshd 走 layout=1 (后续版本)

  XLA_FFI_Stream_Get_Args stream_args;
  memset(&stream_args, 0, sizeof(stream_args));
  stream_args.ctx = frame->ctx;
  if (XLA_FFI_Error* err = frame->api->XLA_FFI_Stream_Get(&stream_args)) return err;

  if (const char* e = fa_lib_check("run_mha_fwd", (const void*)fa_lib().fwd))
    return MakeError(frame, e);
  fa_lib().fwd(p, (hipStream_t)stream_args.stream, /*force_split_kernel=*/false);
  return nullptr;
}

extern "C" XLA_FFI_Error* fa_fwd(XLA_FFI_CallFrame* frame)       { return FaFwdImpl(frame, false); }
extern "C" XLA_FFI_Error* fa_fwd_causal(XLA_FFI_CallFrame* frame) { return FaFwdImpl(frame, true);  }

// ---------------------------- legacy custom call (api_version=0) ------------
// 字段顺序须与 Python _OPAQUE_FMT 一致，描述符共 104 字节。
// layout：0=(b,h,s,d)，1=(b,s,h,d)；dtype：0=fp16，1=bf16。
// 变长路径中，sq/sk 是最大序列长度，total_q/total_k 是张量容量。
// dropout_p 是丢弃概率，算子参数 p_dropout 是保留概率。
// FP8 复用 total_q/total_k 存放 descale 的批/头步长，dtype 表示输出类型。
// 长度掩码复用 has_sinks：1=全局有效 K 长度，2=逐批有效 Q/K 长度。
// lse_unpadded=1 表示按 (h,total_q) 读取 LSE。
struct FaOpaque {
  int32_t b, h, hk, sq, sk, d, dv;
  float scale;
  int32_t causal;
  int32_t layout;   // 0=(b,h,s,d)  1=(b,s,h,d)
  int32_t dtype;    // 0=fp16  1=bf16
  int32_t window_left;    // -1 = 全窗口
  int32_t window_right;   // -1 = 全窗口; >=0 时与 causal 组合成滑动窗口
  float softcap;          // 0 = 关闭
  int32_t has_alibi;      // alibi_slopes_ptr 非空
  int32_t alibi_batch;    // alibi 是否带 batch 维
  int32_t has_sinks;      // s_aux_ptr 非空
  int32_t sink_type;      // 0 none 1 fp32 2 fp16 3 bf16
  // ---- varlen (cu_seqlens 打包, layout=1 3-D (total,h,d)) ----
  int32_t has_varlen;     // 1 = 启用; sq/sk 字段此时为 max_seqlen_q/k
  int32_t total_q;
  int32_t total_k;
  int32_t vbwd_mode;      // 0=3D 自然(row=h*d,head=d,batch=0) 1=packed(row=d,head=d)
  int32_t lse_unpadded;   // 1 = lse 为 (h,total_q) 非填充布局
  float dropout_p;        // torch 语义: 丢弃概率 (params.p_dropout 存保留概率)
  int32_t is_fp8;         // 1 = q/k/v 为 fp8_e4m3 (dtype 字段此时表示输出 dtype)
  // deterministic=1: handler 分配并清零 dq_accum (契约见 FaBwdLegacy)。
  int32_t deterministic;
};

static_assert(sizeof(FaOpaque) == 104, "FaOpaque 与 Python _OPAQUE_FMT 不一致");

// opaque 版本不匹配是**静默无操作**的高危故障 (旧 wrapper + 新 shim 会"什么都没做"),
// 故长度不符时显式报错并拒绝调用。
static bool opaque_ok(size_t len, const char* who) {
  if (len == sizeof(FaOpaque)) return true;
  fprintf(stderr, "[fa_shim] %s: opaque 长度 %zu != %zu —— shim 与 jax_fa_fwd.py "
                  "版本不匹配, 本次调用被拒绝 (请成对更新后重试)\n",
          who, len, sizeof(FaOpaque));
  return false;
}

// hipMemsetAsync: 每次发射前清零调度计数器 (进程级解析一次)
static int (*sem_memset_async)(void*, const void*, size_t, void*) = nullptr;
static void zero_sem(void* sem, void* stream) {
  if (!sem_memset_async) {
    void* hip = dlopen("libgalaxyhip.so.5", RTLD_LAZY | RTLD_GLOBAL);
    if (hip) sem_memset_async = (int (*)(void*, const void*, size_t, void*))dlsym(hip, "hipMemsetAsync");
  }
  if (sem_memset_async) sem_memset_async(sem, nullptr, 64 * sizeof(int32_t), stream);
}


static void apply_dropout(Flash_fwd_params& p, float dp, void* rng_buf,
                          hipStream_t stream);
static void maybe_write_magic(Flash_bwd_params& p, hipStream_t stream);
static int cu_count_of_current_device();
static int (*hip_memset_async)(void*, int, size_t, void*);   // 定义在后


static void FaFwdLegacyImpl(hipStream_t stream, void** buffers, const FaOpaque& op,
                            const int res_base, void* dq_desc, void* dk_desc,
                            void* dv_desc);

// 前向与反向共用窗口归一化，避免掩码语义不一致。
// 先把覆盖全部 K 的窗口转为 -1；启用因果约束时将右窗限制为 0。
// is_causal 必须在把单侧无限窗口替换为 sk 之前判定。
// 单查询的非因果化由调用方决定，prefix 不启用该例外。
struct FaWinCausal { int32_t wl, wr; bool is_causal; };
static FaWinCausal clamp_window_causal(int32_t wl, int32_t wr, int32_t sk,
                                       bool causal, bool force_noncausal) {
  if (sk > 0) {
    if (wl >= sk) wl = -1;
    if (wr >= sk) wr = -1;
  }
  if (force_noncausal) causal = false;
  if (causal) wr = 0;
  FaWinCausal r;
  r.is_causal = (wl < 0 && wr == 0);
  if (wl < 0 && wr >= 0) wl = sk;
  if (wl >= 0 && wr < 0) wr = sk;
  r.wl = wl; r.wr = wr;
  return r;
}

static void FaFwdLegacy(hipStream_t stream, void** buffers, const FaOpaque& op) {
  FaFwdLegacyImpl(stream, buffers, op, 10, nullptr, nullptr, nullptr);
}

static void FaFwdLegacyImpl(hipStream_t stream, void** buffers, const FaOpaque& op,
                            const int res_base, void* dq_desc, void* dk_desc,
                            void* dv_desc) {
  fa_native_error = nullptr;
  auto rm = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
  const int64_t sq_r = rm(op.sq, 32), sk_r = rm(op.sk, 32);
  const int64_t d = rm(op.d, 8), d_r = rm(d, 32);
  const int64_t dv = rm(op.dv, 8);

  Flash_fwd_params p;
  memset(&p, 0, sizeof(p));
  p.q_ptr = buffers[0]; p.k_ptr = buffers[1]; p.v_ptr = buffers[2];
  p.v_dim_stride = 0;
  p.h = op.h; p.h_k = op.hk; p.h_h_k_ratio = op.h / op.hk;
  p.o_ptr = buffers[res_base];
  p.softmax_lse_ptr = buffers[res_base + 1];
  if (op.layout == 1) {   // (b, s, h, d) 连续 —— 真实模型自然布局, 免转置
    p.q_batch_stride = (int32_t)(op.sq * op.h * op.d);
    p.k_batch_stride = (int32_t)(op.sk * op.hk * op.d);
    p.v_batch_stride = (int32_t)(op.sk * op.hk * dv);
    p.o_batch_stride = (int32_t)(op.sq * op.h * dv);
    p.q_row_stride = (int32_t)(op.h * op.d);  p.k_row_stride = (int32_t)(op.hk * op.d);
    p.v_row_stride = (int32_t)(op.hk * dv);   p.o_row_stride = (int32_t)(op.h * dv);
    p.q_head_stride = (int32_t)op.d;          p.k_head_stride = (int32_t)op.d;
    p.v_head_stride = (int32_t)dv;            p.o_head_stride = (int32_t)dv;
  } else {                // layout=0, (b, h, s, d) 连续
    p.q_batch_stride = (int32_t)(op.h * op.sq * op.d);
    p.k_batch_stride = (int32_t)(op.hk * op.sk * op.d);
    p.v_batch_stride = (int32_t)(op.hk * op.sk * dv);
    p.o_batch_stride = (int32_t)(op.h * op.sq * dv);
    p.q_row_stride = (int32_t)op.d;  p.k_row_stride = (int32_t)op.d;
    p.v_row_stride = (int32_t)dv;
    p.q_head_stride = (int32_t)(op.sq * op.d);
    p.k_head_stride = (int32_t)(op.sk * op.d);
    p.v_head_stride = (int32_t)(op.sk * dv);
    p.o_row_stride = (int32_t)dv;
    p.o_head_stride = (int32_t)(op.sq * dv);
  }
  int32_t* workspace = acquire_workspace(stream);
  if (!workspace) return;
  p.rng_state = reinterpret_cast<uint64_t*>(workspace);
  p.dropout_debug_count = reinterpret_cast<uint32_t*>(workspace + 4);
  p.tile_count_semaphore = workspace + 8;
  p.b = op.b; p.seqlen_q = op.sq; p.seqlen_k = op.sk;
  p.d = (int32_t)d; p.d_value = (int32_t)dv;
  p.seqlen_q_rounded = (int32_t)sq_r;
  p.seqlen_k_rounded = (int32_t)sk_r;
  p.d_rounded = (int32_t)d_r;
  p.d_value_rounded = (int32_t)rm(dv, 32);
  // varlen (cu_seqlens 打包, layout=1 3-D (total,h,d)):
  // strides: row=stride(0), head=stride(-2), batch=0
  if (op.has_varlen) {
    // 容量可以超过 cu_seqlens 描述的打包前缀。
    ResolveHipMem();
    if (!hip_memset_async ||
        hip_memset_async(p.o_ptr, 0, int64_t(op.total_q) * op.h * dv * 2, stream) ||
        hip_memset_async(p.softmax_lse_ptr, 0, int64_t(op.total_q) * op.h * 4, stream)) {
      fa_native_error = "fa: varlen output initialization failed";
      fprintf(stderr, "[fa_attention] %s\n", fa_native_error);
      return;
    }
    p.cu_seqlens_q = static_cast<int32_t*>(buffers[8]);
    p.cu_seqlens_k = static_cast<int32_t*>(buffers[9]);
    p.total_q = op.total_q;
    p.total_k = op.total_k;
    p.is_seqlens_k_cumulative = true;
    p.q_batch_stride = 0; p.k_batch_stride = 0;
    p.v_batch_stride = 0; p.o_batch_stride = 0;
    p.q_head_stride = (int32_t)op.d;   p.k_head_stride = (int32_t)op.d;
    p.v_head_stride = (int32_t)dv;
    p.o_head_stride = (int32_t)dv;
    p.q_row_stride = (int32_t)(op.h * op.d);
    p.k_row_stride = (int32_t)(op.hk * op.d);
    p.v_row_stride = (int32_t)(op.hk * dv);
    p.o_row_stride = (int32_t)(op.h * dv);
  }
  // softcap (与 set_params_fprop 同式): scale_softmax=softcap, 内核做 tanh(s*scale_softmax)*softcap
  if (op.softcap > 0.0f) {
    p.softcap = op.scale / op.softcap;
    p.scale_softmax = op.softcap;
    p.scale_softmax_log2 = op.softcap * 1.4426950408889634f;
  } else {
    p.softcap = 0.0f;
    p.scale_softmax = op.scale;
    p.scale_softmax_log2 = op.scale * 1.4426950408889634f;
  }
  apply_dropout(p, op.dropout_p, buffers[3], stream);
  if (fa_native_error) return;
  const FaWinCausal wc = clamp_window_causal(op.window_left, op.window_right,
                                            (int32_t)op.sk, op.causal != 0,
                                            op.sq == 1 && !op.has_alibi);
  p.is_causal = wc.is_causal;
  p.window_size_left = wc.wl;
  p.window_size_right = wc.wr;
  p.num_splits = 1;
  p.partition_size = 0;
  p.arch = 936;
  p.layout = op.layout;
  p.is_bf16 = (op.dtype == 1);
  if (op.is_fp8) {
    // FP8 (e4m3): is_e4m3 选内核分支; is_bf16 选输出 dtype 与模板 elem_type;
    // descale f32 ≥2 维 (b,h), batch/head stride 取 stride(0)/stride(1)
    p.is_e4m3 = true;
    p.q_descale_ptr = (float*)dq_desc;
    p.k_descale_ptr = (float*)dk_desc;
    p.v_descale_ptr = (float*)dv_desc;
    p.q_descale_batch_stride = op.total_q;   // 复用 opaque 字段承载 descale stride
    p.q_descale_head_stride = op.total_k;
    p.k_descale_batch_stride = op.total_q;
    p.k_descale_head_stride = op.total_k;
    p.v_descale_batch_stride = op.total_q;
    p.v_descale_head_stride = op.total_k;
    static const char* fdbg = getenv("FA_DEBUG");
    if (fdbg) {
      printf("[fa-fp8] dq=%p dk=%p dv=%p (batch_stride=%d head_stride=%d) "
             "is_bf16=%d d=%d dv=%d\n", (void*)p.q_descale_ptr,
             (void*)p.k_descale_ptr, (void*)p.v_descale_ptr,
             p.q_descale_batch_stride, p.q_descale_head_stride, (int)p.is_bf16,
             p.d, p.d_value);
    }
  }
  // alibi / attention sinks
  p.alibi_slopes_ptr = op.has_alibi ? buffers[6] : nullptr;
  p.alibi_slopes_batch_stride = op.alibi_batch ? (int32_t)op.h : 0;
  p.s_aux_ptr = op.has_sinks ? buffers[7] : nullptr;
  p.s_aux_type = op.sink_type;
  zero_sem(p.tile_count_semaphore, stream);
  if (const char* e = fa_lib_check("run_mha_fwd", (const void*)fa_lib().fwd)) { fprintf(stderr, "[fa_attention] %s\n", e); return; }
  fa_lib().fwd(p, stream, false);
}

extern "C" void fa_noop_v0(void* stream, void** buffers, const char* opaque,
                           size_t opaque_len, void* status) {
  (void)stream; (void)buffers; (void)opaque; (void)opaque_len; (void)status;
}

// ================ attn_mask / padding_mask: legacy handler ==================
// mask 类型由 op.has_sinks 指定 (语义见 FaOpaque 处说明)；均不支持 causal/window。
static void FaFwdMaskLegacy(hipStream_t stream, void** buffers, const FaOpaque& op) {
  fa_native_error = nullptr;
  auto rm = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
  const int64_t sq_r = rm(op.sq, 32), sk_r = rm(op.sk, 32);
  const int64_t d = rm(op.d, 8), d_r = rm(d, 32);
  const int64_t dv = rm(op.dv, 8), dv_r = rm(dv, 32);

  Flash_fwd_params p;
  memset(&p, 0, sizeof(p));
  p.q_ptr = buffers[0]; p.k_ptr = buffers[1]; p.v_ptr = buffers[2];
  p.v_dim_stride = 0;
  p.h = op.h; p.h_k = op.hk; p.h_h_k_ratio = op.h / op.hk;
  p.o_ptr = buffers[7];
  p.softmax_lse_ptr = buffers[8];
  if (op.layout == 1) {
    p.q_batch_stride = (int32_t)(op.sq * op.h * op.d);
    p.k_batch_stride = (int32_t)(op.sk * op.hk * op.d);
    p.v_batch_stride = (int32_t)(op.sk * op.hk * dv);
    p.o_batch_stride = (int32_t)(op.sq * op.h * dv);
    p.q_row_stride = (int32_t)(op.h * op.d);  p.k_row_stride = (int32_t)(op.hk * op.d);
    p.v_row_stride = (int32_t)(op.hk * dv);   p.o_row_stride = (int32_t)(op.h * dv);
    p.q_head_stride = (int32_t)op.d;          p.k_head_stride = (int32_t)op.d;
    p.v_head_stride = (int32_t)dv;            p.o_head_stride = (int32_t)dv;
  } else {
    p.q_batch_stride = (int32_t)(op.h * op.sq * op.d);
    p.k_batch_stride = (int32_t)(op.hk * op.sk * op.d);
    p.v_batch_stride = (int32_t)(op.hk * op.sk * dv);
    p.o_batch_stride = (int32_t)(op.h * op.sq * dv);
    p.q_row_stride = (int32_t)op.d;   p.k_row_stride = (int32_t)op.d;
    p.v_row_stride = (int32_t)dv;
    p.q_head_stride = (int32_t)(op.sq * op.d);
    p.k_head_stride = (int32_t)(op.sk * op.d);
    p.v_head_stride = (int32_t)(op.sk * dv);
    p.o_row_stride = (int32_t)dv;
    p.o_head_stride = (int32_t)(op.sq * dv);
  }
  int32_t* workspace = acquire_workspace(stream);
  if (!workspace) return;
  p.rng_state = reinterpret_cast<uint64_t*>(workspace);
  p.dropout_debug_count = reinterpret_cast<uint32_t*>(workspace + 4);
  p.tile_count_semaphore = workspace + 8;
  p.b = op.b; p.seqlen_q = op.sq; p.seqlen_k = op.sk;
  p.d = (int32_t)d; p.d_value = (int32_t)dv;
  p.seqlen_q_rounded = (int32_t)sq_r;
  p.seqlen_k_rounded = (int32_t)sk_r;
  p.d_rounded = (int32_t)d_r;
  p.d_value_rounded = (int32_t)dv_r;
  p.scale_softmax = op.scale;
  p.scale_softmax_log2 = op.scale * 1.4426950408889634f;
  p.softcap = 0.0f;
  apply_dropout(p, op.dropout_p, buffers[3], stream);
  if (fa_native_error) return;
  // mask 类型与指针
  if (op.has_sinks == 1) {
    p.attn_mask = (int32_t*)buffers[6];
    p.padding_mask = nullptr;
  } else {
    p.padding_mask = (int32_t*)buffers[6];
    p.attn_mask = nullptr;
  }
  // padding_mask 内核不支持 causal。
  int32_t wl = op.window_left, wr = op.window_right;
  if (op.has_sinks == 2 && op.causal) {
    return;   // wrapper 已守卫, 双保险
  }
  if (wl < 0 && wr < 0 && op.causal) { wr = 0; }
  p.is_causal = (wl < 0 && wr == 0);
  if (wl < 0 && wr >= 0) { wl = (int32_t)op.sk; }
  if (wl >= 0 && wr < 0) { wr = (int32_t)op.sk; }
  p.window_size_left = wl;
  p.window_size_right = wr;
  p.num_splits = 1;
  p.partition_size = 0;
  p.arch = 936;
  p.layout = op.layout;
  p.is_bf16 = (op.dtype == 1);
  p.cu_count = cu_count_of_current_device();   // torch 各入口均设置
  // mask 内核会累加到 o, 预清零 o/lse。
  ResolveHipMem();
  if (hip_memset_async) {
    const int64_t o_bytes = (int64_t)op.b * op.sq * op.h * dv * ((op.dtype == 1) ? 2 : 2);
    int r1 = hip_memset_async(buffers[7], 0, o_bytes, stream);
    int r2 = hip_memset_async(buffers[8], 0, (int64_t)op.b * op.h * op.sq * 4, stream);
    if (getenv("FA_DEBUG"))
      printf("[fa-mask] memset o(%p,%ldB)=%d lse(%p)=%d\n", buffers[7], (long)o_bytes, r1, buffers[8], r2);
  }
  static const char* dbg = getenv("FA_DEBUG");
  if (dbg) {
    printf("[fa-mask] b=%d h=%d hk=%d sq=%d sk=%d d=%d dv=%d layout=%d causal=%d "
           "wl=%d wr=%d kind=%d p_drop=%g scale=%g is_causal=%d\n",
           op.b, op.h, op.hk, op.sq, op.sk, op.d, op.dv, op.layout, op.causal,
           p.window_size_left, p.window_size_right, op.has_sinks, op.dropout_p,
           op.scale, p.is_causal);
    printf("[fa-mask] o=%p lse=%p mask=%p rng=%p sem=%p\n", p.o_ptr,
           p.softmax_lse_ptr, (void*)p.attn_mask, (void*)p.rng_state,
           (void*)p.tile_count_semaphore);
  }
  zero_sem(p.tile_count_semaphore, stream);
  if (fa_lib().ok) fa_lib().fwd(p, stream, false);
}

// ------------------------------ 架构探测 (FP8 能力门) -----------------------
// 按算子库 getArch() 判断门槛：标准 FP8 为 >=938，FP8 PA 为 930 或 >=938。
// 探测结果会缓存；门槛仅用于能力筛选，不代表完整数值验收。
// 公开入口另行检查类型、维度和 descale；INT8 PA 与 MLA-prefix 仍直接拒绝。
extern "C" int fa_arch_raw() {
  static int cached = -1;
  if (cached >= 0) return cached;
  cached = 0;
  int (*getarch)() = fa_lib().get_arch;
  if (!getarch) {
    void* h = dlopen("libflash_attention.so", RTLD_LAZY | RTLD_GLOBAL);
    if (h) getarch = (int (*)())dlsym(h, "_Z7getArchv");
  }
  if (getarch) cached = getarch();
  return cached;
}

extern "C" int fa_arch_fp8_ok() { return fa_arch_raw() >= 938 ? 1 : 0; }

extern "C" int fa_arch_fp8_pa_ok() {
  const int a = fa_arch_raw();
  return (a == 930 || a >= 938) ? 1 : 0;
}

// ------------------------ FP8 (e4m3) 前向: legacy handler -------------------
extern "C" void fa_fwd_fp8_v0(void* stream, void** buffers, const char* opaque,
                              size_t opaque_len, void* /*status*/) {
  if (!opaque_ok(opaque_len, __func__)) return;
  FaOpaque op;
  memcpy(&op, opaque, sizeof(op));
  FaFwdLegacyImpl((hipStream_t)stream, buffers, op, 13, buffers[10], buffers[11],
                  buffers[12]);
}

extern "C" void fa_fwd_mask_v0(void* stream, void** buffers, const char* opaque,
                               size_t opaque_len, void* /*status*/) {
  if (!opaque_ok(opaque_len, __func__)) return;
  FaOpaque op;
  memcpy(&op, opaque, sizeof(op));
  FaFwdMaskLegacy((hipStream_t)stream, buffers, op);
}

extern "C" void fa_fwd_v0(void* stream, void** buffers, const char* opaque, size_t opaque_len, void* /*status*/) {
  if (!opaque_ok(opaque_len, __func__)) return;
  FaOpaque op;
  memcpy(&op, opaque, sizeof(op));
  if (op.softcap > 0.0f) return;   // 本构建 FLASHATTENTION_DISABLE_SOFTCAP
  FaFwdLegacy((hipStream_t)stream, buffers, op);
}


// ================== PA / paged KV cache: legacy handler =====================
// PA 使用 (b,sq,h,d) 的 Q 和 (blocks,page,hk,d/dv) 的分页 K/V，仅返回 O。
// FaPaOpaque 与 Python _PA_FMT 按字段顺序对应，共 92 字节。
// total_q/total_k 在 FP8 PA 中复用为 descale 批/头步长；
// prefix 中按调用参数传递，不应将 total_k 固定解释为 KV 长度之和。
// [0..4]=q,kcache,vcache,block_table,seqlens_k；
// [5..7]=scores_accum,o_accum,占位，[8..10]=三个缩放槽，[11]=o。
// seqlens_k 经 cu_seqlens_k 传入，但表示非累计长度。
// 分页 K/V 的批维对应块：批/行/头步长为 page*hk*d、hk*d、d，V 换用 dv。
// scores_accum 前 ns*b*h*sq 个 float 为 sum，后一段为 max；
// 各工作区不得与只读常量或彼此重叠。
struct FaPaOpaque {
  int32_t b, h, hk, sq, sk;   // sk = max_seqlen_k
  int32_t d, dv;
  float scale;
  int32_t layout;             // 1 = (b,s,h,d) / (num_blocks,page,hk,d)
  int32_t dtype;              // 0 fp16 1 bf16
  int32_t causal;
  int32_t wl, wr;
  int32_t page_block_size;
  int32_t num_splits;
  int32_t partition_size;
  int32_t mtp;                // 多 token 预测 (= sq)
  int32_t ngroups;            // h / hk
  int32_t bt_stride;          // block_table.stride(0) = max_blocks
  // ---- prefix-prefill 复用本 opaque 的追加字段 (PA 路径不用, 传 0) ----
  int32_t total_q;            // prefix: q 的 total 行数 (fp8 PA: descale batch stride)
  int32_t total_k;            // prefix: seqused_k 之和 (fp8 PA: descale head stride)
  int32_t mtp_sq;             // prefix: max_seqlen_q (= sq, 冗余)
  int32_t is_fp8;             // 0=无 1=FP8 PA (q/kvc fp8+3 descale) 2=INT8 PA (q/kvc int8+3 scales)
};

// hip 运行时按需解析 (与 zero_sem 同款, 避免链接期依赖)
static int (*hip_malloc)(void**, size_t) = nullptr;
static int (*hip_free)(void*) = nullptr;
static int (*hip_memcpy)(void*, const void*, size_t, int) = nullptr;
static int (*hip_memcpy_async)(void*, const void*, size_t, int, void*) = nullptr;
static int (*hip_stream_sync)(void*) = nullptr;
static int (*hip_dev_attr)(int*, int, int) = nullptr;
static int (*hip_get_dev)(int*) = nullptr;
static int (*hip_set_dev)(int) = nullptr;
static int (*hip_device_sync)() = nullptr;
static void ResolveHipMem() {
  static std::once_flag resolved;
  std::call_once(resolved, [] {
    void* hip = dlopen("libgalaxyhip.so.5", RTLD_LAZY | RTLD_GLOBAL);
    if (!hip) return;
    hip_malloc = (int (*)(void**, size_t))dlsym(hip, "hipMalloc");
    hip_free = (int (*)(void*))dlsym(hip, "hipFree");
    hip_memcpy = (int (*)(void*, const void*, size_t, int))dlsym(hip, "hipMemcpy");
    hip_memcpy_async = (int (*)(void*, const void*, size_t, int, void*))dlsym(hip, "hipMemcpyAsync");
    hip_stream_sync = (int (*)(void*))dlsym(hip, "hipStreamSynchronize");
    hip_memset_async = (int (*)(void*, int, size_t, void*))dlsym(hip, "hipMemsetAsync");
    hip_dev_attr = (int (*)(int*, int, int))dlsym(hip, "hipDeviceGetAttribute");
    hip_get_dev = (int (*)(int*))dlsym(hip, "hipGetDevice");
    hip_set_dev = (int (*)(int))dlsym(hip, "hipSetDevice");
    hip_device_sync = (int (*)())dlsym(hip, "hipDeviceSynchronize");
  });
}

// 可写工作区按提交线程、设备和流隔离；设备执行可能晚于主机调用返回，
// 不得把 XLA 输入常量或其他流的工作区当作可写暂存区。
// 工作区为 int32[72]：[0..3] 为 RNG，[4..7] 为调试计数，[8..71] 为信号量。
// dropout 输入须提供完整 16 字节 seed/offset；同流复制到内部区后才交给内核回写。
// dq_accum 按容量复用，扩容涉及原流同步和重新分配。
// 缓存淘汰或线程退出时，切换到所属设备并同步后释放，再恢复原设备。
// 无法确认设备已同步时，不释放仍可能在用的内存。
struct FaStreamScratch {
  int device = -1;
  void* workspace = nullptr;
  void* accum = nullptr;
  int64_t accum_capacity = 0;

  FaStreamScratch() = default;
  FaStreamScratch(const FaStreamScratch&) = delete;
  FaStreamScratch& operator=(const FaStreamScratch&) = delete;
  ~FaStreamScratch() {
    if ((!workspace && !accum) || !hip_get_dev || !hip_set_dev ||
        !hip_device_sync || !hip_free) return;
    int previous = 0;
    if (hip_get_dev(&previous) || hip_set_dev(device)) return;
    if (hip_device_sync() == 0) {
      if (workspace) hip_free(workspace);
      if (accum) hip_free(accum);
    }
    hip_set_dev(previous);
  }
};

static FaStreamScratch* stream_scratch(hipStream_t stream) {
  ResolveHipMem();
  int device = 0;
  if (!hip_get_dev || hip_get_dev(&device) != 0 || !hip_malloc || !hip_memset_async) {
    fa_native_error = "fa: HIP scratch allocator unavailable";
    return nullptr;
  }
  static thread_local std::map<std::pair<int, uintptr_t>, FaStreamScratch> cache;
  const auto key = std::make_pair(device, reinterpret_cast<uintptr_t>(stream));
  auto found = cache.find(key);
  if (found == cache.end()) {
    if (cache.size() >= 32) cache.erase(cache.begin());
    found = cache.try_emplace(key).first;
    found->second.device = device;
  }
  return &found->second;
}

static int32_t* acquire_workspace(hipStream_t stream) {
  FaStreamScratch* scratch = stream_scratch(stream);
  if (!scratch) return nullptr;
  if ((!scratch->workspace && hip_malloc(&scratch->workspace, 72 * sizeof(int32_t))) ||
      hip_memset_async(scratch->workspace, 0, 72 * sizeof(int32_t), stream)) {
    fa_native_error = "fa: workspace allocation or initialization failed";
    return nullptr;
  }
  return static_cast<int32_t*>(scratch->workspace);
}
static int cu_count_of_current_device() {
  ResolveHipMem();
  int device = 0;
  if (!hip_get_dev || hip_get_dev(&device)) return 0;
  static thread_local std::map<int, int> counts;
  auto found = counts.find(device);
  if (found != counts.end()) return found->second;
  int& cached = counts[device];
  static const char* env = getenv("FA_NUM_SM");
  if (env && atoi(env) > 0) { cached = atoi(env); return cached; }
  if (hip_dev_attr) {
    int dev = device, v = 0;
    // hipDevAttrMultiProcessorCount 的枚举值随 DTK 版本不同,
    // 两个候选值都尝试, 取第一个合理结果。
    const int attrs[] = {16, 32};
    for (int i = 0; i < 2; ++i) {
      if (hip_dev_attr(&v, attrs[i], dev) == 0 && v > 0 && v < 1024) {
        cached = v;
        break;
      }
    }
  }
  return cached;
}

template <int N> struct ResourceHolder {
  void* ptr[N] = {};
  int64_t n[N] = {};
};

static void* acquire_accum(int64_t bytes, hipStream_t stream, bool* ok) {
  *ok = false;
  FaStreamScratch* scratch = stream_scratch(stream);
  if (!scratch || bytes <= 0) return nullptr;
  void*& ptr = scratch->accum;
  int64_t& cap = scratch->accum_capacity;
  if (bytes > cap) {
    if (hip_stream_sync && ptr) hip_stream_sync(stream);
    if (ptr && hip_free) { hip_free(ptr); ptr = nullptr; cap = 0; }
    void* np = nullptr;
    if (hip_malloc(&np, (size_t)bytes) != 0 || !np) return nullptr;
    ptr = np; cap = bytes;
  }
  if (hip_memset_async(ptr, 0, (size_t)bytes, stream) != 0) return ptr;
  *ok = true;
  return ptr;
}

static void apply_dropout(Flash_fwd_params& p, float dp, void* rng_buf,
                          hipStream_t stream) {
  if (dp <= 0.0f) {
    p.p_dropout = 1.0f;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.p_dropout_in_uint8_t = 255;
    p.rand_seed = 0; p.rand_offset = 0;
    return;
  }
  if (dp >= 1.0f) dp = 0.9999f;
  const float keep = 1.0f - dp;
  p.p_dropout = keep;
  p.p_dropout_in_uint8_t = (uint8_t)(keep * 255.0f);
  p.rp_dropout = 1.0f / keep;
  p.scale_softmax_rp_dropout = p.rp_dropout * p.scale_softmax;
  ResolveHipMem();
  if (!hip_memcpy_async || !hip_stream_sync || !p.rng_state || !rng_buf) {
    fa_native_error = "fa: dropout RNG copy unavailable";
    return;
  }
  uint64_t host[2] = {0, 0};
  int copy = hip_memcpy_async(p.rng_state, rng_buf, 16, hipMemcpyDeviceToDevice, stream);
  int read = hip_memcpy_async(host, rng_buf, 16, hipMemcpyDeviceToHost, stream);
  int sync = hip_stream_sync(stream);
  if (copy || read || sync) {
    fa_native_error = "fa: dropout RNG copy failed";
    return;
  }
  p.rand_seed = host[0];
  p.rand_offset = host[1];
}

// FA_DEBUG 用于诊断输出；FA_PREFIX_*、FA_VBWD_*、FA_BWD_DET 和 FA_NUM_SM
// 还可能覆盖参数、工作区或发射方式，不能视为只读日志开关。
// 诊断结果不能替代正常配置的数值验收。
static void maybe_write_magic(Flash_bwd_params& p, hipStream_t stream) {
  static const char* e = getenv("FA_VBWD_MAGIC");
  if (!e || e[0] != '1') return;
  ResolveHipMem();
  static float v7 = 7.0f, v8 = 8.0f, v9 = 9.0f, v10 = 10.0f;
  if (hip_memcpy_async) {
    hip_memcpy_async(p.dq_ptr, &v7, 4, 2, stream);
    hip_memcpy_async(p.dk_ptr, &v8, 4, 2, stream);
    hip_memcpy_async(p.dv_ptr, &v9, 4, 2, stream);
    hip_memcpy_async(p.dsoftmax_sum, &v10, 4, 2, stream);
  } else if (hip_memcpy) {
    hip_memcpy(p.dq_ptr, &v7, 4, 2);
    hip_memcpy(p.dk_ptr, &v8, 4, 2);
    hip_memcpy(p.dv_ptr, &v9, 4, 2);
    hip_memcpy(p.dsoftmax_sum, &v10, 4, 2);
  } else {
    return;
  }
  printf("[fa] magic 写入 dq=%p dk=%p dv=%p dsm=%p%s\n", p.dq_ptr, p.dk_ptr,
         p.dv_ptr, p.dsoftmax_sum, hip_memcpy_async ? " (async, 同流)" : " (sync)");
  if (hip_stream_sync && hip_memcpy) {
    float got[4] = {-1.f, -1.f, -1.f, -1.f};
    hip_stream_sync(stream);
    hip_memcpy(&got[0], p.dq_ptr, 4, 1);
    hip_memcpy(&got[1], p.dk_ptr, 4, 1);
    hip_memcpy(&got[2], p.dv_ptr, 4, 1);
    hip_memcpy(&got[3], p.dsoftmax_sum, 4, 1);
    printf("[fa] 发射前回读: dq=%g dk=%g dv=%g dsm=%g (期望 7/8/9/10)\n",
           (double)got[0], (double)got[1], (double)got[2], (double)got[3]);
  }
}

// ================ prefix-prefill (varlen Q + 分页 KV): legacy handler =========
// prefix 将三维 Q 与分页 K/V 组合，复用 FaPaOpaque。
// [0..5]=q,kcache,vcache,block_table,seqused_k,cuq；
// [6]=cu_k 占位，[7]=sem（当前未读取），[8..9]=o,lse。
// cu_seqlens_k 与 seqused_k 均指向 [4] 的逐序列 KV 长度，使用非累计语义。
// Q/O 批步长为 0，行步长为 h*d/h*dv；K/V 分页步长沿用 PA。
// LSE 按 (h,total_q) 排列。
static void FaPrefixLegacy(hipStream_t stream, void** buffers, const FaPaOpaque& op) {
  const int64_t d_r = ((int64_t)op.d + 31) / 32 * 32;
  const int64_t dv_r = (((int64_t)op.dv + 7) / 8 * 8 + 31) / 32 * 32;
  const int32_t page = op.page_block_size;

  Flash_fwd_params p;
  memset(&p, 0, sizeof(p));
  p.q_ptr = buffers[0]; p.k_ptr = buffers[1]; p.v_ptr = buffers[2];
  p.o_ptr = buffers[8];
  p.h = op.h; p.h_k = op.hk; p.h_h_k_ratio = op.h / op.hk;
  p.b = op.b;
  p.seqlen_q = op.sq; p.seqlen_k = op.sk;      // max_seqlen_q/k
  p.seqlen_q_rounded = (int32_t)(((int64_t)op.sq + 31) / 32 * 32);
  p.seqlen_k_rounded = (int32_t)(((int64_t)op.sk + 31) / 32 * 32);
  p.d = op.d; p.d_value = op.dv;
  p.d_rounded = (int32_t)d_r; p.d_value_rounded = (int32_t)dv_r;
  static int pre_cu = [] { const char* e = getenv("FA_PREFIX_CU"); return e ? atoi(e) : 1; }();
  if (pre_cu) {
    p.cu_seqlens_q = (int32_t*)buffers[5];
  }
  p.cu_seqlens_k = (int32_t*)buffers[4];
  static int pre_mtp = [] { const char* e = getenv("FA_PREFIX_MTP"); return e ? atoi(e) : -999; }();
  if (pre_mtp != -999) p.mtp = pre_mtp;
  static int pre_bt = [] { const char* e = getenv("FA_PREFIX_BT"); return e ? atoi(e) : 1; }();
  if (!pre_bt) { p.block_table = nullptr; p.block_table_batch_stride = 0; }
  static int pre_seq = [] { const char* e = getenv("FA_PREFIX_SEQ"); return e ? atoi(e) : 1; }();
  if (!pre_seq) { p.seqused_k = nullptr; p.is_seqlens_k_cumulative = false; }
  static int pre_cum = [] { const char* e = getenv("FA_PREFIX_CUM"); return e ? atoi(e) : 0; }();
  if (pre_cum) p.is_seqlens_k_cumulative = true;
  static int pre_split = [] { const char* e = getenv("FA_PREFIX_SPLIT"); return e ? atoi(e) : -1; }();
  if (pre_split >= 0) { p.num_splits = pre_split; p.partition_size = pre_split ? 128 : 0; }
  static int pre_page = [] { const char* e = getenv("FA_PREFIX_PAGE"); return e ? atoi(e) : -1; }();
  if (pre_page >= 0) p.page_block_size = pre_page;
  static int pre_lse = [] { const char* e = getenv("FA_PREFIX_LSE"); return e ? atoi(e) : 1; }();
  if (!pre_lse) p.softmax_lse_ptr = nullptr;
  static int pre_b = [] { const char* e = getenv("FA_PREFIX_B"); return e ? atoi(e) : -1; }();
  if (pre_b >= 0) p.b = pre_b;
  static int pre_tq = [] { const char* e = getenv("FA_PREFIX_TQ"); return e ? atoi(e) : -1; }();
  if (pre_tq >= 0) p.total_q = pre_tq;
  static int pre_msq = [] { const char* e = getenv("FA_PREFIX_MSQ"); return e ? atoi(e) : -1; }();
  if (pre_msq >= 0) { p.mtp = pre_msq; }
  p.seqused_k = (int32_t*)buffers[4];
  p.is_seqlens_k_cumulative = false;
  p.total_q = op.total_q;
  p.total_k = op.total_k;
  p.q_batch_stride = 0; p.o_batch_stride = 0;
  p.q_row_stride = (int32_t)(op.h * op.d);
  p.o_row_stride = (int32_t)(op.h * op.dv);
  p.q_head_stride = (int32_t)op.d; p.o_head_stride = (int32_t)op.dv;
  p.k_batch_stride = (int32_t)(page * op.hk * op.d);
  p.k_row_stride = (int32_t)(op.hk * op.d);
  p.k_head_stride = (int32_t)op.d;
  p.v_batch_stride = (int32_t)(page * op.hk * op.dv);
  p.v_row_stride = (int32_t)(op.hk * op.dv);
  p.v_head_stride = (int32_t)op.dv;
  p.v_dim_stride = 0;
  p.block_table = (int32_t*)buffers[3];
  p.block_table_batch_stride = op.bt_stride;
  p.page_block_size = page;
  p.softmax_lse_ptr = buffers[9];
  p.unpadded_lse = true;                         // lse (h,total_q)
  p.mtp = op.mtp_sq > 0 ? op.mtp_sq : op.sq;     // 源码: params.mtp = max_seqlen_q
  p.ngroups = 0;
  p.scale_softmax = op.scale;
  p.scale_softmax_log2 = op.scale * 1.4426950408889634f;
  p.softcap = 0.0f;
  p.p_dropout = 1.0f; p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = op.scale;
  p.p_dropout_in_uint8_t = 255;
  const FaWinCausal wc = clamp_window_causal(op.wl, op.wr, (int32_t)op.sk,
                                            op.causal != 0, false);
  p.is_causal = wc.is_causal;
  p.window_size_left = wc.wl; p.window_size_right = wc.wr;
  p.arch = 936;
  p.layout = 1;
  p.is_bf16 = (op.dtype == 1);
  p.cu_count = cu_count_of_current_device();
  static const char* dbg2 = getenv("FA_DEBUG");
  if (dbg2) {
    printf("[fa-prefix] b=%d h=%d hk=%d msq=%d msk=%d d=%d dv=%d page=%d total_q=%d "
           "bt_stride=%d causal=%d wl=%d wr=%d q=%p kc=%p vc=%p bt=%p seq=%p cu=%p "
           "o=%p lse=%p\n", op.b, op.h, op.hk, op.sq, op.sk, op.d, op.dv, page,
           op.total_q, op.bt_stride, op.causal, p.window_size_left,
           p.window_size_right, p.q_ptr, p.k_ptr, p.v_ptr, (void*)p.block_table,
           (void*)p.seqused_k, (void*)p.cu_seqlens_q, p.o_ptr,
           p.softmax_lse_ptr);
  }
  if (fa_lib().ok) fa_lib().fwd(p, stream, false);
}

// 此处保留 MLA-prefix 的底层布局；公开入口直接拒绝调用，
// 不能以目标已注册或符号存在推断功能可用。
// sq/sk/num_splits 槽分别承载 total_q/max_seqlen_q/is_mtp。
// scores_mem 分为 max、sum、LSE 三段，每段 h*total_q 个 float；
// 此保留路径的 LSE 写入工作区第三段，不是声明的结果槽 buffers[10]。
struct FaMlaOpaque {
  int32_t b, h, hk, sq;
  int32_t sk;
  float scale;
  int32_t dtype;             // 0 fp16 1 bf16
  int32_t causal;
  int32_t page;
  int32_t bt_stride;
  int32_t num_splits;
  int32_t partition_size;
};

// ================ MLA prefix-prefill (chunked prefill): legacy handler ========
static void FaMlaPrefixLegacy(hipStream_t stream, void** buffers,
                              const FaMlaOpaque& op) {
  Flash_fwd_mla_params p;
  memset(&p, 0, sizeof(p));
  p.q_ptr = buffers[0];
  p.qv_ptr = buffers[1];
  p.k_ptr = buffers[2];
  p.v_ptr = buffers[3];
  p.o_ptr = buffers[9];
  p.block_table = (int32_t*)buffers[4];
  p.block_table_batch_stride = op.bt_stride;
  p.page_block_size = op.page;
  p.cu_seqlens_q = (int32_t*)buffers[6];
  p.cu_seqlens_k_new = (int32_t*)buffers[7];
  p.cu_seqlens_k = (int32_t*)buffers[5];      // cache_seqlens (现有 KV 长度)
  {
    float* sm = (float*)buffers[8];           // (3, qheads, total_q): max|sum|lse
    const int64_t per = (int64_t)op.h * op.sq;
    p.scores_max_ptr = sm;
    p.scores_sum_ptr = sm + per;
    p.softmax_lse_ptr = sm + 2 * per;
  }
  p.b = op.b; p.h = op.h; p.h_k = op.hk;
  p.h_h_k_ratio = op.h / op.hk;
  p.d = 576; p.d_v = 512;
  p.total_q = op.sq;                          // op.sq 槽位 = total_q
  p.seqlen_q = op.sk;                         // op.sk 槽位 = max_seqlen_q
  p.scale_softmax = op.scale;
  p.scale_softmax_log2 = op.scale * 1.4426950408889634f;
  p.is_causal = op.causal != 0;
  p.mtp = op.num_splits;                      // 复用槽位承载 is_mtp
  p.q_row_stride = op.h * 576;  p.q_head_stride = 576;   // q (total_q,h,576)
  p.q_batch_stride = 0;
  p.qv_row_stride = op.h * 512; p.qv_head_stride = 512;
  p.qv_batch_stride = 0;
  p.k_batch_stride = op.page * op.hk * 576;
  p.k_row_stride = op.hk * 576; p.k_head_stride = 576;
  p.v_batch_stride = op.page * op.hk * 512;
  p.v_row_stride = op.hk * 512; p.v_head_stride = 512;
  p.o_row_stride = op.h * 512; p.o_head_stride = 512;
  p.o_batch_stride = 0;
  p.layout = 1;
  p.is_bf16 = (op.dtype == 1);
  p.cu_count = cu_count_of_current_device();
  static const char* dbg = getenv("FA_DEBUG");
  if (dbg) {
    printf("[fa-mla-prefix] q=%p qv=%p kc=%p vc=%p (b=%d h=%d hk=%d total_q=%d msq=%d "
           "page=%d is_mtp=%d causal=%d)\n", p.q_ptr, p.qv_ptr, p.k_ptr, p.v_ptr,
           p.b, p.h, p.h_k, p.total_q, p.seqlen_q, p.page_block_size, p.mtp,
           (int)p.is_causal);
  }
  if (const char* e = fa_lib_check("run_fwd_prefix_prefill_mla", (const void*)fa_lib().prefix_mla)) { fprintf(stderr, "[fa_attention] %s\n", e); return; }
  fa_lib().prefix_mla(p, stream);
}

extern "C" void fa_mla_prefix_v0(void* stream, void** buffers, const char* opaque,
                                 size_t opaque_len, void* /*status*/) {
  if (opaque_len < sizeof(FaMlaOpaque)) return;
  FaMlaOpaque op;
  memcpy(&op, opaque, sizeof(op));
  FaMlaPrefixLegacy((hipStream_t)stream, buffers, op);
}

extern "C" void fa_prefix_v0(void* stream, void** buffers, const char* opaque,
                             size_t opaque_len, void* /*status*/) {
  if (opaque_len < sizeof(FaPaOpaque)) return;
  FaPaOpaque op;
  memcpy(&op, opaque, sizeof(op));
  FaPrefixLegacy((hipStream_t)stream, buffers, op);
}

static void FaPaLegacy(hipStream_t stream, void** buffers, const FaPaOpaque& op) {
  const int res_base = 11;
  const int64_t d_r = ((int64_t)op.d + 31) / 32 * 32;
  const int64_t dv_r = (((int64_t)op.dv + 7) / 8 * 8 + 31) / 32 * 32;
  const int32_t page = op.page_block_size;

  Flash_fwd_params p;
  memset(&p, 0, sizeof(p));
  p.q_ptr = buffers[0]; p.k_ptr = buffers[1]; p.v_ptr = buffers[2];
  p.o_ptr = buffers[res_base];
  p.h = op.h; p.h_k = op.hk; p.h_h_k_ratio = op.h / op.hk;
  p.b = op.b;
  p.seqlen_q = op.sq; p.seqlen_k = op.sk;
  p.seqlen_q_rounded = (int32_t)(((int64_t)op.sq + 31) / 32 * 32);
  p.seqlen_k_rounded = (int32_t)(((int64_t)op.sk + 31) / 32 * 32);
  p.d = op.d; p.d_value = op.dv;
  p.d_rounded = (int32_t)d_r; p.d_value_rounded = (int32_t)dv_r;
  p.q_batch_stride = (int32_t)(op.sq * op.h * op.d);
  p.q_row_stride = (int32_t)(op.h * op.d);   p.q_head_stride = (int32_t)op.d;
  p.o_batch_stride = (int32_t)(op.sq * op.h * op.dv);
  p.o_row_stride = (int32_t)(op.h * op.dv);  p.o_head_stride = (int32_t)op.dv;
  p.k_batch_stride = (int32_t)(page * op.hk * op.d);
  p.k_row_stride = (int32_t)(op.hk * op.d);  p.k_head_stride = (int32_t)op.d;
  p.v_batch_stride = (int32_t)(page * op.hk * op.dv);
  p.v_row_stride = (int32_t)(op.hk * op.dv); p.v_head_stride = (int32_t)op.dv;
  p.v_dim_stride = 0;
  p.block_table = (int32_t*)buffers[3];
  p.block_table_batch_stride = op.bt_stride;
  p.page_block_size = page;
  p.cu_seqlens_k = (int32_t*)buffers[4];
  p.is_seqlens_k_cumulative = false;
  p.seqused_k = nullptr;
  p.softmax_lse_ptr = nullptr;    // PA 路径无 LSE 输出
  p.unpadded_lse = true;
  p.mtp = op.mtp;
  p.ngroups = 0;   // 与 torch mha_fwd_kvcache_base 一致 (该字段仅 varlen/prefix 变体设置)
  // split-kv accum (num_splits<=1 时内核不使用这些指针)
  {
    float* sc = (float*)buffers[5];
    const int64_t per = (int64_t)op.num_splits * op.b * op.h * op.sq;
    p.scores_sum_ptr = sc;
    p.scores_max_ptr = (sc && op.num_splits > 0) ? sc + per : sc;
  }
  p.softmax_lseaccum_ptr = nullptr;   // 与 torch kvcache 路径一致 (该方案不用)
  p.oaccum_ptr = buffers[6];
  p.num_splits = op.num_splits;
  p.partition_size = op.partition_size;
  p.splitkv_use_fp32_as_accum = false;   // oaccum 与 q 同 dtype (torch 启发式路径)
  // scale / dropout: 无 dropout = params.p_dropout 存保留概率 1
  p.scale_softmax = op.scale;
  p.scale_softmax_log2 = op.scale * 1.4426950408889634f;
  p.softcap = 0.0f;                       // softcap 由 wrapper 侧守卫拒绝
  p.p_dropout = 1.0f; p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = op.scale;
  p.p_dropout_in_uint8_t = 255;
  // 窗口 / causal (厂商规则见 clamp_window_causal): sq==1 且无 alibi → 强制非 causal;
  // causal → 先令 wr=0; 夹紧前判定 is_causal
  const FaWinCausal wc = clamp_window_causal(op.wl, op.wr, (int32_t)op.sk,
                                            op.causal != 0, op.sq == 1);
  p.is_causal = wc.is_causal;
  p.window_size_left = wc.wl;
  p.window_size_right = wc.wr;
  p.arch = fa_arch_raw();
  p.layout = 1;
  p.is_bf16 = (op.dtype == 1);
  if (op.is_fp8 == 2) {
    // INT8 PA: is_int8 选内核分支 (run_int8_fwd_splitkv_dispatch, d 仅 128);
    // scales 先按 (b,h) f32 传入。
    p.is_int8 = true;
    p.scales_q_ptr = (float*)buffers[8];
    p.scales_k_ptr = (float*)buffers[9];
    p.scales_v_ptr = (float*)buffers[10];
    p.total_scale_q = op.total_q;   // scales_q 元素数 (标定)
    static const char* idbg = getenv("FA_DEBUG");
    if (idbg) {
      printf("[fa-pa-int8] arch=%d sq=%p sk=%p sv=%p total_scale_q=%d is_bf16=%d\n",
             p.arch, (void*)p.scales_q_ptr, (void*)p.scales_k_ptr,
             (void*)p.scales_v_ptr, p.total_scale_q, (int)p.is_bf16);
    }
  } else if (op.is_fp8 == 1) {
    // FP8 PA: is_e4m3 选内核分支; descale f32 连续 (b,h), stride=(h,1);
    // 要求 d==dv==128 (wrapper 已守卫)。
    p.is_e4m3 = true;
    p.q_descale_ptr = (float*)buffers[8];
    p.k_descale_ptr = (float*)buffers[9];
    p.v_descale_ptr = (float*)buffers[10];
    p.q_descale_batch_stride = op.total_q;
    p.q_descale_head_stride = op.total_k;
    p.k_descale_batch_stride = op.total_q;
    p.k_descale_head_stride = op.total_k;
    p.v_descale_batch_stride = op.total_q;
    p.v_descale_head_stride = op.total_k;
    static const char* fdbg = getenv("FA_DEBUG");
    if (fdbg) {
      printf("[fa-pa-fp8] arch=%d dq=%p dk=%p dv=%p stride=(%d,%d) is_bf16=%d\n",
             p.arch, (void*)p.q_descale_ptr, (void*)p.k_descale_ptr,
             (void*)p.v_descale_ptr, p.q_descale_batch_stride,
             p.q_descale_head_stride, (int)p.is_bf16);
    }
  }
  p.cu_count = cu_count_of_current_device();
  if (op.is_fp8 == 2) {
    // INT8 PA: dispatch 须走 run_int8_fwd_kvcache。
    if (const char* e = fa_lib_check("run_int8_fwd_kvcache", (const void*)fa_lib().int8_kvcache)) { fprintf(stderr, "[fa_attention] %s\n", e); return; }
    fa_lib().int8_kvcache(p, stream, /*force_split_kernel=*/true);
  } else {
    if (const char* e = fa_lib_check("run_mha_fwd_kvcache", (const void*)fa_lib().kvcache)) { fprintf(stderr, "[fa_attention] %s\n", e); return; }
    fa_lib().kvcache(p, stream, /*force_split_kernel=*/true);
  }
}

extern "C" void fa_fwd_kvcache_v0(void* stream, void** buffers, const char* opaque,
                                  size_t opaque_len, void* /*status*/) {
  if (opaque_len < sizeof(FaPaOpaque)) return;
  FaPaOpaque op;
  memcpy(&op, opaque, sizeof(op));
  FaPaLegacy((hipStream_t)stream, buffers, op);
}


// ---------------------------- bwd: legacy handler ---------------------------
// Q/K 的长度、容量与步长分别计算，不能假定二者相等。
// 变长累计终点可以小于张量容量；O、LSE 和梯度先按完整容量清零。
// LSE 头间隔使用 capacity_q，不能用累计终点替代。
// 反向序列行数按 128 对齐，前向为 32；dsoftmax_sum 为 (b,h,round_up(sq,128))。
// 稠密梯度的物理行数和批/头步长使用对齐长度，不能沿用输入的紧凑步长。
// dk/dv 按查询头展开，GQA 的分组归并由调用方完成。
// 公共缩放、窗口、dropout 和布局参数须在变长分支提前返回前完成初始化。
// 确定性 dq_accum 为 fp32，每次发射前清零；分片容量、步长和数量须与发射器一致。
static void FaBwdLegacy(hipStream_t stream, void** buffers, const FaOpaque& op) {
  fa_native_error = nullptr;
  auto rm = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
  const int64_t sq_r = rm(op.sq, 128), sk_r = rm(op.sk, 128);   // bwd: 128 对齐
  const int64_t d = rm(op.d, 8), d_r = rm(d, 32);
  const int64_t dv = rm(op.dv, 8), dv_r = rm(dv, 32);
  const bool gqa = op.h != op.hk;

  Flash_bwd_params p;
  memset(&p, 0, sizeof(p));
  p.q_ptr = buffers[0]; p.k_ptr = buffers[1]; p.v_ptr = buffers[2];
  p.o_ptr = buffers[3];
  p.do_ptr = buffers[4];
  p.softmax_lse_ptr = buffers[5];
  int32_t* workspace = acquire_workspace(stream);
  if (!workspace) return;
  p.rng_state = reinterpret_cast<uint64_t*>(workspace);
  p.dropout_debug_count = reinterpret_cast<uint32_t*>(workspace + 4);
  p.tile_count_semaphore = workspace + 8;
  p.alibi_slopes_ptr = op.has_alibi ? buffers[9] : nullptr;
  p.alibi_slopes_batch_stride = op.alibi_batch ? (int32_t)op.h : 0;
  p.s_aux_ptr = op.has_sinks ? buffers[10] : nullptr;
  p.s_aux_type = op.sink_type;
  const int res_base = op.has_varlen ? 13 : 11;
  if (op.has_varlen) {   // varlen 的两个 cu_seqlens 是操作数, 排在结果之前
    p.cu_seqlens_q = static_cast<int32_t*>(buffers[11]);
    p.cu_seqlens_k = static_cast<int32_t*>(buffers[12]);
  }
  p.dq_ptr = buffers[res_base];
  p.dk_ptr = buffers[res_base + 1];
  p.dv_ptr = buffers[res_base + 2];
  p.dsoftmax_sum = buffers[res_base + 3];
  if (op.has_varlen) {
    ResolveHipMem();
    if (!hip_memset_async ||
        hip_memset_async(p.dq_ptr, 0, int64_t(op.total_q) * op.h * op.d * 2, stream) ||
        hip_memset_async(p.dk_ptr, 0, int64_t(op.total_k) * op.h * d_r * 2, stream) ||
        hip_memset_async(p.dv_ptr, 0, int64_t(op.total_k) * op.h * dv_r * 2, stream) ||
        hip_memset_async(p.dsoftmax_sum, 0, int64_t(op.b) * op.h * sq_r * 4, stream)) {
      fa_native_error = "fa: varlen gradient initialization failed";
      fprintf(stderr, "[fa_attention] %s\n", fa_native_error);
      return;
    }
  }
  p.dq_accum_ptr = nullptr;   // 非确定性路径不使用 (deterministic 分支见下)
  p.dk_accum_ptr = nullptr; p.dv_accum_ptr = nullptr;
  p.dq_accum_split_stride = 0;
  p.se_balance_cnt = 0;
  p.deterministic = false;
  {
    static int det_env = [] { const char* e = getenv("FA_BWD_DET"); return e ? atoi(e) : 0; }();
    if (op.deterministic || det_env) {
      const int num_sm = cu_count_of_current_device();
      const int nsplits = (num_sm + op.b * op.h - 1) / (op.b * op.h);
      const int64_t per_split = (int64_t)op.b * sq_r * op.h * d_r;
      const int64_t bytes = per_split * nsplits * 4;
      bool ok = false;
      void* acc = acquire_accum(bytes, (hipStream_t)stream, &ok);
      if (getenv("FA_DEBUG"))
        printf("[fa-det] num_sm=%d b=%d h=%d nsplits=%d per_split=%ld bytes=%ld "
               "acc=%p ok=%d\n", num_sm, op.b, op.h, nsplits, (long)per_split,
               (long)bytes, acc, (int)ok);
      if (!acc || !ok) {
        fa_native_error = "fa: deterministic gradient workspace allocation failed";
        return;
      }
      if (acc) {
        p.dq_accum_ptr = acc;
        p.dq_accum_split_stride = (int32_t)per_split;
        p.deterministic = true;
        p.num_splits = nsplits;
      }
    } else if (getenv("FA_DEBUG")) {
      printf("[fa-det] FA_BWD_DET 未设置 → 非确定性路径\n");
    }
  }
  p.v_dim_stride = 0;
  p.h = op.h; p.h_k = op.hk; p.h_h_k_ratio = op.h / op.hk;

  if (op.layout == 1) {   // (b, s, h, d)
    p.q_batch_stride = (int32_t)(op.sq * op.h * op.d);
    p.k_batch_stride = (int32_t)(op.sk * op.hk * op.d);
    p.v_batch_stride = (int32_t)(op.sk * op.hk * dv);
    p.o_batch_stride = (int32_t)(op.sq * op.h * dv);
    p.do_batch_stride = (int32_t)(op.sq * op.h * dv);
    // 输出缓冲行数按对齐长度计; varlen 的输出是 (total_q,h,d) 紧凑布局, 保持原样。
    const int64_t dq_rows = op.has_varlen ? op.sq : sq_r;
    const int64_t dkv_rows = op.has_varlen ? op.sk : sk_r;
    p.dq_batch_stride = (int32_t)(dq_rows * op.h * op.d);
    p.dk_batch_stride = (int32_t)(dkv_rows * (gqa ? op.h : op.hk) * d_r);
    p.dv_batch_stride = (int32_t)(dkv_rows * (gqa ? op.h : op.hk) * dv_r);
    p.q_row_stride = (int32_t)(op.h * op.d);   p.q_head_stride = (int32_t)op.d;
    p.k_row_stride = (int32_t)(op.hk * op.d);  p.k_head_stride = (int32_t)op.d;
    p.v_row_stride = (int32_t)(op.hk * dv);    p.v_head_stride = (int32_t)dv;
    p.o_row_stride = (int32_t)(op.h * dv);     p.o_head_stride = (int32_t)dv;
    p.do_row_stride = (int32_t)(op.h * dv);    p.do_head_stride = (int32_t)dv;
    p.dq_row_stride = (int32_t)(op.h * op.d);  p.dq_head_stride = (int32_t)op.d;
    p.dk_row_stride = (int32_t)(op.h * d_r);   p.dk_head_stride = (int32_t)d_r;
    p.dv_row_stride = (int32_t)(op.h * dv_r);  p.dv_head_stride = (int32_t)dv_r;
  } else {                // layout=0, (b, h, s, d)
    p.q_batch_stride = (int32_t)(op.h * op.sq * op.d);
    p.k_batch_stride = (int32_t)(op.hk * op.sk * op.d);
    p.v_batch_stride = (int32_t)(op.hk * op.sk * dv);
    p.o_batch_stride = (int32_t)(op.h * op.sq * dv);
    p.do_batch_stride = (int32_t)(op.h * op.sq * dv);
    const int64_t dq_rows0 = op.has_varlen ? op.sq : sq_r;
    const int64_t dkv_rows0 = op.has_varlen ? op.sk : sk_r;
    p.dq_batch_stride = (int32_t)(op.h * dq_rows0 * op.d);
    p.dk_batch_stride = (int32_t)((gqa ? op.h : op.hk) * dkv_rows0 * d_r);
    p.dv_batch_stride = (int32_t)((gqa ? op.h : op.hk) * dkv_rows0 * dv_r);
    p.q_row_stride = (int32_t)op.d;   p.q_head_stride = (int32_t)(op.sq * op.d);
    p.k_row_stride = (int32_t)op.d;   p.k_head_stride = (int32_t)(op.sk * op.d);
    p.v_row_stride = (int32_t)dv;     p.v_head_stride = (int32_t)(op.sk * dv);
    p.o_row_stride = (int32_t)dv;     p.o_head_stride = (int32_t)(op.sq * dv);
    p.do_row_stride = (int32_t)dv;    p.do_head_stride = (int32_t)(op.sq * dv);
    p.dq_row_stride = (int32_t)op.d;  p.dq_head_stride = (int32_t)(dq_rows0 * op.d);
    p.dk_row_stride = (int32_t)d_r;   p.dk_head_stride = (int32_t)(dkv_rows0 * d_r);
    p.dv_row_stride = (int32_t)dv_r;  p.dv_head_stride = (int32_t)(dkv_rows0 * dv_r);
  }

  p.b = op.b; p.seqlen_q = op.sq; p.seqlen_k = op.sk;
  p.d = (int32_t)d; p.d_value = (int32_t)dv;
  p.seqlen_q_rounded = (int32_t)sq_r;
  p.seqlen_k_rounded = (int32_t)sk_r;
  static int vbwd_qkv_mode = [] { const char* e = getenv("FA_VBWD_QKV"); return e ? atoi(e) : 0; }();
  static int vbwd_layout = [] { const char* e = getenv("FA_VBWD_LAYOUT"); return e ? atoi(e) : -1; }();
  p.d_rounded = (int32_t)d_r;
  p.d_value_rounded = (int32_t)rm(dv, 32);
  // softcap 契约与前向一致: 即使当前库禁用 softcap, 也补完整反向参数链路。
  if (op.softcap > 0.0f) {
    p.softcap = op.scale / op.softcap;
    p.scale_softmax = op.softcap;
    p.scale_softmax_log2 = op.softcap * 1.4426950408889634f;
  } else {
    p.softcap = 0.0f;
    p.scale_softmax = op.scale;
    p.scale_softmax_log2 = op.scale * 1.4426950408889634f;
  }
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = p.scale_softmax;
  const FaWinCausal wc = clamp_window_causal(op.window_left, op.window_right,
                                            (int32_t)op.sk, op.causal != 0,
                                            op.sq == 1 && !op.has_alibi);
  p.is_causal = wc.is_causal;
  p.window_size_left = wc.wl;
  p.window_size_right = wc.wr;
  p.num_splits = 1;
  p.partition_size = 0;
  p.arch = 936;
  p.layout = op.layout;
  p.is_bf16 = (op.dtype == 1);
  p.p_dropout = 1.0f;              // 保留概率 (无 dropout)
  p.p_dropout_in_uint8_t = 0;
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = op.scale;
  apply_dropout(p, op.dropout_p, buffers[6], stream);
  if (fa_native_error) return;
  static int bis_nosem = [] { const char* e = getenv("FA_VBWD_NOSEM"); return e ? atoi(e) : 0; }();
  static int bis_semval = [] { const char* e = getenv("FA_VBWD_SEMVAL"); return e ? atoi(e) : -1; }();
  if (bis_nosem) {
    p.tile_count_semaphore = nullptr;   // torch 的 bwd 不设该字段 (memset 后为 NULL)
  } else if (bis_semval >= 0) {
    p.tile_count_semaphore = workspace + 8;
    zero_sem(p.tile_count_semaphore, stream);
  } else {
    zero_sem(p.tile_count_semaphore, stream);
  }
  if (op.has_varlen) {   // varlen 的 stride/seqlen/accum 覆盖上面的公共值
    // varlen: q (total_q,h,d) / k,v (total_k,hk,dv) 3-D; dk/dv 输出为
    // (total_k,h,d_r)/(total_k,h,dv_r) 展开布局, 由框架沿 group 轴求和。
    p.total_q = op.total_q;
    p.total_k = op.total_k;
    p.is_seqlens_k_cumulative = true;
    p.unpadded_lse = op.lse_unpadded != 0;
    if (op.vbwd_mode == 1) {
      // packed 约定: q 视作 (total*h, d) 二维, 行=头=d。
      p.q_batch_stride = (int32_t)op.d;
      p.o_batch_stride = (int32_t)dv;
      p.q_head_stride = (int32_t)op.d;   p.k_head_stride = (int32_t)op.d;
      p.v_head_stride = (int32_t)dv;     p.o_head_stride = (int32_t)dv;
      p.do_head_stride = (int32_t)dv;
      p.dq_head_stride = (int32_t)op.d;
      p.dk_head_stride = (int32_t)d_r;   p.dv_head_stride = (int32_t)dv_r;
      p.q_row_stride = (int32_t)op.d;
      p.k_row_stride = (int32_t)op.d;
      p.v_row_stride = (int32_t)dv;
      p.o_row_stride = (int32_t)dv;
      p.do_row_stride = (int32_t)dv;
      p.dq_row_stride = (int32_t)op.d;
      p.dk_row_stride = (int32_t)d_r;
      p.dv_row_stride = (int32_t)dv_r;
    } else {
      p.q_batch_stride = (int32_t)(op.h * op.d);
      p.o_batch_stride = (int32_t)(op.h * dv);
      p.q_head_stride = (int32_t)op.d;    p.k_head_stride = (int32_t)op.d;
      p.v_head_stride = (int32_t)dv;      p.o_head_stride = (int32_t)dv;
      p.do_head_stride = (int32_t)dv;
      p.dq_head_stride = (int32_t)op.d;
      p.dk_head_stride = (int32_t)d_r;    p.dv_head_stride = (int32_t)dv_r;
      p.q_row_stride = (int32_t)(op.h * op.d);
      p.k_row_stride = (int32_t)(op.hk * op.d);
      p.v_row_stride = (int32_t)(op.hk * dv);
      p.o_row_stride = (int32_t)(op.h * dv);
      p.do_row_stride = (int32_t)(op.h * dv);
      p.dq_row_stride = (int32_t)(op.h * op.d);
      p.dk_row_stride = (int32_t)(op.h * d_r);
      p.dv_row_stride = (int32_t)(op.h * dv_r);
    }
    if (vbwd_qkv_mode == 1) {          // bhsd 式: head=q.stride(0)=h*d, row 同
      p.q_head_stride = (int32_t)(op.h * op.d); p.k_head_stride = (int32_t)(op.hk * op.d);
      p.v_head_stride = (int32_t)(op.hk * dv);   p.o_head_stride = (int32_t)(op.h * dv);
      p.do_head_stride = (int32_t)(op.h * dv);   p.dq_head_stride = (int32_t)(op.h * op.d);
      p.dk_head_stride = (int32_t)(op.h * d_r);  p.dv_head_stride = (int32_t)(op.h * dv_r);
    } else if (vbwd_qkv_mode == 2) {   // packed: head=d, row=d
      p.q_row_stride = (int32_t)op.d; p.k_row_stride = (int32_t)op.d;
      p.v_row_stride = (int32_t)dv;   p.o_row_stride = (int32_t)dv;
      p.do_row_stride = (int32_t)dv;  p.dq_row_stride = (int32_t)op.d;
      p.dk_row_stride = (int32_t)d_r; p.dv_row_stride = (int32_t)dv_r;
    } else if (vbwd_qkv_mode == 3) {   // head=h*d, row=d
      p.q_head_stride = (int32_t)(op.h * op.d); p.k_head_stride = (int32_t)(op.hk * op.d);
      p.v_head_stride = (int32_t)(op.hk * dv);   p.o_head_stride = (int32_t)(op.h * dv);
      p.do_head_stride = (int32_t)(op.h * dv);   p.dq_head_stride = (int32_t)(op.h * op.d);
      p.dk_head_stride = (int32_t)(op.h * d_r);  p.dv_head_stride = (int32_t)(op.h * dv_r);
      p.q_row_stride = (int32_t)op.d; p.k_row_stride = (int32_t)op.d;
      p.v_row_stride = (int32_t)dv;   p.o_row_stride = (int32_t)dv;
      p.do_row_stride = (int32_t)dv;  p.dq_row_stride = (int32_t)op.d;
      p.dk_row_stride = (int32_t)d_r; p.dv_row_stride = (int32_t)dv_r;
    }
    if (vbwd_layout >= 0) p.layout = vbwd_layout;
    static int cal_seq = [] { const char* e = getenv("FA_VBWD_SEQ"); return e ? atoi(e) : 0; }();
    static int cal_b = [] { const char* e = getenv("FA_VBWD_B"); return e ? atoi(e) : -1; }();
    if (cal_seq == 1) { p.seqlen_q = op.total_q; p.seqlen_k = op.total_k; }
    else if (cal_seq == 2) {
      p.seqlen_q_rounded = (int32_t)(((int64_t)op.total_q + 127) / 128 * 128);
      p.seqlen_k_rounded = (int32_t)(((int64_t)op.total_k + 127) / 128 * 128);
    } else if (cal_seq == 3) { p.seqlen_q = op.total_q; p.seqlen_k = op.total_k;
      p.seqlen_q_rounded = (int32_t)(((int64_t)op.total_q + 127) / 128 * 128);
      p.seqlen_k_rounded = (int32_t)(((int64_t)op.total_k + 127) / 128 * 128); }
    if (cal_b >= 0) p.b = cal_b;
    static int cal_ostr = [] { const char* e = getenv("FA_VBWD_OSTR"); return e ? atoi(e) : 0; }();
    if (cal_ostr == 1) {          // packed: row=head=dv, batch=0
      p.o_row_stride = (int32_t)dv; p.o_head_stride = (int32_t)dv;
      p.do_row_stride = (int32_t)dv; p.do_head_stride = (int32_t)dv;
      p.o_batch_stride = 0; p.do_batch_stride = 0;
    } else if (cal_ostr == 2) {   // row=h*dv, head=dv, batch=0
      p.o_batch_stride = 0; p.do_batch_stride = 0;
    } else if (cal_ostr == 3) {   // row=dv, head=h*dv, batch=0
      p.o_row_stride = (int32_t)dv; p.o_head_stride = (int32_t)(op.h * dv);
      p.do_row_stride = (int32_t)dv; p.do_head_stride = (int32_t)(op.h * dv);
      p.o_batch_stride = 0; p.do_batch_stride = 0;
    } else if (cal_ostr == 4) {   // row=h*dv, head=dv, batch=tq*h*dv
      p.o_batch_stride = (int32_t)(op.total_q * op.h * dv);
      p.do_batch_stride = (int32_t)(op.total_q * op.h * dv);
    }
    static int cal_tot = [] { const char* e = getenv("FA_VBWD_TOTALOVR"); return e ? atoi(e) : -1; }();
    if (cal_tot >= 0) { p.total_q = cal_tot; p.total_k = cal_tot; }
    static int bis_batch = [] { const char* e = getenv("FA_VBWD_BATCH"); return e ? atoi(e) : 0; }();
    if (bis_batch) {   // dense 式 batch stride (标定)
      p.q_batch_stride = (int32_t)(op.sq * op.h * op.d);
      p.o_batch_stride = (int32_t)(op.sq * op.h * dv);
      p.do_batch_stride = (int32_t)(op.sq * op.h * dv);
      p.dq_batch_stride = (int32_t)(op.sq * op.h * op.d);
      p.dk_batch_stride = (int32_t)(op.sk * op.h * d_r);
      p.dv_batch_stride = (int32_t)(op.sk * op.h * dv_r);
    }
    static const char* acc_env = getenv("FA_VBWD_ACCUM");
    static int ns_env = [] { const char* e = getenv("FA_VBWD_NS"); return e ? atoi(e) : -1; }();
    if (ns_env >= 0) p.num_splits = ns_env;
    ResourceHolder<2> acc;   // 见下方实现: 分配/回读/释放
    if (acc_env && acc_env[0] == '1') {
      ResolveHipMem();
      const int64_t nq = (int64_t)op.total_q * op.h * d_r;
      const int64_t nk = (int64_t)op.total_k * op.h * d_r;
      const int64_t nv = (int64_t)op.total_k * op.h * dv_r;
      if (hip_malloc) {
        hip_malloc(&acc.ptr[0], nq * 4); hip_malloc(&acc.ptr[1], nk * 4);
        acc.n[0] = nq; acc.n[1] = nk;
        p.dq_accum_ptr = acc.ptr[0]; p.dk_accum_ptr = acc.ptr[1];
        p.dv_accum_ptr = acc.ptr[1];   // dk/dv 共用观察 (仅标定)
        p.dq_accum_split_stride = 0;
        printf("[fa] varlen bwd: 已分配 dq/dk_accum (%ld,%ld 元素)\n", (long)nq, (long)nk);
      }
    }
    // configure 预跑负责填 num_splits / dq_accum_split_stride /
    // se_balance_cnt 等 varlen 调度字段。
    static int cal_cfg = [] { const char* e = getenv("FA_VBWD_CONFIGURE"); return e ? atoi(e) : 0; }();
    if (cal_cfg) {
      if (fa_lib().ok) fa_lib().bwd(p, stream, /*configure=*/true);  // configure 预跑, 检查在正式发射处
      printf("[fa] varlen bwd: configure 后 num_splits=%d partition_size=%d "
             "dq_accum_split_stride=%d se_balance_cnt=%d\n", p.num_splits,
             p.partition_size, p.dq_accum_split_stride, p.se_balance_cnt);
    }
    maybe_write_magic(p, stream);   // 先埋魔数再发射 (同流, 判据才干净)
    if (const char* e = fa_lib_check("run_mha_bwd", (const void*)fa_lib().bwd)) { fprintf(stderr, "[fa_attention] %s\n", e); return; }
    fa_lib().bwd(p, stream, false);
    if (acc.ptr[0]) {   // 回读: 内核是否写了 accum
      float* host = (float*)malloc(acc.n[0] * 4);
      if (host && hip_memcpy) {
        hip_memcpy(host, acc.ptr[0], acc.n[0] * 4, 1);
        int nz = 0; float mx = 0.f;
        for (int64_t i = 0; i < acc.n[0]; i++) { if (host[i] != 0.f) { nz++; if (fabsf(host[i]) > mx) mx = fabsf(host[i]); } }
        printf("[fa] dq_accum 非零 %d/%ld, max=%g\n", nz, (long)acc.n[0], (double)mx);
        free(host);
      }
      if (hip_free) { hip_free(acc.ptr[0]); hip_free(acc.ptr[1]); }
    }
    return;
  }
  maybe_write_magic(p, stream);   // 先埋魔数再发射 (同流, 判据才干净)
  if (const char* e = fa_lib_check("run_mha_bwd", (const void*)fa_lib().bwd)) { fprintf(stderr, "[fa_attention] %s\n", e); return; }
  fa_lib().bwd(p, stream, false);
}

extern "C" void fa_bwd_v0(void* stream, void** buffers, const char* opaque, size_t opaque_len, void* /*status*/) {
  if (!opaque_ok(opaque_len, __func__)) return;
  FaOpaque op;
  memcpy(&op, opaque, sizeof(op));
  FaBwdLegacy((hipStream_t)stream, buffers, op);
}

// ====== 类型化 FFI handler（XLA_FFI_DEFINE_HANDLER） =========================
// 类型化 FFI 通过 XLA_FFI_DEFINE_HANDLER 绑定，参数适配后复用发射实现。
// 清零、异步拷贝和内核发射使用 XLA 提供的 HIP 流，不得改用默认流。
// 部分路径包含内存分配或主机侧流同步，未声明命令缓冲捕获兼容性。
namespace fa_ffi = xla::ffi;

// 在 XLA 流上回读 Q/K 长度并同步后校验，不能提前读取主机副本。
// 累计长度从 0 起、严格递增且终点不超过容量；逐序列模式允许 Q 长度为 0，
// K 长度须为正；require_equal 只约束非空 Q。
// 成功后返回同流上的设备副本，下游必须消费这些返回值，
// 才能保证校验先于注意力内核读取长度。
static fa_ffi::Error FaValidateLengthsTypedImpl(
    fa_ffi::Buffer<fa_ffi::S32> q_lengths,
    fa_ffi::Buffer<fa_ffi::S32> k_lengths,
    fa_ffi::Result<fa_ffi::Buffer<fa_ffi::S32>> checked_q,
    fa_ffi::Result<fa_ffi::Buffer<fa_ffi::S32>> checked_k,
    hipStream_t stream, int32_t q_capacity, int32_t k_capacity,
    int32_t max_q, int32_t max_k, int32_t cumulative, int32_t require_equal) {
  auto invalid = [](const char* text) {
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, text);
  };
  auto internal = [](const char* text) {
    return fa_ffi::Error(fa_ffi::ErrorCode::kInternal, text);
  };
  const auto qd = q_lengths.dimensions(), kd = k_lengths.dimensions();
  const auto oq = checked_q->dimensions(), ok = checked_k->dimensions();
  if (qd.size() != 1 || kd.size() != 1 || oq.size() != 1 || ok.size() != 1 ||
      qd[0] < (cumulative ? 2 : 1) || kd[0] != qd[0] ||
      oq[0] != qd[0] || ok[0] != qd[0] ||
      (cumulative != 0 && cumulative != 1) ||
      (require_equal != 0 && require_equal != 1))
    return invalid("fa lengths: expected matching int32 length vectors");
  if (q_capacity <= 0 || k_capacity <= 0 || max_q <= 0 || max_k <= 0 ||
      max_q > q_capacity || max_k > k_capacity)
    return invalid("fa lengths: invalid capacity or maximum sequence length");
  ResolveHipMem();
  if (!hip_memcpy_async || !hip_stream_sync)
    return internal("fa lengths: HIP copy/synchronize unavailable");
  std::vector<int32_t> qh(qd[0]), kh(qd[0]);
  const size_t bytes = qh.size() * sizeof(int32_t);
  int qe = hip_memcpy_async(qh.data(), q_lengths.untyped_data(), bytes,
                           hipMemcpyDeviceToHost, stream);
  int ke = hip_memcpy_async(kh.data(), k_lengths.untyped_data(), bytes,
                           hipMemcpyDeviceToHost, stream);
  int se = hip_stream_sync(stream);
  if (qe || ke || se) return internal("fa lengths: device length read failed");
  if (cumulative && (qh[0] != 0 || kh[0] != 0 ||
                     qh.back() > q_capacity || kh.back() > k_capacity))
    return invalid("fa lengths: cumulative lengths must start at 0 and fit capacity");
  for (size_t i = cumulative ? 1 : 0; i < qh.size(); ++i) {
    int64_t qn = cumulative ? int64_t(qh[i]) - qh[i - 1] : qh[i];
    int64_t kn = cumulative ? int64_t(kh[i]) - kh[i - 1] : kh[i];
    if (qn < (cumulative ? 1 : 0) || qn > max_q || kn < 1 || kn > max_k ||
        (require_equal && qn != 0 && qn != kn)) {
      char message[256];
      snprintf(message, sizeof(message),
               "fa lengths: invalid sequence %zu (query=%lld, key_value=%lld); "
               "check range and causal/window equal-length requirement",
               i - (cumulative ? 1 : 0), (long long)qn, (long long)kn);
      return invalid(message);
    }
  }
  qe = hip_memcpy_async(checked_q->untyped_data(), q_lengths.untyped_data(), bytes,
                       hipMemcpyDeviceToDevice, stream);
  ke = hip_memcpy_async(checked_k->untyped_data(), k_lengths.untyped_data(), bytes,
                       hipMemcpyDeviceToDevice, stream);
  if (qe || ke) return internal("fa lengths: checked length copy failed");
  return fa_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kFaValidateLengthsTyped, FaValidateLengthsTypedImpl,
    fa_ffi::Ffi::Bind().Arg<fa_ffi::Buffer<fa_ffi::S32>>()
        .Arg<fa_ffi::Buffer<fa_ffi::S32>>()
        .Ret<fa_ffi::Buffer<fa_ffi::S32>>()
        .Ret<fa_ffi::Buffer<fa_ffi::S32>>()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<int32_t>("q_capacity").Attr<int32_t>("k_capacity")
        .Attr<int32_t>("max_q").Attr<int32_t>("max_k")
        .Attr<int32_t>("cumulative").Attr<int32_t>("require_equal"));

extern "C" void* fa_validate_lengths_typed_handler() {
  return (void*)kFaValidateLengthsTyped;
}

static fa_ffi::Error FaAttnLengthTypedImpl(
    fa_ffi::AnyBuffer q, fa_ffi::AnyBuffer k, fa_ffi::AnyBuffer v,
    fa_ffi::Buffer<fa_ffi::S32> length,
    fa_ffi::Result<fa_ffi::AnyBuffer> out,
    fa_ffi::Result<fa_ffi::Buffer<fa_ffi::F32>> lse,
    fa_ffi::Result<fa_ffi::Buffer<fa_ffi::S32>> scratch,
    hipStream_t stream, float scale, int32_t layout, int32_t valid_k) {
  auto invalid = [](const char* text) {
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, text);
  };
  const auto qd = q.dimensions(), kd = k.dimensions(), vd = v.dimensions();
  if (qd.size() != 4 || kd.size() != 4 || vd.size() != 4 ||
      (layout != 0 && layout != 1))
    return invalid("fa_attn_length: q/k/v 须为 4-D, layout 须为 0/1");
  const int si = layout == 1 ? 1 : 2, hi = layout == 1 ? 2 : 1;
  const int64_t b = qd[0], sq = qd[si], sk = kd[si], h = qd[hi], hk = kd[hi];
  if (q.element_type() != fa_ffi::F16 && q.element_type() != fa_ffi::BF16)
    return invalid("fa_attn_length: 仅支持 fp16/bf16");
  if (k.element_type() != q.element_type() || v.element_type() != q.element_type() ||
      out->element_type() != q.element_type())
    return invalid("fa_attn_length: q/k/v/o dtype 必须一致");
  if (b <= 0 || sq <= 0 || sk <= 0 || h <= 0 || hk <= 0 || h % hk != 0 ||
      kd[0] != b || vd[0] != b || vd[si] != sk || vd[hi] != hk ||
      qd[3] != 128 || kd[3] != 128 || vd[3] != 128)
    return invalid("fa_attn_length: batch/KV shape/head 数不匹配或 head_dim != 128");
  if (q.element_count() >= (1ULL << 31) || k.element_count() >= (1ULL << 31) ||
      v.element_count() >= (1ULL << 31))
    return invalid("fa_attn_length: q/k/v 元素数必须小于 2^31");
  if (valid_k < 1 || valid_k > sk || !std::isfinite(scale) || scale <= 0)
    return invalid("fa_attn_length: 需要 1 <= valid_k <= sk 和有限正 scale");
  const auto od = out->dimensions(), ld = lse->dimensions();
  if (od.size() != 4 || ld.size() != 3 || ld[0] != b || ld[1] != h || ld[2] != sq)
    return invalid("fa_attn_length: 输出 shape 错误");
  for (int i = 0; i < 4; ++i)
    if (od[i] != qd[i]) return invalid("fa_attn_length: o shape 必须等于 q shape");
  if (length.dimensions().size() != 1 || length.dimensions()[0] != 1 ||
      scratch->dimensions().size() != 1 || scratch->dimensions()[0] != 72)
    return invalid("fa_attn_length: 需要 int32[1] 长度和 int32[72] scratch");
  if (const char* err = fa_lib_check("run_mha_fwd", (const void*)fa_lib().fwd))
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, err);
  ResolveHipMem();
  if (!hip_memset_async)
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, "hipMemsetAsync 不可用");
  if (hip_memset_async(scratch->untyped_data(), 0, scratch->size_bytes(), stream) != 0)
    return fa_ffi::Error(fa_ffi::ErrorCode::kInternal, "fa_attn_length: scratch 清零失败");
  // scratch 是 XLA 结果缓冲而非输入常量, 内核可写 rng/sem。
  auto* work = scratch->typed_data();
  void* buffers[] = {q.untyped_data(), k.untyped_data(), v.untyped_data(),
                    work, work + 4, work + 8, length.untyped_data(),
                    out->untyped_data(), lse->untyped_data()};
  FaOpaque op{};
  op.b = b; op.h = h; op.hk = hk; op.sq = sq; op.sk = sk;
  op.d = 128; op.dv = 128; op.scale = scale; op.layout = layout;
  op.dtype = q.element_type() == fa_ffi::BF16 ? 1 : 0;
  op.window_left = -1; op.window_right = -1; op.has_sinks = 1;
  FaFwdMaskLegacy(stream, buffers, op);
  if (fa_native_error) return fa_ffi::Error(fa_ffi::ErrorCode::kInternal, fa_native_error);
  return fa_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kFaAttnLengthTyped, FaAttnLengthTypedImpl,
    fa_ffi::Ffi::Bind().Arg<fa_ffi::AnyBuffer>().Arg<fa_ffi::AnyBuffer>()
        .Arg<fa_ffi::AnyBuffer>().Arg<fa_ffi::Buffer<fa_ffi::S32>>()
        .Ret<fa_ffi::AnyBuffer>().Ret<fa_ffi::Buffer<fa_ffi::F32>>()
        .Ret<fa_ffi::Buffer<fa_ffi::S32>>()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<float>("scale").Attr<int32_t>("layout").Attr<int32_t>("valid_k"));

extern "C" void* fa_attn_length_typed_handler() { return (void*)kFaAttnLengthTyped; }

// 由 attrs + 缓冲形状填充 FaOpaque（dense 全推、varlen 的 msq/msk 来自 attrs）
static const char* FillOpaqueFromShapes(FaOpaque& op, const int64_t* qd, int qr,
                                        const int64_t* kd, int kr,
                                        const int64_t* vd, int vr,
                                        int64_t cu_len) {
  int64_t H, HK, D, DV;
  if (op.has_varlen) {
    if (qr != 3 || kr != 3 || vr != 3) return "varlen 的 q/k/v 须为 3-D";
    op.total_q = (int32_t)qd[0];
    op.total_k = (int32_t)kd[0];
    H = qd[1]; HK = kd[1]; D = qd[2]; DV = vd[2];
    if (cu_len > 0) op.b = (int32_t)cu_len;
    if (op.sq <= 0 || op.sk <= 0)
      return "varlen 必须通过 msq/msk attrs 传 max_seqlen";
  } else if (op.layout == 1) {                      // (b, s, h, d)
    if (qr != 4 || kr != 4 || vr != 4) return "layout=1 的 q/k/v 须为 4-D";
    op.b = (int32_t)qd[0]; op.sq = (int32_t)qd[1]; H = qd[2]; D = qd[3];
    HK = kd[2]; op.sk = (int32_t)kd[1]; DV = vd[3];
  } else {                                          // (b, h, s, d)
    if (qr != 4 || kr != 4 || vr != 4) return "layout=0 的 q/k/v 须为 4-D";
    op.b = (int32_t)qd[0]; H = qd[1]; op.sq = (int32_t)qd[2]; D = qd[3];
    HK = kd[1]; op.sk = (int32_t)kd[2]; DV = vd[3];
  }
  op.h = (int32_t)H; op.hk = (int32_t)HK; op.d = (int32_t)D; op.dv = (int32_t)DV;
  if (op.scale <= 0.f) op.scale = (float)(1.0 / sqrt((double)D));
  if (op.softcap > 0.0f) return "本构建不支持 softcap (FLASHATTENTION_DISABLE_SOFTCAP)";
  return nullptr;
}

// 缓冲按操作数在前、结果在后解释；占位参数也占索引。
// 普通前向：[0..9]=q,k,v,rng,dbg,sem,alibi,saux,cuq,cuk，[10..11]=o,lse。
// FP8 前向在 [10..12] 追加三个 descale，[13..14]=o,lse。
// 长度掩码前向：[0..6]=q,k,v,rng,dbg,sem,length，[7..8]=o,lse。
// 反向：[0..10]=q,k,v,o,do,lse,rng,dbg,sem,alibi,saux；
// 变长反向另加 [11..12]=cuq,cuk，随后为 dq,dk_exp,dv_exp,dsoftmax_sum。
// 反向结果起点在稠密路径为 11，在变长路径为 13。
static bool CollectBuffers(fa_ffi::RemainingArgs args, fa_ffi::RemainingRets rets,
                           void** bufs, int* n_out, int64_t dims[4][8],
                           int ranks[4], int64_t* cu_len) {
  int n = 0;
  for (int64_t i = 0; i < args.size(); ++i) {
    auto b = args.get<fa_ffi::AnyBuffer>(i);
    if (!b.has_value() || n >= 31) return false;
    bufs[n++] = b->untyped_data();          // args: ErrorOr<AnyBuffer>
    if (i < 4) {
      auto d = b->dimensions();
      ranks[i] = (int)d.size();
      for (size_t j = 0; j < d.size() && j < 8; ++j) dims[i][j] = d[j];
    }
    if (i == 8 || i == 11) {          // cu_seqlens_q (fwd:8 / bwd:11)
      auto d = b->dimensions();
      if (d.size() >= 1) *cu_len = d[0] - 1;
    }
  }
  for (int64_t j = 0; j < rets.size(); ++j) {
    auto b = rets.get<fa_ffi::AnyBuffer>(j);
    if (!b.has_value() || n >= 32) return false;
    bufs[n++] = (*b)->untyped_data();
  }
  *n_out = n;
  return true;
}

static fa_ffi::Error FaFwdTypedImpl(
    fa_ffi::RemainingArgs args, fa_ffi::RemainingRets rets, hipStream_t stream,
    float scale, int32_t causal, int32_t layout, int32_t dtype, int32_t wl,
    int32_t wr, float softcap, int32_t has_alibi, int32_t alibi_batch,
    int32_t has_sinks, int32_t sink_type, int32_t has_varlen, int32_t msq,
    int32_t msk, int32_t vbwd_mode, int32_t lse_unpadded, float dropout_p,
    int32_t is_fp8, int32_t deterministic) {
  if (args.size() < 10 || rets.size() != 2)
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument,
                         "fa_fwd_ffi: 期望 args=q,k,v,rng,dbg,sem,alibi,saux,cuq,cuk; rets=o,lse");
  FaOpaque op;
  memset(&op, 0, sizeof(op));
  op.scale = scale; op.causal = causal; op.layout = layout; op.dtype = dtype;
  op.window_left = wl; op.window_right = wr; op.softcap = softcap;
  op.has_alibi = has_alibi; op.alibi_batch = alibi_batch;
  op.has_sinks = has_sinks; op.sink_type = sink_type;
  op.has_varlen = has_varlen; op.sq = msq; op.sk = msk;
  op.vbwd_mode = vbwd_mode; op.lse_unpadded = lse_unpadded;
  op.dropout_p = dropout_p; op.is_fp8 = is_fp8; op.deterministic = deterministic;
  void* bufs[32];
  int n = 0, ranks[4] = {0, 0, 0, 0};
  int64_t dims[4][8] = {}, cu_len = -1;
  if (!CollectBuffers(args, rets, bufs, &n, dims, ranks, &cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, "fa_fwd_ffi: 缓冲解码失败");
  if (const char* err = FillOpaqueFromShapes(op, dims[0], ranks[0], dims[1], ranks[1],
                                             dims[2], ranks[2], cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, err);
  if (const char* err = fa_lib_check("run_mha_fwd", (const void*)fa_lib().fwd))
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, err);
  fa_fwd_v0(stream, bufs, (const char*)&op, sizeof(op), nullptr);
  if (fa_native_error) return fa_ffi::Error(fa_ffi::ErrorCode::kInternal, fa_native_error);
  return fa_ffi::Error::Success();
}

static fa_ffi::Error FaBwdTypedImpl(
    fa_ffi::RemainingArgs args, fa_ffi::RemainingRets rets, hipStream_t stream,
    float scale, int32_t causal, int32_t layout, int32_t dtype, int32_t wl,
    int32_t wr, float softcap, int32_t has_alibi, int32_t alibi_batch,
    int32_t has_sinks, int32_t sink_type, int32_t has_varlen, int32_t msq,
    int32_t msk, int32_t vbwd_mode, int32_t lse_unpadded, float dropout_p,
    int32_t is_fp8, int32_t deterministic) {
  if (args.size() < 11 || rets.size() != 4)
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument,
                         "fa_bwd_ffi: 期望 args=q,k,v,o,do,lse,rng,dbg,sem,alibi,saux[,cuq,cuk]; rets=dq,dk,dv,dsm");
  FaOpaque op;
  memset(&op, 0, sizeof(op));
  op.scale = scale; op.causal = causal; op.layout = layout; op.dtype = dtype;
  op.window_left = wl; op.window_right = wr; op.softcap = softcap;
  op.has_alibi = has_alibi; op.alibi_batch = alibi_batch;
  op.has_sinks = has_sinks; op.sink_type = sink_type;
  op.has_varlen = has_varlen; op.sq = msq; op.sk = msk;
  op.vbwd_mode = vbwd_mode; op.lse_unpadded = lse_unpadded;
  op.dropout_p = dropout_p; op.is_fp8 = is_fp8; op.deterministic = deterministic;
  void* bufs[32];
  int n = 0, ranks[4] = {0, 0, 0, 0};
  int64_t dims[4][8] = {}, cu_len = -1;
  if (!CollectBuffers(args, rets, bufs, &n, dims, ranks, &cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, "fa_bwd_ffi: 缓冲解码失败");
  if (const char* err = FillOpaqueFromShapes(op, dims[0], ranks[0], dims[1], ranks[1],
                                             dims[2], ranks[2], cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, err);
  if (const char* err = fa_lib_check("run_mha_bwd", (const void*)fa_lib().bwd))
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, err);
  fa_bwd_v0(stream, bufs, (const char*)&op, sizeof(op), nullptr);
  if (fa_native_error) return fa_ffi::Error(fa_ffi::ErrorCode::kInternal, fa_native_error);
  return fa_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kFaFwdTyped, FaFwdTypedImpl,
    fa_ffi::Ffi::Bind().RemainingArgs().RemainingRets()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<float>("scale").Attr<int32_t>("causal").Attr<int32_t>("layout")
        .Attr<int32_t>("dtype").Attr<int32_t>("wl").Attr<int32_t>("wr")
        .Attr<float>("softcap").Attr<int32_t>("has_alibi")
        .Attr<int32_t>("alibi_batch").Attr<int32_t>("has_sinks")
        .Attr<int32_t>("sink_type").Attr<int32_t>("has_varlen")
        .Attr<int32_t>("msq").Attr<int32_t>("msk").Attr<int32_t>("vbwd_mode")
        .Attr<int32_t>("lse_unpadded").Attr<float>("dropout_p")
        .Attr<int32_t>("is_fp8").Attr<int32_t>("deterministic"));
XLA_FFI_DEFINE_HANDLER(kFaBwdTyped, FaBwdTypedImpl,
    fa_ffi::Ffi::Bind().RemainingArgs().RemainingRets()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<float>("scale").Attr<int32_t>("causal").Attr<int32_t>("layout")
        .Attr<int32_t>("dtype").Attr<int32_t>("wl").Attr<int32_t>("wr")
        .Attr<float>("softcap").Attr<int32_t>("has_alibi")
        .Attr<int32_t>("alibi_batch").Attr<int32_t>("has_sinks")
        .Attr<int32_t>("sink_type").Attr<int32_t>("has_varlen")
        .Attr<int32_t>("msq").Attr<int32_t>("msk").Attr<int32_t>("vbwd_mode")
        .Attr<int32_t>("lse_unpadded").Attr<float>("dropout_p")
        .Attr<int32_t>("is_fp8").Attr<int32_t>("deterministic"));

extern "C" void* fa_fwd_typed_handler() { return (void*)kFaFwdTyped; }
extern "C" void* fa_bwd_typed_handler() { return (void*)kFaBwdTyped; }

// PA (分页解码) 的类型化 FFI handler: 维度由形状推 (page 取自 kcache.dims[1]),
// sk/num_splits/partition_size/... 由 attrs 传 (无法从形状推)。
static fa_ffi::Error FaPaTypedImpl(
    fa_ffi::RemainingArgs args, fa_ffi::RemainingRets rets, hipStream_t stream,
    float scale, int32_t causal, int32_t layout, int32_t dtype, int32_t wl,
    int32_t wr, int32_t sk, int32_t num_splits, int32_t partition_size,
    int32_t mtp, int32_t ngroups, int32_t bt_stride, int32_t total_q,
    int32_t total_k, int32_t mtp_sq, int32_t is_fp8) {
  if (args.size() != 11 || rets.size() != 1)
    return fa_ffi::Error(
        fa_ffi::ErrorCode::kInvalidArgument,
        "fa_pa_ffi: 期望 args=q,kcache,vcache,bt,seqlens,sc_acc,o_acc,dummy,3×descale; rets=o");
  FaPaOpaque op;
  memset(&op, 0, sizeof(op));
  op.scale = scale; op.causal = causal; op.layout = layout; op.dtype = dtype;
  op.wl = wl; op.wr = wr; op.sk = sk; op.num_splits = num_splits;
  op.partition_size = partition_size; op.mtp = mtp; op.ngroups = ngroups;
  op.bt_stride = bt_stride; op.total_q = total_q; op.total_k = total_k;
  op.mtp_sq = mtp_sq; op.is_fp8 = is_fp8;
  void* bufs[32];
  int n = 0, ranks[4] = {0, 0, 0, 0};
  int64_t dims[4][8] = {}, cu_len = -1;
  if (!CollectBuffers(args, rets, bufs, &n, dims, ranks, &cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, "fa_pa_ffi: 缓冲解码失败");
  if (ranks[0] != 4 || ranks[1] != 4 || ranks[2] != 4 || ranks[3] != 2)
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument,
                         "fa_pa_ffi: q/kcache/vcache 须 4-D, block_table 须 2-D");
  op.b = (int32_t)dims[0][0];
  op.sq = (int32_t)dims[0][1];
  op.h = (int32_t)dims[0][2];
  op.d = (int32_t)dims[0][3];
  op.page_block_size = (int32_t)dims[1][1];
  op.hk = (int32_t)dims[1][2];
  op.dv = (int32_t)dims[2][3];
  if (const char* err = fa_lib_check("run_mha_fwd_kvcache", (const void*)fa_lib().kvcache))
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, err);
  fa_fwd_kvcache_v0(stream, bufs, (const char*)&op, sizeof(op), nullptr);
  return fa_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kFaPaTyped, FaPaTypedImpl,
    fa_ffi::Ffi::Bind().RemainingArgs().RemainingRets()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<float>("scale").Attr<int32_t>("causal").Attr<int32_t>("layout")
        .Attr<int32_t>("dtype").Attr<int32_t>("wl").Attr<int32_t>("wr")
        .Attr<int32_t>("sk").Attr<int32_t>("num_splits")
        .Attr<int32_t>("partition_size").Attr<int32_t>("mtp")
        .Attr<int32_t>("ngroups").Attr<int32_t>("bt_stride")
        .Attr<int32_t>("total_q").Attr<int32_t>("total_k")
        .Attr<int32_t>("mtp_sq").Attr<int32_t>("is_fp8"));

extern "C" void* fa_pa_typed_handler() { return (void*)kFaPaTyped; }

// prefix-prefill 的类型化 FFI handler (FaPaOpaque; 操作数 q,kc,vc,bt,seqused,cuq,cuk,sem
// → 结果 o,lse)。page/bt_stride/b 由形状推; msq/sk/total_k 由 attrs 传 (无法从形状推)。
static fa_ffi::Error FaPrefixTypedImpl(
    fa_ffi::RemainingArgs args, fa_ffi::RemainingRets rets, hipStream_t stream,
    float scale, int32_t causal, int32_t dtype, int32_t wl, int32_t wr,
    int32_t sk, int32_t msq, int32_t total_k) {
  if (args.size() != 8 || rets.size() != 2)
    return fa_ffi::Error(
        fa_ffi::ErrorCode::kInvalidArgument,
        "fa_prefix_ffi: 期望 args=q,kcache,vcache,bt,seqused,cuq,cuk,sem; rets=o,lse");
  FaPaOpaque op;
  memset(&op, 0, sizeof(op));
  op.scale = scale; op.causal = causal; op.dtype = dtype; op.layout = 1;
  op.wl = wl; op.wr = wr; op.sk = sk; op.sq = msq; op.mtp_sq = msq;
  op.total_q = 0; op.total_k = total_k;
  op.num_splits = 0; op.partition_size = 0; op.mtp = 0; op.ngroups = 0;
  void* bufs[32];
  int n = 0, ranks[4] = {0, 0, 0, 0};
  int64_t dims[4][8] = {}, cu_len = -1;
  if (!CollectBuffers(args, rets, bufs, &n, dims, ranks, &cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, "fa_prefix_ffi: 缓冲解码失败");
  if (ranks[0] != 3 || ranks[1] != 4 || ranks[2] != 4 || ranks[3] != 2)
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument,
                         "fa_prefix_ffi: q 须 3-D, kcache/vcache 4-D, block_table 2-D");
  op.total_q = (int32_t)dims[0][0];
  op.h = (int32_t)dims[0][1];
  op.d = (int32_t)dims[0][2];
  op.page_block_size = (int32_t)dims[1][1];
  op.hk = (int32_t)dims[1][2];
  op.dv = (int32_t)dims[2][3];
  op.b = (int32_t)dims[3][0];
  op.bt_stride = (int32_t)dims[3][1];
  if (const char* err = fa_lib_check("run_mha_fwd_kvcache", (const void*)fa_lib().kvcache))
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, err);
  fa_prefix_v0(stream, bufs, (const char*)&op, sizeof(op), nullptr);
  return fa_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kFaPrefixTyped, FaPrefixTypedImpl,
    fa_ffi::Ffi::Bind().RemainingArgs().RemainingRets()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<float>("scale").Attr<int32_t>("causal").Attr<int32_t>("dtype")
        .Attr<int32_t>("wl").Attr<int32_t>("wr").Attr<int32_t>("sk")
        .Attr<int32_t>("msq").Attr<int32_t>("total_k"));

extern "C" void* fa_prefix_typed_handler() { return (void*)kFaPrefixTyped; }

static fa_ffi::Error FaFp8TypedImpl(
    fa_ffi::RemainingArgs args, fa_ffi::RemainingRets rets, hipStream_t stream,
    float scale, int32_t causal, int32_t layout, int32_t dtype, int32_t wl,
    int32_t wr) {
  if (args.size() != 13 || rets.size() != 2)
    return fa_ffi::Error(
        fa_ffi::ErrorCode::kInvalidArgument,
        "fa_fp8_ffi: 期望 args=q,k,v,rng,dbg,sem,alibi,saux,cuq,cuk,desc_q,desc_k,desc_v; rets=o,lse");
  FaOpaque op;
  memset(&op, 0, sizeof(op));
  op.scale = scale; op.causal = causal; op.layout = layout; op.dtype = dtype;
  op.window_left = wl; op.window_right = wr; op.is_fp8 = 1;
  void* bufs[32];
  int n = 0, ranks[4] = {0, 0, 0, 0};
  int64_t dims[4][8] = {}, cu_len = -1;
  if (!CollectBuffers(args, rets, bufs, &n, dims, ranks, &cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, "fa_fp8_ffi: 缓冲解码失败");
  if (const char* err = FillOpaqueFromShapes(op, dims[0], ranks[0], dims[1], ranks[1],
                                             dims[2], ranks[2], cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument, err);
  if (const char* err = fa_lib_check("run_mha_fwd", (const void*)fa_lib().fwd))
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, err);
  FaFwdLegacyImpl(stream, bufs, op, 13, bufs[10], bufs[11], bufs[12]);
  if (fa_native_error) return fa_ffi::Error(fa_ffi::ErrorCode::kInternal, fa_native_error);
  return fa_ffi::Error::Success();
}

static fa_ffi::Error FaMlaPrefixTypedImpl(
    fa_ffi::RemainingArgs args, fa_ffi::RemainingRets rets, hipStream_t stream,
    float scale, int32_t causal, int32_t dtype, int32_t msq, int32_t is_mtp) {
  if (args.size() != 9 || rets.size() != 2)
    return fa_ffi::Error(
        fa_ffi::ErrorCode::kInvalidArgument,
        "fa_mla_prefix_ffi: 期望 args=q,qv,kcache,vcache,pt,cs,cuq,cuk_new,scores_mem; rets=o,lse");
  FaMlaOpaque op;
  memset(&op, 0, sizeof(op));
  op.scale = scale; op.causal = causal; op.dtype = dtype;
  op.sq = 0;                 // 占位, 下面按形状填 total_q
  op.sk = msq;               // legacy: op.sk 槽位 = max_seqlen_q
  op.num_splits = is_mtp;    // legacy: 第 11 个 int 是 is_mtp
  op.partition_size = 0;
  void* bufs[32];
  int n = 0, ranks[4] = {0, 0, 0, 0};
  int64_t dims[4][8] = {}, cu_len = -1;
  if (!CollectBuffers(args, rets, bufs, &n, dims, ranks, &cu_len))
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument,
                         "fa_mla_prefix_ffi: 缓冲解码失败");
  if (ranks[0] != 3 || ranks[2] != 4 || ranks[4] != 2)
    return fa_ffi::Error(fa_ffi::ErrorCode::kInvalidArgument,
                         "fa_mla_prefix_ffi: q 须 3-D, kcache 4-D, page_table 2-D");
  op.sq = (int32_t)dims[0][0];      // total_q
  op.h = (int32_t)dims[0][1];
  op.page = (int32_t)dims[2][1];
  op.hk = (int32_t)dims[2][2];
  op.b = (int32_t)dims[4][0];
  op.bt_stride = (int32_t)dims[4][1];
  if (const char* err = fa_lib_check("run_fwd_prefix_prefill_mla", (const void*)fa_lib().prefix_mla))
    return fa_ffi::Error(fa_ffi::ErrorCode::kFailedPrecondition, err);
  fa_mla_prefix_v0(stream, bufs, (const char*)&op, sizeof(op), nullptr);
  return fa_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kFaFp8Typed, FaFp8TypedImpl,
    fa_ffi::Ffi::Bind().RemainingArgs().RemainingRets()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<float>("scale").Attr<int32_t>("causal").Attr<int32_t>("layout")
        .Attr<int32_t>("dtype").Attr<int32_t>("wl").Attr<int32_t>("wr"));

XLA_FFI_DEFINE_HANDLER(kFaMlaPrefixTyped, FaMlaPrefixTypedImpl,
    fa_ffi::Ffi::Bind().RemainingArgs().RemainingRets()
        .Ctx<fa_ffi::PlatformStream<hipStream_t>>()
        .Attr<float>("scale").Attr<int32_t>("causal").Attr<int32_t>("dtype")
        .Attr<int32_t>("msq").Attr<int32_t>("is_mtp"));

extern "C" void* fa_fp8_typed_handler() { return (void*)kFaFp8Typed; }
extern "C" void* fa_mla_prefix_typed_handler() { return (void*)kFaMlaPrefixTyped; }
