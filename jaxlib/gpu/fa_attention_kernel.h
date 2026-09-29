// Hygon FlashAttention 的 XLA 调用入口。
// api_version=0 使用旧式 ABI：buffers 按操作数、结果的顺序排列，
// opaque 的字段顺序须与 Python 打包格式一致，stream 由 XLA 提供。
// api_version=1 返回 XLA_FFI_DEFINE_HANDLER 生成的类型化处理器指针。
#ifndef JAXLIB_GPU_FA_ATTENTION_KERNEL_H_
#define JAXLIB_GPU_FA_ATTENTION_KERNEL_H_

#include <cstddef>
#include <cstdint>

extern "C" {
// ---- legacy custom call 入口 (api_version=0) ----
void fa_fwd_v0(void* stream, void** buffers, const char* opaque, size_t len, void* status);
void fa_bwd_v0(void* stream, void** buffers, const char* opaque, size_t len, void* status);
void fa_fwd_kvcache_v0(void* stream, void** buffers, const char* opaque, size_t len, void* status);
void fa_prefix_v0(void* stream, void** buffers, const char* opaque, size_t len, void* status);
void fa_mla_prefix_v0(void* stream, void** buffers, const char* opaque, size_t len, void* status);
void fa_fwd_fp8_v0(void* stream, void** buffers, const char* opaque, size_t len, void* status);

// ---- XLA FFI (api_version=1) handler 指针 ----
void* fa_fwd_typed_handler();
void* fa_bwd_typed_handler();
void* fa_pa_typed_handler();
void* fa_prefix_typed_handler();
void* fa_fp8_typed_handler();
void* fa_mla_prefix_typed_handler();
void* fa_attn_length_typed_handler();
void* fa_validate_lengths_typed_handler();

// ---- arch 探测 ----
int fa_arch_raw();
int fa_arch_fp8_ok();
int fa_arch_fp8_pa_ok();
}

#endif  // JAXLIB_GPU_FA_ATTENTION_KERNEL_H_
