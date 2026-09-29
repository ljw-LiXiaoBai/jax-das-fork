// 向 Python 导出 ROCM 调用目标及架构探测接口。
// registrations() 返回 {平台: [(名称, capsule, api_version)]}；
// 0 表示旧式自定义调用，1 表示类型化 FFI。
// capsule 名称须为 "xla._CUSTOM_CALL_TARGET"。
// 默认使用 nanobind；定义 FA_HG_USE_PYBIND11 时使用 pybind11。
#include "jaxlib/gpu/fa_attention_kernel.h"

#if defined(FA_HG_USE_PYBIND11)
#include <pybind11/pybind11.h>
namespace nb = pybind11;
#define NB_MODULE(name, m) PYBIND11_MODULE(name, m)
#else
#include <nanobind/nanobind.h>
namespace nb = nanobind;
#endif

namespace {

nb::capsule TargetCapsule(void* fn) {
  return nb::capsule(fn, "xla._CUSTOM_CALL_TARGET");
}

// FFI handler 由 XLA_FFI_DEFINE_HANDLER 生成, 以函数指针形式导出
nb::capsule FfiCapsule(void* (*getter)()) {
  return nb::capsule(getter(), "xla._CUSTOM_CALL_TARGET");
}

}  // namespace

NB_MODULE(fa_attention, m) {
  m.def("registrations", []() {
    nb::list legacy;
    legacy.append(nb::make_tuple("fa_fwd_v0", TargetCapsule((void*)&fa_fwd_v0), 0));
    legacy.append(nb::make_tuple("fa_bwd_v0", TargetCapsule((void*)&fa_bwd_v0), 0));
    legacy.append(nb::make_tuple("fa_fwd_kvcache_v0",
                                TargetCapsule((void*)&fa_fwd_kvcache_v0), 0));
    legacy.append(nb::make_tuple("fa_prefix_v0", TargetCapsule((void*)&fa_prefix_v0), 0));
    legacy.append(nb::make_tuple("fa_mla_prefix_v0",
                                TargetCapsule((void*)&fa_mla_prefix_v0), 0));
    legacy.append(nb::make_tuple("fa_fwd_fp8_v0", TargetCapsule((void*)&fa_fwd_fp8_v0), 0));

    nb::list ffi;
    ffi.append(nb::make_tuple("fa_fwd_ffi", FfiCapsule(&fa_fwd_typed_handler), 1));
    ffi.append(nb::make_tuple("fa_attn_length_ffi", FfiCapsule(&fa_attn_length_typed_handler), 1));
    ffi.append(nb::make_tuple("fa_validate_lengths_ffi", FfiCapsule(&fa_validate_lengths_typed_handler), 1));
    ffi.append(nb::make_tuple("fa_bwd_ffi", FfiCapsule(&fa_bwd_typed_handler), 1));
    ffi.append(nb::make_tuple("fa_pa_ffi", FfiCapsule(&fa_pa_typed_handler), 1));
    ffi.append(nb::make_tuple("fa_prefix_ffi", FfiCapsule(&fa_prefix_typed_handler), 1));
    ffi.append(nb::make_tuple("fa_fp8_ffi", FfiCapsule(&fa_fp8_typed_handler), 1));
    ffi.append(nb::make_tuple("fa_mla_prefix_ffi",
                             FfiCapsule(&fa_mla_prefix_typed_handler), 1));

    nb::list all;                 // (name, capsule, api_version) 三元组列表
    for (auto item : legacy) all.append(item);
    for (auto item : ffi) all.append(item);
    nb::dict dict;
    dict["ROCM"] = all;
    return dict;
  });

  m.def("arch_raw", []() { return (int)fa_arch_raw(); });
  m.def("arch_fp8_ok", []() { return (int)fa_arch_fp8_ok(); });
  m.def("arch_fp8_pa_ok", []() { return (int)fa_arch_fp8_pa_ok(); });
}
