# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HIPC FlashAttention 的 JAX 接入层。

提供稠密、变长、分页缓存及 FP8 注意力接口，约束和求导范围见各接口。
"""
import functools
import inspect
import os
import struct

import numpy as np

import jax
import jax.numpy as jnp
from jax._src import core, dispatch
from jax._src.custom_partitioning import custom_partitioning
from jax._src.interpreters import batching, mlir
from jax._src.lib import xla_client
from jax.sharding import NamedSharding, PartitionSpec

_USE_FFI = os.environ.get("FA_LEGACY", "0") != "1"
FWD_FFI_TARGET = "fa_fwd_ffi"
BWD_FFI_TARGET = "fa_bwd_ffi"
PA_FFI_TARGET = "fa_pa_ffi"
PREFIX_FFI_TARGET = "fa_prefix_ffi"
FP8_FFI_TARGET = "fa_fp8_ffi"
MLA_PREFIX_FFI_TARGET = "fa_mla_prefix_ffi"
FWD_TARGET = "fa_fwd_v0"
BWD_TARGET = "fa_bwd_v0"
PA_TARGET = "fa_fwd_kvcache_v0"
MASK_TARGET = "fa_fwd_mask_v0"
PREFIX_TARGET = "fa_prefix_v0"
FP8_TARGET = "fa_fwd_fp8_v0"
MLA_PREFIX_TARGET = "fa_mla_prefix_v0"
_REGISTERED = False


def mark_registered() -> None:
    global _REGISTERED
    _REGISTERED = True


def _load_plugin():
    import importlib
    import importlib.util
    from jaxlib import plugin_support

    names = ["jaxlib.rocm"] + plugin_support._PLUGIN_MODULE_NAMES["rocm"]
    candidates = [name + ".fa_attention" for name in names] + ["jaxlib.fa_attention"]
    for name in candidates:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError as e:
            if e.name and (name == e.name or name.startswith(e.name + ".")):
                continue
            raise
        if spec is not None:
            return importlib.import_module(name)
    raise ModuleNotFoundError(
        "fa_attention ROCm 插件未安装，请构建并安装包含该模块的插件 wheel",
        name="fa_attention_plugin")


def _register(shim_path: str = None) -> None:
    """注册旧式自定义调用与 XLA FFI，注册参数 api_version 分别为 0 和 1。

    FA_LEGACY=1 启用保留的旧式路径；dropout 前向沿用旧式 ABI，
    专用有效长度内核及长度校验仅提供 FFI。
    """
    global _REGISTERED, _PLUGIN_MOD
    if _REGISTERED:
        return
    mod = _load_plugin()
    for platform, targets in mod.registrations().items():
        for name, capsule, api_version in targets:
            xla_client.register_custom_call_target(
                name.encode() if isinstance(name, str) else name, capsule,
                platform=platform, api_version=int(api_version))
    _PLUGIN_MOD = mod
    _REGISTERED = True


def _rm(x, m):
    return (x + m - 1) // m * m


def _shapes(q_aval, k_aval, v_aval, layout):
    if layout == 1:
        b, s, h, d = q_aval.shape
        sq = s
        sk, hk, dv = k_aval.shape[1], k_aval.shape[2], v_aval.shape[-1]
    else:
        b, h, sq, d = q_aval.shape
        sk, hk, dv = k_aval.shape[2], k_aval.shape[1], v_aval.shape[-1]
    return b, h, hk, sq, sk, d, dv


def _pack_opaque(b, h, hk, sq, sk, d, dv, scale, causal, layout, dtype,
                 wl, wr, softcap, has_alibi, alibi_batch, has_sinks, sink_type,
                 has_varlen, total_q, total_k, vbwd_mode=0, lse_unpadded=0,
                 dropout_p=0.0, is_fp8=0, deterministic=0):
    return struct.pack(_OPAQUE_FMT, b, h, hk, sq, sk, d, dv,
                       float(scale), int(causal), int(layout), int(dtype),
                       int(wl), int(wr), float(softcap), int(has_alibi),
                       int(alibi_batch), int(has_sinks), int(sink_type),
                       int(has_varlen), int(total_q), int(total_k),
                       int(vbwd_mode), int(lse_unpadded), float(dropout_p),
                       int(is_fp8), int(deterministic))


_OPAQUE_FMT = "<7if5if9ifii"  # 字段顺序及类型须与对应的 C++ ABI 描述符保持一致。
assert struct.calcsize(_OPAQUE_FMT) == 104



def _rng_operand(seed) -> jnp.ndarray:
    """以 int32[4] 保存 Philox 的两个 uint64 状态，缓冲共 16 字节。

    避免关闭 x64 时 uint64 数组被缩窄。内核会回写状态，
    传给内核的可写状态不得复用调用方的只读常量。
    """
    lo = int(seed) & 0xFFFFFFFF
    hi = (int(seed) >> 32) & 0xFFFFFFFF
    return jnp.asarray(jnp.array([lo, hi, 0, 0], dtype=jnp.int32))


def _contract():
    # 不全局缓存 JAX 数组，避免首次调用处于 jit 追踪时发生追踪值逃逸。
    return {"rng": jnp.zeros((4,), jnp.int32),
            "dbg": jnp.zeros((1,), jnp.int32),
            "sem": jnp.zeros((64,), jnp.int32)}


def _placeholder():
    return jnp.zeros((1,), jnp.float32)


# ------------------------------- fwd ----------------------------------------
_fa_p = core.Primitive("fa_fwd_v0")
_fa_p.multiple_results = True
_fa_p.def_impl(functools.partial(dispatch.apply_primitive, _fa_p))


def _fa_abstract(*avals, **params):
    q, k, v = avals[0], avals[1], avals[2]
    layout = params.get("layout", 1)
    if q.dtype not in (jnp.float16, jnp.bfloat16):
        raise TypeError(f"fa_fwd 仅支持 fp16/bf16, 得到 {q.dtype}")
    if q.ndim == 3:
        total_q, h, _ = q.shape
        dv = v.shape[-1]
        o_aval = q.update(shape=(total_q, h, dv), weak_type=False)
        lse_aval = core.ShapedArray((h, total_q), jnp.float32)
        return [o_aval, lse_aval]
    b, h, hk, sq, sk, d, dv = _shapes(q, k, v, layout)
    o_shape = (b, sq, h, dv) if layout == 1 else (b, h, sq, dv)
    o_aval = q.update(shape=o_shape, weak_type=False)
    lse_aval = core.ShapedArray((b, h, sq), jnp.float32)
    return [o_aval, lse_aval]


_fa_p.def_abstract_eval(_fa_abstract)


def _fa_lower(ctx, q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, *, scale, causal,
              layout, wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
              sink_type, has_varlen, msq, msk, dropout_p):
    q_aval, k_aval, v_aval = ctx.avals_in[0], ctx.avals_in[1], ctx.avals_in[2]
    if has_varlen:
        total_q, h, d = q_aval.shape
        total_k, hk, dv = k_aval.shape[0], k_aval.shape[1], v_aval.shape[2]
        b = ctx.avals_in[8].shape[0] - 1
        sq, sk = msq, msk
        opaque = _pack_opaque(b, h, hk, int(sq), int(sk), d, dv, scale, causal,
                              layout, 1 if q_aval.dtype == jnp.bfloat16 else 0,
                              wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
                              sink_type, 1, total_q, total_k, 0, 0,
                              float(dropout_p))
    else:
        b, h, hk, sq, sk, d, dv = _shapes(q_aval, k_aval, v_aval, layout)
        total_q = total_k = 0
        opaque = _pack_opaque(b, h, hk, sq, sk, d, dv, scale, causal, layout,
                              1 if q_aval.dtype == jnp.bfloat16 else 0,
                              wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
                              sink_type, 0, 0, 0, 0, 0, float(dropout_p))

    def layouts(a):
        return mlir.dense_int_array(tuple(range(a.ndim - 1, -1, -1)))

    out = mlir.custom_call(
        FWD_TARGET,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk],
        backend_config=opaque,
        operand_layouts=[layouts(a) for a in ctx.avals_in],
        result_layouts=[layouts(a) for a in ctx.avals_out],
    )
    return [out.results[0], out.results[1]]


mlir.register_lowering(_fa_p, _fa_lower, platform="rocm")


def _fa_fwd_impl(q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, scale, causal,
                 layout, wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
                 sink_type, has_varlen, msq, msk, dropout_p):
    if _USE_FFI and not dropout_p:  # dropout retains its existing legacy ABI
        return _fa_fwd_ffi(q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, scale,
                           causal, layout, wl, wr, softcap, has_alibi, alibi_batch,
                           has_sinks, sink_type, has_varlen, msq, msk, dropout_p)
    return tuple(_fa_p.bind(q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk,
                            scale=scale, causal=causal, layout=layout, wl=wl, wr=wr,
                            softcap=softcap, has_alibi=has_alibi,
                            alibi_batch=alibi_batch, has_sinks=has_sinks,
                            sink_type=sink_type, has_varlen=has_varlen, msq=msq,
                            msk=msk, dropout_p=dropout_p))


def _padded_spec(arg_info):
    spec = None if arg_info.sharding is None else arg_info.sharding.spec
    if spec is None:
        return (None,) * arg_info.ndim
    return tuple(spec) + (None,) * (arg_info.ndim - len(tuple(spec)))


def _sequential_partitioned(fun, dynamic_count):
    """逐次映射完整内核调用，保留操作数维数及各自的累计长度。

    custom_partitioning 放在顺序 vmap 内部；布局和最大长度等静态选项
    通过闭包传递，避免变成追踪值。
    """
    from jax.custom_batching import sequential_vmap
    signature = inspect.signature(fun.fun)

    @functools.wraps(fun.fun)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = tuple(bound.arguments.values())
        static = values[dynamic_count:]

        @sequential_vmap
        def mapped(*dynamic):
            return fun(*dynamic, *static)

        return mapped(*values[:dynamic_count])

    return call


def _sequential_batch(primitive, batched_args, batch_dims, **params):
    mapped = tuple(i for i, dim in enumerate(batch_dims) if dim is not None)
    xs = tuple(batching.moveaxis(batched_args[i], batch_dims[i], 0)
               for i in mapped)

    def body(slices):
        args = list(batched_args)
        for i, value in zip(mapped, slices):
            args[i] = value
        return primitive.bind(*args, **params)

    results = jax.lax.map(body, xs)
    if primitive.multiple_results:
        return results, (0,) * len(results)
    return results, 0


def _axis_size(mesh, axis):
    if axis is None:
        return 1
    axes = axis if isinstance(axis, tuple) else (axis,)
    return int(np.prod([mesh.shape[name] for name in axes]))


def _dense_partition_specs(mesh, arg_shapes, layout, *, backward=False):
    if layout not in (0, 1):
        raise ValueError("fa: layout must be 0 or 1")
    q, k, v = arg_shapes[:3]
    if any(a.ndim != 4 for a in (q, k, v)):
        raise ValueError("fa: dense partitioning requires rank-4 q/k/v")
    hi, si = (2, 1) if layout == 1 else (1, 2)
    specs = tuple(_padded_spec(a) for a in arg_shapes)
    qs, ks, vs = specs[:3]
    checked = specs[:5] if backward else specs[:3]
    for sp in checked:
        if sp[si] is not None or sp[3] is not None:
            raise ValueError("fa: sequence/feature sharding is unsupported; use batch/head sharding")
        if (sp[0], sp[hi]) != (qs[0], qs[hi]):
            raise ValueError("fa: q/k/v/o/do batch/head sharding must match")
    h, hk = q.shape[hi], k.shape[hi]
    if hk <= 0 or h % hk or v.shape[hi] != hk:
        raise ValueError("fa: q heads must be divisible by matching k/v heads (GQA)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("fa: q/k/v batch sizes must match")
    if hk % _axis_size(mesh, qs[hi]) or q.shape[0] % _axis_size(mesh, qs[0]):
        raise ValueError("fa: shards must contain whole KV head groups and whole batches")
    lse_spec = (qs[0], qs[hi], None)
    if backward and specs[5] != lse_spec:
        raise ValueError("fa bwd: LSE must use (batch, head, replicated sequence) sharding")
    return specs, hi, lse_spec


def _canonical_dense_shapes(mesh, arg_shapes, layout, *, backward=False,
                            has_sinks=False):
    hi, si = (2, 1) if layout == 1 else (1, 2)
    qs = _padded_spec(arg_shapes[0])
    count = 5 if backward else 3
    for arg in arg_shapes[:count]:
        spec = _padded_spec(arg)
        if spec[si] is not None or spec[3] is not None:
            raise ValueError("fa: sequence/feature sharding is unsupported; use batch/head sharding")
    canonical = []
    for i, arg in enumerate(arg_shapes):
        if i < count:
            spec = qs
        elif backward and i == 5:
            spec = (qs[0], qs[hi], None)
        elif i == (10 if backward else 7) and has_sinks:
            spec = (qs[hi],)
        else:
            spec = (None,) * arg.ndim
        canonical.append(jax.ShapeDtypeStruct(
            arg.shape, arg.dtype, sharding=NamedSharding(mesh, PartitionSpec(*spec))))
    return tuple(canonical)


def _check_aux_sharding(arg_shapes, first, *, sink_index, has_sinks, head_axis):
    for i in range(first, len(arg_shapes)):
        spec = _padded_spec(arg_shapes[i])
        expected = ((head_axis,) if i == sink_index and has_sinks
                    else (None,) * arg_shapes[i].ndim)
        if spec != expected:
            raise ValueError("fa: scratch/metadata must be replicated; sinks must follow query heads")


def _fa_fwd_infer_sharding(scale, causal, layout, wl, wr, softcap, has_alibi,
                           alibi_batch, has_sinks, sink_type, has_varlen, msq,
                           msk, dropout_p, mesh, arg_shapes, result_shapes):
    if has_varlen:
        if any(any(x is not None for x in _padded_spec(a)) for a in arg_shapes):
            raise ValueError("fa varlen: packed sharding is unsupported; use replicated inputs or sequential vmap")
        return (NamedSharding(mesh, PartitionSpec(None, None, None)),
                NamedSharding(mesh, PartitionSpec(None, None)))
    arg_shapes = _canonical_dense_shapes(mesh, arg_shapes, layout, has_sinks=has_sinks)
    specs, hi, lse_spec = _dense_partition_specs(mesh, arg_shapes, layout)
    _check_aux_sharding(arg_shapes, 3, sink_index=7, has_sinks=has_sinks,
                        head_axis=specs[0][hi])
    if dropout_p and any(x is not None for x in specs[0]):
        raise ValueError("fa: dropout sharding is unsupported (global RNG offsets required)")
    if has_sinks and any(x is not None for x in specs[0]):
        raise ValueError("fa: sharded sinks are unsupported")
    return (NamedSharding(mesh, PartitionSpec(*specs[0])),
            NamedSharding(mesh, PartitionSpec(*lse_spec)))


def _fa_fwd_partition(scale, causal, layout, wl, wr, softcap, has_alibi,
                      alibi_batch, has_sinks, sink_type, has_varlen, msq, msk,
                      dropout_p, mesh, arg_shapes, result_shapes):
    if not has_varlen:
        arg_shapes = _canonical_dense_shapes(mesh, arg_shapes, layout, has_sinks=has_sinks)
    arg_shardings = tuple(a.sharding for a in arg_shapes)
    out_shardings = _fa_fwd_infer_sharding(
        scale, causal, layout, wl, wr, softcap, has_alibi, alibi_batch,
        has_sinks, sink_type, has_varlen, msq, msk, dropout_p,
        mesh, arg_shapes, result_shapes)
    impl = functools.partial(_fa_fwd_impl, scale=scale, causal=causal, layout=layout,
                             wl=wl, wr=wr, softcap=softcap, has_alibi=has_alibi,
                             alibi_batch=alibi_batch, has_sinks=has_sinks,
                             sink_type=sink_type, has_varlen=has_varlen,
                             msq=msq, msk=msk, dropout_p=dropout_p)
    return mesh, impl, out_shardings, arg_shardings


def _fa_fwd_propagate(*args):
    # static options precede (mesh, user result shapes).
    return jax.tree.map(lambda shape: shape.sharding, args[-1])


def _attention_sharding_rule(arg_types, *, layout, backward=False,
                             has_varlen=False, has_sinks=False):
    """稠密注意力仅支持批次和头的本地分片，每个分片须包含完整 KV 头组。

    q/k/v/o/do 的批次和头分片须对齐，LSE 序列轴及辅助元数据保持复制。
    打包变长数据不做本地切分，以保留序列归属和累计长度基准；
    dropout 和 sinks 路径不支持分片。
    反向 dk/dv 按查询头展开，由外层规则在本地 KV 分片内按组归并。
    """
    shapes = [tuple(mlir.ir.RankedTensorType(t).shape) for t in arg_types]
    hi = 1 if has_varlen or layout == 0 else 2
    h, hk = shapes[0][hi], shapes[1][hi]
    if hk <= 0 or h % hk or shapes[2][hi] != hk:
        raise ValueError("fa: invalid GQA head sizes")
    if has_varlen:
        inputs = ["t (hk group) d", "tk hk d", "tk hk dv"]
        outputs = ["t (hk group) dv", "(hk group) t"]
        replicated = ["t", "tk", "hk", "group", "d", "dv"]
    else:
        q = "b sq (hk group) d" if layout == 1 else "b (hk group) sq d"
        k = "b sk hk d" if layout == 1 else "b hk sk d"
        v = "b sk hk dv" if layout == 1 else "b hk sk dv"
        o = q.replace(" d", " dv")
        inputs = [q, k, v]
        outputs = [o, "b (hk group) sq"]
        replicated = ["sq", "sk", "group", "d", "dv"]
        if backward:
            inputs += [o, o, "b (hk group) sq"]
            outputs = [q, k.replace("hk", "(hk group)"),
                       v.replace("hk", "(hk group)"), "b (hk group) sr"]
            replicated.append("sr")
    sink_index = 10 if backward else 7
    for i in range(len(inputs), len(shapes)):
        if i == sink_index and has_sinks:
            if shapes[i] != (h,):
                raise ValueError("fa: sinks must have shape (query_heads,)")
            inputs.append("(hk group)")
        else:
            factors = [f"aux{i}d{j}" for j in range(len(shapes[i]))]
            inputs.append(" ".join(factors))
            replicated.extend(factors)
    # Shardy 的特殊因子索引须按因子首次出现的顺序排列。
    factors = dict.fromkeys(" ".join(inputs + outputs)
                            .replace("(", " ").replace(")", " ").split())
    return (", ".join(inputs) + " -> " + ", ".join(outputs),
            dict(group=h // hk, need_replication_factors=tuple(
                factor for factor in factors if factor in replicated)))


def _fa_fwd_sharding_rule(scale, causal, layout, wl, wr, softcap, has_alibi,
                          alibi_batch, has_sinks, sink_type, has_varlen, msq,
                          msk, dropout_p, mesh, arg_types, result_types):
    return _attention_sharding_rule(arg_types, layout=layout,
                                    has_varlen=has_varlen, has_sinks=has_sinks)


_fa_fwd_cp = custom_partitioning(_fa_fwd_impl, static_argnums=tuple(range(10, 24)))
_fa_fwd_cp.def_partition(
    infer_sharding_from_operands=_fa_fwd_infer_sharding,
    partition=_fa_fwd_partition,
    propagate_user_sharding=_fa_fwd_propagate,
    sharding_rule=_fa_fwd_sharding_rule)
_fa_fwd_cp = _sequential_partitioned(_fa_fwd_cp, 10)


def _fa_fwd_batching(batched_args, batch_dims, **params):
    return _sequential_batch(_fa_p, batched_args, batch_dims, **params)


batching.primitive_batchers[_fa_p] = _fa_fwd_batching


# ------------------------------- bwd ----------------------------------------
_fa_bwd_p = core.Primitive("fa_bwd_v0")
_fa_bwd_p.multiple_results = True
_fa_bwd_p.def_impl(functools.partial(dispatch.apply_primitive, _fa_bwd_p))


def _fa_bwd_abstract(*avals, **params):
    """稠密反向 dq/dk/dv 的序列行数须向上对齐到 128。

    对外返回时仅裁去梯度的补齐行，dsm 保留补齐形状；
    批处理规则须保留全部四个结果，不重复裁剪。
    """
    q, k, v = avals[0], avals[1], avals[2]
    layout = params.get("layout", 1)
    b, h, hk, sq, sk, d, dv = _shapes(q, k, v, layout)
    sq_r, sk_r = _rm(sq, 128), _rm(sk, 128)
    if layout == 1:
        dq_aval = q.update(shape=(b, sq_r, h, d), weak_type=False)
        dk_aval = core.ShapedArray((b, sk_r, h, d), q.dtype)
        dv_aval = core.ShapedArray((b, sk_r, h, dv), q.dtype)
    else:
        dq_aval = q.update(shape=(b, h, sq_r, d), weak_type=False)
        dk_aval = core.ShapedArray((b, h, sk_r, d), q.dtype)
        dv_aval = core.ShapedArray((b, h, sk_r, dv), q.dtype)
    dsm_aval = core.ShapedArray((b, h, sq_r), jnp.float32)
    return [dq_aval, dk_aval, dv_aval, dsm_aval]


_fa_bwd_p.def_abstract_eval(_fa_bwd_abstract)


def _fa_bwd_lower(ctx, q, k, v, o, do, lse, rng, dbg, sem, alibi, saux, *,
                  scale, causal, layout, wl, wr, softcap, has_alibi, alibi_batch,
                  has_sinks, sink_type, dropout_p, deterministic=0):
    q_aval, k_aval, v_aval = ctx.avals_in[0], ctx.avals_in[1], ctx.avals_in[2]
    b, h, hk, sq, sk, d, dv = _shapes(q_aval, k_aval, v_aval, layout)
    dtype = 1 if q_aval.dtype == jnp.bfloat16 else 0
    opaque = _pack_opaque(b, h, hk, sq, sk, d, dv, scale, causal, layout, dtype,
                          wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
                          sink_type, 0, 0, 0, 0, 0, float(dropout_p),
                          deterministic=int(deterministic))

    def layouts(a):
        return mlir.dense_int_array(tuple(range(a.ndim - 1, -1, -1)))

    out = mlir.custom_call(
        BWD_TARGET,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[q, k, v, o, do, lse, rng, dbg, sem, alibi, saux],
        backend_config=opaque,
        operand_layouts=[layouts(a) for a in ctx.avals_in],
        result_layouts=[layouts(a) for a in ctx.avals_out],
    )
    return list(out.results)


mlir.register_lowering(_fa_bwd_p, _fa_bwd_lower, platform="rocm")


# ------------------------- 形态 A: XLA FFI 前端 ------------------------------
def _ffi_attrs(**kw):
    """FFI 属性使用与 C++ 约定一致的 numpy int32/float32 标量。

    普通形状由缓冲推导，变长最大序列长度 msq/msk 须显式传入。
    """
    d = dict(causal=np.int32(0), layout=np.int32(1), dtype=np.int32(0),
             wl=np.int32(-1), wr=np.int32(-1), has_alibi=np.int32(0),
             alibi_batch=np.int32(0), has_sinks=np.int32(0), sink_type=np.int32(0),
             has_varlen=np.int32(0), msq=np.int32(0), msk=np.int32(0),
             vbwd_mode=np.int32(0), lse_unpadded=np.int32(1), is_fp8=np.int32(0),
             deterministic=np.int32(0), scale=np.float32(0), softcap=np.float32(0),
             dropout_p=np.float32(0))
    d.update(kw)
    return d


def _fa_fwd_ffi(q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, scale, causal,
                layout, wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
                sink_type, has_varlen, msq, msk, dropout_p):
    if has_varlen:
        total_q, h, d = q.shape
        dv = v.shape[2]
        o_spec = jax.ShapeDtypeStruct((total_q, h, dv), q.dtype)
        lse_spec = jax.ShapeDtypeStruct((h, total_q), jnp.float32)
    else:
        b, h, hk, sq, sk, d, dv = _shapes(
            jax.core.ShapedArray(q.shape, q.dtype),
            jax.core.ShapedArray(k.shape, k.dtype),
            jax.core.ShapedArray(v.shape, v.dtype), layout)
        o_shape = (b, sq, h, dv) if layout == 1 else (b, h, sq, dv)
        o_spec = jax.ShapeDtypeStruct(o_shape, q.dtype)
        lse_spec = jax.ShapeDtypeStruct((b, h, sq), jnp.float32)
    fn = jax.ffi.ffi_call(FWD_FFI_TARGET, (o_spec, lse_spec),
                          vmap_method="sequential")
    return fn(q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk,
              **_ffi_attrs(scale=np.float32(scale), causal=np.int32(int(causal)),
                           layout=np.int32(int(layout)), wl=np.int32(int(wl)),
                           wr=np.int32(int(wr)), softcap=np.float32(softcap),
                           has_alibi=np.int32(int(has_alibi)),
                           alibi_batch=np.int32(int(alibi_batch)),
                           has_sinks=np.int32(int(has_sinks)),
                           sink_type=np.int32(int(sink_type)),
                           has_varlen=np.int32(int(has_varlen)),
                           msq=np.int32(int(msq)), msk=np.int32(int(msk)),
                           dropout_p=np.float32(dropout_p),
                           dtype=np.int32(1 if q.dtype == jnp.bfloat16 else 0)))


def _fa_bwd_ffi(q, k, v, o, do, lse, rng, dbg, sem, alibi, saux, scale, causal,
                layout, wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
                sink_type, dropout_p, deterministic=0):
    b, h, hk, sq, sk, d, dv = _shapes(
        jax.core.ShapedArray(q.shape, q.dtype),
        jax.core.ShapedArray(k.shape, k.dtype),
        jax.core.ShapedArray(v.shape, v.dtype), layout)
    sq_r, sk_r = _rm(sq, 128), _rm(sk, 128)
    if layout == 1:
        dq_spec = jax.ShapeDtypeStruct((b, sq_r, h, d), q.dtype)
        dk_spec = jax.ShapeDtypeStruct((b, sk_r, h, d), q.dtype)
        dv_spec = jax.ShapeDtypeStruct((b, sk_r, h, dv), q.dtype)
    else:
        dq_spec = jax.ShapeDtypeStruct((b, h, sq_r, d), q.dtype)
        dk_spec = jax.ShapeDtypeStruct((b, h, sk_r, d), q.dtype)
        dv_spec = jax.ShapeDtypeStruct((b, h, sk_r, dv), q.dtype)
    dsm_spec = jax.ShapeDtypeStruct((b, h, sq_r), jnp.float32)
    fn = jax.ffi.ffi_call(BWD_FFI_TARGET, (dq_spec, dk_spec, dv_spec, dsm_spec),
                          vmap_method="sequential")
    dq, dk, dv, dsm = fn(q, k, v, o, do, lse, rng, dbg, sem, alibi, saux,
                         **_ffi_attrs(scale=np.float32(scale),
                                      causal=np.int32(int(causal)),
                                      layout=np.int32(int(layout)),
                                      dtype=np.int32(1 if q.dtype == jnp.bfloat16 else 0),
                                      wl=np.int32(int(wl)), wr=np.int32(int(wr)),
                                      softcap=np.float32(softcap),
                                      has_alibi=np.int32(int(has_alibi)),
                                      alibi_batch=np.int32(int(alibi_batch)),
                                      has_sinks=np.int32(int(has_sinks)),
                                      sink_type=np.int32(int(sink_type)),
                                      dropout_p=np.float32(dropout_p),
                                      deterministic=np.int32(int(deterministic))))
    if layout == 1:
        return dq[:, :sq], dk[:, :sk], dv[:, :sk], dsm
    return dq[:, :, :sq], dk[:, :, :sk], dv[:, :, :sk], dsm


def _fa_bwd_impl(q, k, v, o, do, lse, rng, dbg, sem, alibi, saux, scale, causal,
                 layout, wl, wr, softcap, has_alibi, alibi_batch, has_sinks,
                 sink_type, dropout_p, deterministic=0):
    if _USE_FFI:      # 形态 A: XLA FFI
        return _fa_bwd_ffi(q, k, v, o, do, lse, rng, dbg, sem, alibi, saux, scale,
                           causal, layout, wl, wr, softcap, has_alibi, alibi_batch,
                           has_sinks, sink_type, dropout_p, deterministic)
    dq, dk, dv, dsm = _fa_bwd_p.bind(
        q, k, v, o, do, lse, rng, dbg, sem, alibi, saux,
        scale=scale, causal=causal, layout=layout, wl=wl, wr=wr, softcap=softcap,
        has_alibi=has_alibi, alibi_batch=alibi_batch, has_sinks=has_sinks,
        sink_type=sink_type, dropout_p=dropout_p, deterministic=deterministic)
    if layout == 1:
        sq, sk = q.shape[1], k.shape[1]
        return dq[:, :sq], dk[:, :sk], dv[:, :sk], dsm
    sq, sk = q.shape[2], k.shape[2]
    return dq[:, :, :sq], dk[:, :, :sk], dv[:, :, :sk], dsm


def _fa_bwd_infer_sharding(scale, causal, layout, wl, wr, softcap, has_alibi,
                           alibi_batch, has_sinks, sink_type, dropout_p,
                           deterministic, mesh, arg_shapes, result_shapes):
    arg_shapes = _canonical_dense_shapes(
        mesh, arg_shapes, layout, backward=True, has_sinks=has_sinks)
    specs, hi, lse_spec = _dense_partition_specs(
        mesh, arg_shapes, layout, backward=True)
    _check_aux_sharding(arg_shapes, 6, sink_index=10, has_sinks=has_sinks,
                        head_axis=specs[0][hi])
    if dropout_p and any(x is not None for x in specs[0]):
        raise ValueError("fa: dropout sharding is unsupported (global RNG offsets required)")
    if has_sinks and any(x is not None for x in specs[0]):
        raise ValueError("fa: sharded sinks are unsupported")
    return (NamedSharding(mesh, PartitionSpec(*specs[0])),
            NamedSharding(mesh, PartitionSpec(*specs[1])),
            NamedSharding(mesh, PartitionSpec(*specs[2])),
            NamedSharding(mesh, PartitionSpec(*lse_spec)))


def _fa_bwd_partition(scale, causal, layout, wl, wr, softcap, has_alibi,
                      alibi_batch, has_sinks, sink_type, dropout_p,
                      deterministic, mesh, arg_shapes, result_shapes):
    arg_shapes = _canonical_dense_shapes(
        mesh, arg_shapes, layout, backward=True, has_sinks=has_sinks)
    arg_shardings = tuple(a.sharding for a in arg_shapes)
    out_shardings = _fa_bwd_infer_sharding(
        scale, causal, layout, wl, wr, softcap, has_alibi, alibi_batch,
        has_sinks, sink_type, dropout_p, deterministic, mesh, arg_shapes,
        result_shapes)
    impl = functools.partial(_fa_bwd_impl, scale=scale, causal=causal, layout=layout,
                             wl=wl, wr=wr, softcap=softcap, has_alibi=has_alibi,
                             alibi_batch=alibi_batch, has_sinks=has_sinks,
                             sink_type=sink_type, dropout_p=dropout_p,
                             deterministic=deterministic)
    return mesh, impl, out_shardings, arg_shardings


def _fa_bwd_propagate(*args):
    return jax.tree.map(lambda shape: shape.sharding, args[-1])


def _fa_bwd_sharding_rule(scale, causal, layout, wl, wr, softcap, has_alibi,
                          alibi_batch, has_sinks, sink_type, dropout_p,
                          deterministic, mesh, arg_types, result_types):
    return _attention_sharding_rule(arg_types, layout=layout, backward=True,
                                    has_sinks=has_sinks)


_fa_bwd_cp = custom_partitioning(_fa_bwd_impl, static_argnums=tuple(range(11, 23)))
_fa_bwd_cp.def_partition(
    infer_sharding_from_operands=_fa_bwd_infer_sharding,
    partition=_fa_bwd_partition,
    propagate_user_sharding=_fa_bwd_propagate,
    sharding_rule=_fa_bwd_sharding_rule)
_fa_bwd_cp = _sequential_partitioned(_fa_bwd_cp, 11)


def _fa_bwd_batching(batched_args, batch_dims, **params):
    return _sequential_batch(_fa_bwd_p, batched_args, batch_dims, **params)


batching.primitive_batchers[_fa_bwd_p] = _fa_bwd_batching


# ------------------------------- 入口 ----------------------------------------
def _sink_type_of(x):
    if x is None:
        return 0
    if x.dtype == jnp.float32:
        return 1
    if x.dtype == jnp.float16:
        return 2
    if x.dtype == jnp.bfloat16:
        return 3
    return 0


def _alibi_metadata(alibi_slopes, b, h):
    """Validate alibi's enabled-kernel contract and derive its two flags.

    The C++ ABI assumes a contiguous rank-1 ``(h,)`` or rank-2 ``(b,h)`` fp32
    buffer; JAX arrays do not expose a portable stride API, so callers must
    provide the standard dense form.
    """
    if alibi_slopes is None:
        return _placeholder(), 0, 0
    if alibi_slopes.dtype != jnp.float32:
        raise TypeError(
            "alibi_slopes 必须是 float32，"
            f"得到 {alibi_slopes.dtype}")
    if alibi_slopes.ndim == 1:
        if tuple(alibi_slopes.shape) != (h,):
            raise ValueError(
                f"alibi_slopes rank-1 shape 必须是 ({h},)，"
                f"得到 {tuple(alibi_slopes.shape)}")
        return alibi_slopes, 1, 0
    if alibi_slopes.ndim == 2:
        if tuple(alibi_slopes.shape) != (b, h):
            raise ValueError(
                f"alibi_slopes rank-2 shape 必须是 ({b}, {h})，"
                f"得到 {tuple(alibi_slopes.shape)}")
        return alibi_slopes, 1, 1
    raise ValueError(
        "alibi_slopes 只能是 rank-1 (h,) 或 rank-2 (b,h)，"
        f"得到 rank={alibi_slopes.ndim}")


def fa_fwd(q, k, v, *, causal: bool = False, softmax_scale=None, layout: int = 1,
           window_size=(-1, -1), softcap: float = 0.0, alibi_slopes=None,
           sinks=None, dropout_p: float = 0.0, seed: int = 0):
    """稠密注意力前向，输入为 fp16/bf16，支持 MHA/GQA/MQA。

    layout=1：q 为 (b,sq,h,d)，k/v 为 (b,sk,hk,d/dv)；
    layout=0：交换序列轴与头轴。h 须为 hk 的整数倍。
    返回同布局的 o 和 float32 LSE (b,h,sq)。
    支持因果、滑动窗口及 dropout；seed 指定随机种子。
    sinks 为每个查询头的辅助标量，类型为 fp16/bf16/fp32。
    softcap>0 或提供 alibi_slopes 时抛出 ValueError。
    需要自动反向求导时使用 fa_fwd_custom。
    """
    _register()
    b, h, _, _, _, _, _ = _shapes(
        jax.core.ShapedArray(q.shape, q.dtype),
        jax.core.ShapedArray(k.shape, k.dtype),
        jax.core.ShapedArray(v.shape, v.dtype), layout)
    alibi, has_alibi, alibi_batch = _alibi_metadata(alibi_slopes, b, h)
    if softcap > 0.0:
        raise ValueError(
            "当前 HIPC 接入不支持 softcap；算子组已确认后续不再支持。")
    if alibi_slopes is not None:
        raise ValueError(
            "当前 HIPC 接入不支持 alibi；算子组已确认后续不再支持。")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    saux = _placeholder() if sinks is None else sinks
    if dropout_p > 0.0:
        rng = _rng_operand(seed)
        return _fa_fwd_dropout_impl(q, k, v, rng, float(dropout_p),
                                    float(softmax_scale), bool(causal),
                                    int(layout), int(window_size[0]),
                                    int(window_size[1]))
    c = _contract()
    o, lse = _fa_fwd_cp(
        q, k, v, c["rng"], c["dbg"], c["sem"], alibi, saux,
        _cu_placeholder(), _cu_placeholder(),
        float(softmax_scale), bool(causal), int(layout),
        int(window_size[0]), int(window_size[1]), float(softcap),
        int(has_alibi), int(alibi_batch),
        int(sinks is not None), _sink_type_of(sinks),
        has_varlen=0, msq=0, msk=0, dropout_p=0.0)
    return o, lse


_dropout_rng_p = core.Primitive("fa_fwd_dropout_v0")


def _fa_fwd_dropout_impl(q, k, v, rng, dropout_p, scale, causal, layout, wl, wr):
    c = _contract()
    return _fa_fwd_cp(q, k, v, rng, c["dbg"], c["sem"], _placeholder(),
                       _placeholder(), _cu_placeholder(), _cu_placeholder(),
                       scale=scale, causal=causal, layout=layout, wl=wl, wr=wr,
                       softcap=0.0, has_alibi=0, alibi_batch=0, has_sinks=0,
                       sink_type=0, has_varlen=0, msq=0, msk=0,
                       dropout_p=dropout_p)


def _cu_placeholder():
    return jnp.zeros((2,), jnp.int32)


def _length_array(lengths, name):
    """转换长度元数据，避免隐式整数缩窄。

    已有数组或追踪值须为 int32，Python 整数序列须先检查 int32 范围。
    """
    if isinstance(lengths, (list, tuple)):
        for value in lengths:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must contain int32 lengths, not bool")
            if isinstance(value, (int, np.integer)):
                if not -(2 ** 31) <= value < 2 ** 31:
                    raise ValueError(f"{name} values must fit int32")
            elif (getattr(value, "dtype", None) != jnp.int32
                  or getattr(value, "ndim", None) != 0):
                # Scalar int32 tracers also occur when jit maps a Python list.
                raise TypeError(f"{name} must contain int32 integer lengths")
        return jnp.asarray(lengths, dtype=jnp.int32)
    if getattr(lengths, "dtype", None) != jnp.int32:
        raise TypeError(f"{name} must have dtype int32 before conversion")
    return jnp.asarray(lengths)


def _validated_lengths(q_lengths, k_lengths, *, q_capacity, k_capacity,
                       max_q, max_k, cumulative=True, require_equal=False):
    """返回运行时校验后的设备长度副本。

    原生校验会复制长度到主机并同步设备流，不在 Python 中读取设备值。
    后续计算必须消费校验副本，维持校验先于内核执行的数据依赖；
    该要求同样适用于 jit、vmap 和旧式调用路径。
    """
    for name, lengths in (("q_lengths", q_lengths), ("k_lengths", k_lengths)):
        if lengths.ndim != 1 or lengths.dtype != jnp.int32:
            raise TypeError(f"{name} must be a rank-1 int32 array")
    if q_lengths.shape != k_lengths.shape or q_lengths.shape[0] < (2 if cumulative else 1):
        raise ValueError("q/k lengths must have matching nonempty batch shapes")
    _register()
    specs = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype)
                  for x in (q_lengths, k_lengths))
    return jax.ffi.ffi_call("fa_validate_lengths_ffi", specs,
                            vmap_method="sequential")(
        q_lengths, k_lengths, q_capacity=np.int32(q_capacity),
        k_capacity=np.int32(k_capacity), max_q=np.int32(max_q),
        max_k=np.int32(max_k), cumulative=np.int32(bool(cumulative)),
        require_equal=np.int32(bool(require_equal)))


def _varlen_metadata(q, k, v, cuq, cuk, max_q, max_k, softmax_scale, window_size):
    """Validate static metadata without reading cumulative device lengths."""
    if any(x.ndim != 3 for x in (q, k, v)):
        raise ValueError("varlen q/k/v must be rank-3 packed (total, heads, dim)")
    if q.dtype not in (jnp.float16, jnp.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("varlen q/k/v must have the same fp16/bf16 dtype")
    tq, h, d = q.shape
    tk, hk, dk = k.shape
    if min(tq, tk, h, hk) <= 0 or h % hk or k.shape[:2] != v.shape[:2] or d != dk:
        raise ValueError("varlen q/k/v capacities, head counts, or dimensions do not match")
    # 原生反向行步长按头维度向上对齐到 32，紧凑结果缓冲须与该步长一致。
    if any(dim <= 0 or dim > 256 or dim % 32 for dim in (d, v.shape[2])):
        raise ValueError("varlen head dimensions must be positive multiples of 32 <= 256")
    if max(tq, tk, h * d, h * v.shape[2]) > np.iinfo(np.int32).max:
        raise ValueError("varlen capacities and row strides must fit int32")
    for name, cu in (("cu_seqlens_q", cuq), ("cu_seqlens_k", cuk)):
        if cu.ndim != 1 or cu.dtype != jnp.int32:
            raise TypeError(f"{name} must be a rank-1 int32 array")
    if cuq.shape != cuk.shape or cuq.shape[0] < 2:
        raise ValueError("varlen cumulative lengths must have matching (batch+1,) shapes")
    for name, value, capacity in (("max_seqlen_q", max_q, tq), ("max_seqlen_k", max_k, tk)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or not 0 < value <= min(capacity, np.iinfo(np.int32).max)):
            raise ValueError(f"{name} must be a positive static int32 integer <= capacity")
    batch = cuq.shape[0] - 1
    dv = v.shape[2]
    products = (tq * h * d, tq * h * dv, tk * h * d, tk * h * dv,
                batch * h * _rm(int(max_q), 128),
                batch * h * _rm(int(max_k), 128),
                _rm(int(max_q), 128) * h * d, _rm(int(max_k), 128) * h * dv)
    if max(products) > np.iinfo(np.int32).max:
        raise ValueError("varlen native buffer sizes and expanded gradient offsets must fit int32")
    if len(window_size) != 2 or any(
            not isinstance(x, (int, np.integer)) or x < -1
            or x > np.iinfo(np.int32).max for x in window_size):
        raise ValueError("window_size must contain two static integers >= -1")
    with np.errstate(over="ignore", under="ignore"):
        scale = float(np.float32(d ** -0.5 if softmax_scale is None else softmax_scale))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("varlen softmax_scale must be finite and positive in float32")
    return int(max_q), int(max_k), scale, tuple(int(x) for x in window_size)


def _varlen_static_lengths(lengths, batch, capacity, max_length, name):
    if not isinstance(lengths, (tuple, list, np.ndarray)):
        raise ValueError(f"{name} must be explicit static host lengths for deterministic varlen AD")
    values = np.asarray(lengths)
    if values.ndim != 1 or values.shape != (batch,) or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain one integer length per sequence")
    values = tuple(int(x) for x in values)
    if any(x <= 0 or x > max_length for x in values) or sum(values) > capacity:
        raise ValueError(f"{name} must be positive, bounded by max_seqlen and fit capacity")
    return values


def _varlen_checked_static_cu(cu, lengths, capacity, max_length):
    expected = jnp.asarray(np.concatenate(([0], np.cumsum(lengths))), jnp.int32)
    checked, _ = _validated_lengths(
        cu, expected, q_capacity=capacity, k_capacity=capacity,
        max_q=max_length, max_k=max_length, require_equal=True)
    return checked


def _fa_varlen_forward(q, k, v, cuq, cuk, sinks, options):
    max_q, max_k, scale, causal, window, _deterministic, _lens_q, _lens_k = options
    c = _contract()
    return _fa_fwd_cp(
        q, k, v, c["rng"], c["dbg"], c["sem"], _placeholder(),
        _placeholder() if sinks is None else sinks, cuq, cuk,
        scale, causal, 1, window[0], window[1], 0.0,
        0, 0, int(sinks is not None), _sink_type_of(sinks),
        has_varlen=1, msq=max_q, msk=max_k, dropout_p=0.0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _fa_varlen_ad(q, k, v, cuq, cuk, sinks, options):
    return _fa_varlen_forward(q, k, v, cuq, cuk, sinks, options)


def _fa_varlen_ad_fwd(q, k, v, cuq, cuk, sinks, options):
    if sinks is not None:
        raise ValueError("fa_fwd_varlen: automatic differentiation with sinks is not supported")
    o, lse = _fa_varlen_forward(q, k, v, cuq, cuk, sinks, options)
    return (o, lse), (q, k, v, o, lse, cuq, cuk)


def _fa_varlen_ad_bwd(options, residual, cotangents):
    q, k, v, o, lse, cuq, cuk = residual
    max_q, max_k, scale, causal, window, deterministic, lens_q, lens_k = options
    # The public wrapper explicitly stop-gradients LSE; it is not a loss output.
    dq, dk, dv = fa_varlen_bwd(
        q, k, v, o, cotangents[0], lse, cuq, cuk, max_q, max_k,
        causal=causal, softmax_scale=scale, window_size=window,
        mode="dense_loop" if deterministic else "kernel",
        deterministic=deterministic, seq_lens_q=lens_q, seq_lens_k=lens_k)
    return dq, dk, dv, None, None, None


_fa_varlen_ad.defvjp(_fa_varlen_ad_fwd, _fa_varlen_ad_bwd)


def fa_fwd_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                  *, causal: bool = False, softmax_scale=None, window_size=(-1, -1),
                  sinks=None, deterministic: bool = False, seq_lens_q=None,
                  seq_lens_k=None):
    """打包变长注意力，支持 q/k/v 一阶反向求导。

    q/k/v 为 (capacity_q,h,d)/(capacity_k,hk,d)/(capacity_k,hk,dv)，
    类型同为 fp16/bf16，h 须为 hk 的整数倍；d/dv 为不超过 256 的正 32 倍数。
    cu_seqlens_q/k 为 int32[b+1]，从零开始严格递增，末值不超过容量；
    每段长度不超过静态 max_seqlen，缩放值在 float32 中须为有限正数。
    非因果全局注意力允许独立 Q/K 长度；因果或窗口注意力要求逐序列等长。
    返回 o (capacity_q,h,dv) 和停止梯度的 float32 LSE (h,capacity_q)。
    sinks 仅支持前向，带 sinks 求导会报错；不支持前向模式或高阶求导。
    deterministic=True 要求相等的静态 seq_lens_q/k，并与设备长度核对；
    否则不应提供 seq_lens_q/k。运行时校验存在主机复制和设备流同步开销。
    """
    max_q, max_k, scale, window = _varlen_metadata(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale, window_size)
    if sinks is not None and (sinks.shape != (q.shape[1],)
                              or sinks.dtype not in (jnp.float16, jnp.bfloat16, jnp.float32)):
        raise ValueError("varlen sinks must have shape (query_heads,) and fp16/bf16/fp32 dtype")
    lens_q = lens_k = None
    if deterministic:
        batch = cu_seqlens_q.shape[0] - 1
        lens_q = _varlen_static_lengths(seq_lens_q, batch, q.shape[0], max_q, "seq_lens_q")
        lens_k = _varlen_static_lengths(seq_lens_k, batch, k.shape[0], max_k, "seq_lens_k")
        if lens_q != lens_k:
            raise ValueError("deterministic varlen AD only supports equal Q/K sequence lengths")
    elif seq_lens_q is not None or seq_lens_k is not None:
        raise ValueError("seq_lens_q/k are only used with deterministic=True")
    cuq, cuk = _validated_lengths(
        cu_seqlens_q, cu_seqlens_k, q_capacity=q.shape[0], k_capacity=k.shape[0],
        max_q=max_q, max_k=max_k, require_equal=bool(causal) or window != (-1, -1))
    if deterministic:
        cuq = _varlen_checked_static_cu(cuq, lens_q, q.shape[0], max_q)
        cuk = _varlen_checked_static_cu(cuk, lens_k, k.shape[0], max_k)
    options = (max_q, max_k, scale, bool(causal), window, bool(deterministic), lens_q, lens_k)
    o, lse = _fa_varlen_ad(q, k, v, cuq, cuk, sinks, options)
    return o, jax.lax.stop_gradient(lse)


@functools.partial(jax.custom_vjp, nondiff_argnames=(
    "causal", "softmax_scale", "layout", "window_size", "softcap",
    "has_alibi", "alibi_batch", "has_sinks", "sink_type", "dropout_p", "seed",
    "deterministic"))
def fa_fwd_custom(q, k, v, causal: bool = False, softmax_scale=None, layout: int = 1,
                  window_size=(-1, -1), softcap: float = 0.0, alibi_slopes=None,
                  sinks=None, has_alibi: int = 0, alibi_batch: int = 0,
                  has_sinks: int = 0, sink_type: int = 0, dropout_p: float = 0.0,
                  seed: int = 0, deterministic: bool = False):
    """稠密注意力的一阶反向求导入口，输入布局和输出形状同 fa_fwd。

    返回 (o,lse)，反向计算 q/k/v 梯度，不对辅助参数求导。
    LSE 保存前向状态，反向忽略其余切，不应作为可微损失输出。
    deterministic=True 选择确定性反向；不支持前向模式或高阶求导。
    """
    out, _res = _fa_fwd_custom_fwd(
        q, k, v, causal=causal, softmax_scale=softmax_scale, layout=layout,
        window_size=window_size, softcap=softcap, alibi_slopes=alibi_slopes,
        sinks=sinks, has_alibi=has_alibi, alibi_batch=alibi_batch,
        has_sinks=has_sinks, sink_type=sink_type, dropout_p=dropout_p,
        seed=seed, deterministic=deterministic)
    return out


def _fa_fwd_custom_fwd(q, k, v, causal: bool = False, softmax_scale=None,
                       layout: int = 1, window_size=(-1, -1), softcap: float = 0.0,
                       alibi_slopes=None, sinks=None, has_alibi: int = 0,
                       alibi_batch: int = 0, has_sinks: int = 0, sink_type: int = 0,
                       dropout_p: float = 0.0, seed: int = 0,
                       deterministic: bool = False):
    """(内部) custom_vjp 的 fwd: 返回 ((o, lse), residual)。

    用户请调 fa_fwd_custom (它恒返回 (o, lse)); 本函数仅作为 defvjp 的 fwd 使用。
    deterministic=True: 反向走确定性路径 (dq 累加顺序固定, 跨运行逐位可复现)。"""
    b, h, _, _, _, _, _ = _shapes(
        jax.core.ShapedArray(q.shape, q.dtype),
        jax.core.ShapedArray(k.shape, k.dtype),
        jax.core.ShapedArray(v.shape, v.dtype), layout)
    alibi, derived_has_alibi, derived_alibi_batch = _alibi_metadata(
        alibi_slopes, b, h)
    # 对外仍保留兼容参数，但真实 flags 必须来自 tensor，避免调用者默认值
    # (has_alibi=0/alibi_batch=0) 把有效 alibi 静默关闭。
    has_alibi = derived_has_alibi
    alibi_batch = derived_alibi_batch
    _register()
    if alibi_slopes is not None:
        raise ValueError(
            "当前 HIPC 接入不支持 alibi；算子组已确认后续不再支持。")
    if softcap > 0.0:
        raise ValueError(
            "当前 HIPC 接入不支持 softcap；算子组已确认后续不再支持。")
    saux = _placeholder() if sinks is None else sinks
    if dropout_p > 0.0:
        rng = _rng_operand(seed)
        o, lse = _fa_fwd_dropout_impl(q, k, v, rng, float(dropout_p),
                                      float(softmax_scale if softmax_scale is not None
                                            else q.shape[-1] ** -0.5),
                                      bool(causal), int(layout),
                                      int(window_size[0]), int(window_size[1]))
        return (o, lse), (q, k, v, o, lse, alibi, saux, rng,
                             float(dropout_p), int(has_alibi), int(alibi_batch))
    c = _contract()
    o, lse = _fa_fwd_cp(
        q, k, v, c["rng"], c["dbg"], c["sem"], alibi, saux,
        _cu_placeholder(), _cu_placeholder(),
        float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5),
        bool(causal), int(layout), int(window_size[0]), int(window_size[1]),
        float(softcap), int(has_alibi), int(alibi_batch), int(has_sinks),
        int(sink_type), has_varlen=0, msq=0, msk=0, dropout_p=0.0)
    return (o, lse), (q, k, v, o, lse, alibi, saux, c["rng"],
                       0.0, int(has_alibi), int(alibi_batch))


def _fa_custom_bwd(causal, softmax_scale, layout, window_size, softcap,
                   has_alibi, alibi_batch, has_sinks, sink_type, dropout_p, seed,
                   deterministic, res, ct):
    q, k, v, o, lse, alibi, saux, rng, dp, derived_has_alibi, derived_alibi_batch = res
    has_alibi = derived_has_alibi
    alibi_batch = derived_alibi_batch
    do = ct[0]
    _register()
    c = _contract()
    b, h, hk, sq, sk, d, dv = _shapes(
        jax.core.ShapedArray(q.shape, q.dtype),
        jax.core.ShapedArray(k.shape, k.dtype),
        jax.core.ShapedArray(v.shape, v.dtype), layout)
    g = h // hk
    dq, dk_exp, dv_exp, _dsm = _fa_bwd_cp(
        q, k, v, o, do, lse, rng, c["dbg"], c["sem"], alibi, saux,
        float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5),
        bool(causal), int(layout), int(window_size[0]), int(window_size[1]),
        float(softcap), int(has_alibi), int(alibi_batch), int(has_sinks),
        int(sink_type), float(dp if dp else dropout_p),
        deterministic=int(bool(deterministic)))
    if layout == 1:
        dk = dk_exp.reshape(b, sk, hk, g, d).sum(3)
        dv = dv_exp.reshape(b, sk, hk, g, dv).sum(3)
    else:
        dk = dk_exp.reshape(b, hk, g, sk, d).sum(2)
        dv = dv_exp.reshape(b, hk, g, sk, dv).sum(2)
    return dq, dk, dv, None, None   # q,k,v + alibi_slopes + sinks 的 cotangent


fa_fwd_custom.defvjp(_fa_fwd_custom_fwd, _fa_custom_bwd)


# ===================== PA / paged KV cache 前向 =====================
_PA_FMT = "<7if11i4i"     # 22×i32 + f32 = 92B, 与 C++ FaPaOpaque 一致
assert struct.calcsize(_PA_FMT) == 92


def _pack_pa_opaque(b, h, hk, sq, sk, d, dv, scale, layout, dtype, causal,
                    wl, wr, page, num_splits, partition_size, mtp, ngroups,
                    bt_stride, total_q=0, total_k=0, mtp_sq=0, is_fp8=0):
    return struct.pack(_PA_FMT, b, h, hk, sq, sk, d, dv, float(scale),
                       int(layout), int(dtype), int(causal), int(wl), int(wr),
                       int(page), int(num_splits), int(partition_size),
                       int(mtp), int(ngroups), int(bt_stride),
                       int(total_q), int(total_k), int(mtp_sq), int(is_fp8))


_PLUGIN_MOD = None


def _arch_raw() -> int:
    """库内 getArch() 的原始 arch id (930=gfx92a, 936=gfx9xx, 938+ 等)。"""
    if not hasattr(_arch_raw, "_v"):
        try:
            _arch_raw._v = int(_PLUGIN_MOD.arch_raw()) if _PLUGIN_MOD else 0
        except Exception:
            _arch_raw._v = 0
    return _arch_raw._v


def _fp8_pa_arch_ok() -> bool:
    """FP8 PagedAttention 门槛: gfx92a(930) 专用分支, 或 gfx938+ 通用。"""
    if not hasattr(_fp8_pa_arch_ok, "_v"):
        try:
            _fp8_pa_arch_ok._v = bool(_PLUGIN_MOD.arch_fp8_pa_ok()) if _PLUGIN_MOD else False
        except Exception:
            _fp8_pa_arch_ok._v = False
    return _fp8_pa_arch_ok._v


def _fp8_arch_ok() -> bool:
    """FP8 内核 arch 门槛探测 (shim 调库内 getArch(); 930 或 >=938 才支持)。"""
    if not hasattr(_fp8_arch_ok, "_v"):
        try:
            _fp8_arch_ok._v = bool(_PLUGIN_MOD.arch_fp8_ok()) if _PLUGIN_MOD else False
        except Exception:
            _fp8_arch_ok._v = False
    return _fp8_arch_ok._v


def _pa_plan(b, ngroups, sk, page, dv_r, num_splits):
    """split-kv 规划。逐行对齐 torch mha_fwd_kvcache_base 的启发式，
    便于与 torch 对拍；num_splits<=1 表示不切分 (单 split, 无 accum 缓冲)。
    返回 (num_splits, partition_size)。"""
    if dv_r not in (64, 128, 512):
        return 1, 0                      # torch: allow_splitkv 的 dtype/尺寸前提
    if num_splits is not None:
        ns = int(num_splits)
        if ns <= 1:
            return 1, 0
        if ns > 1024:
            raise ValueError("fa_fwd_kvcache: num_splits 最大 1024")
        # 分段长度须为页大小的整数倍，且至少为 128。
        ps = -(-sk // (ns * page)) * page
        ps = max(ps, -(-128 // page) * page)
        return ns, ps
    # num_splits=None -> torch 的 partition_size 启发式 (device_cu 固定 128)
    threshold, device_cu = 128, 128
    use_max_regroup = (ngroups > 1 and ngroups not in (29, 16, 8, 4, 2, 9, 7, 5, 3))
    actual_ngroup = 1 if use_max_regroup else ngroups
    partition_size = 0
    if (b * actual_ngroup < threshold and sk >= 1024) or (sk >= 8192):
        if sk <= 1024:
            partition_size = 128
        elif sk <= 2048:
            partition_size = 256
        elif sk <= 32768:
            partition_size = 512
        else:
            partition_size = 1024
        if ngroups == 1:
            partition_size = 1024
        while ngroups > 1 and (b * actual_ngroup * (sk // partition_size)) < threshold:
            if partition_size < 256:
                break
            partition_size //= 2
    if partition_size >= 128 and partition_size % page == 0:
        ns = max(1, sk // partition_size)
        if ns <= 1024:
            return ns, partition_size
    return 1, 0


_pa_p = core.Primitive("fa_fwd_kvcache_v0")


def _pa_abstract(q, kcache, vcache, block_table, seqlens_k, scores_accum,
                 o_accum, dummy, dq_desc, dk_desc, dv_desc, *, b, h, hk, sq, sk,
                 d, dv, scale, causal, layout, dtype, wl, wr, page, num_splits,
                 partition_size, mtp, ngroups, bt_stride, is_fp8):
    if q.dtype not in (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.int8):
        raise TypeError(f"fa_fwd_kvcache 支持 fp16/bf16/fp8_e4m3/int8, 得到 {q.dtype}")
    if is_fp8:
        od = jnp.bfloat16 if dtype == 1 else jnp.float16
        return core.ShapedArray((b, sq, h, dv), od)
    return q.update(shape=(b, sq, h, dv), weak_type=False)


def _fa_pa_ffi(q, kcache, vcache, block_table, seqlens_k, scores_accum, o_accum,
               dummy, dq_desc, dk_desc, dv_desc, *, scale, causal, layout, dtype,
               wl, wr, sk, num_splits, partition_size, mtp, ngroups, bt_stride,
               total_q, total_k, mtp_sq, is_fp8):
    o_spec = jax.ShapeDtypeStruct((q.shape[0], q.shape[1], q.shape[2], vcache.shape[3]),
                                  q.dtype)
    fn = jax.ffi.ffi_call(PA_FFI_TARGET, o_spec, vmap_method="sequential")
    # 分页处理器有独立的属性集合，不能传入 _ffi_attrs 的全部默认属性。
    return fn(q, kcache, vcache, block_table, seqlens_k, scores_accum, o_accum,
              dummy, dq_desc, dk_desc, dv_desc,
              scale=np.float32(scale), causal=np.int32(int(causal)),
              layout=np.int32(int(layout)), dtype=np.int32(int(dtype)),
              wl=np.int32(int(wl)), wr=np.int32(int(wr)),
              sk=np.int32(int(sk)), num_splits=np.int32(int(num_splits)),
              partition_size=np.int32(int(partition_size)),
              mtp=np.int32(int(mtp)), ngroups=np.int32(int(ngroups)),
              bt_stride=np.int32(int(bt_stride)), total_q=np.int32(int(total_q)),
              total_k=np.int32(int(total_k)), mtp_sq=np.int32(int(mtp_sq)),
              is_fp8=np.int32(int(is_fp8)))


def _pa_impl(q, kcache, vcache, block_table, seqlens_k, scores_accum, o_accum,
             dummy, dq_desc, dk_desc, dv_desc, *, b, h, hk, sq, sk, d, dv, scale,
             causal, layout, dtype, wl, wr, page, num_splits, partition_size,
             mtp, ngroups, bt_stride, is_fp8):
    if _USE_FFI:
        return (_fa_pa_ffi(q, kcache, vcache, block_table, seqlens_k, scores_accum,
                           o_accum, dummy, dq_desc, dk_desc, dv_desc, scale=scale,
                           causal=causal, layout=layout, dtype=dtype, wl=wl, wr=wr,
                           sk=sk, num_splits=num_splits,
                           partition_size=partition_size, mtp=mtp, ngroups=ngroups,
                           bt_stride=bt_stride, total_q=(h if is_fp8 else 0),
                           total_k=(1 if is_fp8 else 0), mtp_sq=0, is_fp8=is_fp8),)
    return dispatch.apply_primitive(
        _pa_p, q, kcache, vcache, block_table, seqlens_k, scores_accum,
        o_accum, dummy, dq_desc, dk_desc, dv_desc, b=b, h=h, hk=hk, sq=sq, sk=sk,
        d=d, dv=dv, scale=scale, causal=causal, layout=layout, dtype=dtype,
        wl=wl, wr=wr, page=page, num_splits=num_splits,
        partition_size=partition_size, mtp=mtp, ngroups=ngroups,
        bt_stride=bt_stride, is_fp8=is_fp8)


_pa_p.def_impl(functools.partial(dispatch.apply_primitive, _pa_p))
_pa_p.def_abstract_eval(_pa_abstract)


def _pa_lower(ctx, q, kcache, vcache, block_table, seqlens_k, scores_accum,
              o_accum, dummy, dq_desc, dk_desc, dv_desc, *, b, h, hk, sq, sk, d,
              dv, scale, causal, layout, dtype, wl, wr, page, num_splits,
              partition_size, mtp, ngroups, bt_stride, is_fp8):
    opaque = _pack_pa_opaque(b, h, hk, sq, sk, d, dv, scale, layout, dtype,
                             causal, wl, wr, page, num_splits, partition_size,
                             mtp, ngroups, bt_stride,
                             total_q=(h if is_fp8 else 0),
                             total_k=(1 if is_fp8 else 0), mtp_sq=0,
                             is_fp8=is_fp8)

    def layouts(a):
        return mlir.dense_int_array(tuple(range(a.ndim - 1, -1, -1)))

    out = mlir.custom_call(
        PA_TARGET,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[q, kcache, vcache, block_table, seqlens_k, scores_accum,
                  o_accum, dummy, dq_desc, dk_desc, dv_desc],
        backend_config=opaque,
        operand_layouts=[layouts(a) for a in ctx.avals_in],
        result_layouts=[layouts(a) for a in ctx.avals_out],
    )
    res = out.results
    if not isinstance(res, (list, tuple)):
        res = [res]
    return list(res)


def _eager_init() -> None:
    _contract()
    _placeholder()


_eager_init()


mlir.register_lowering(_pa_p, _pa_lower, platform="rocm")
batching.primitive_batchers[_pa_p] = functools.partial(_sequential_batch, _pa_p)


def fa_fwd_kvcache(q, kcache, vcache, block_table, seqlens_k, max_seqlen_k,
                   *, causal: bool = False, softmax_scale=None,
                   window_size=(-1, -1), num_splits=None, debug_accum=False,
                   q_descale=None, k_descale=None, v_descale=None,
                   out_dtype=jnp.bfloat16, _int8=False):
    """分页 KV 缓存注意力，仅支持前向。

    q 为 (b,sq,h,d)，kcache/vcache 为 (num_blocks,page,hk,d/dv)，
    普通路径使用 fp16/bf16，h 须为 hk 的整数倍。
    block_table 为 int32[b,max_blocks] 物理块号；
    seqlens_k 为 int32[b] 实际 KV 长度，max_seqlen_k 为静态最大长度。
    通常返回 o (b,sq,h,dv)，不返回 LSE；
    debug_accum=True 时额外返回分段统计和输出缓冲。
    num_splits=None 使用启发式划分，1 表示不划分。
    FP8 要求输入为 float8_e4m3fn、提供三个 float32 descale、
    头维度为 128，且通过硬件检查。INT8 路径不可用。
    """
    _register()
    if q.ndim != 4 or kcache.ndim != 4 or vcache.ndim != 4:
        raise ValueError("fa_fwd_kvcache: q/kcache/vcache 均为 4-D")
    is_fp8 = (q.dtype == jnp.float8_e4m3fn)
    if _int8:
        raise NotImplementedError(
            "fa_fwd_kvcache: INT8 分页路径不可用 —— 上游 run_int8_fwd_kvcache 已停止维护, "
            "scales 语义未文档化 (待算子组新接口); FP8 路径 (q/kcache/vcache=float8_e4m3fn) "
            "不受影响。")
    if is_fp8:
        if kcache.dtype != jnp.float8_e4m3fn or vcache.dtype != jnp.float8_e4m3fn:
            raise ValueError("fa_fwd_kvcache: FP8 时 kcache/vcache 须同为 float8_e4m3fn")
        if q_descale is None or k_descale is None or v_descale is None:
            raise ValueError("fa_fwd_kvcache: FP8 需 q/k/v_descale (float32, (b,h))")
        if q.shape[-1] != 128 or vcache.shape[-1] != 128:
            raise ValueError("fa_fwd_kvcache: FP8 PA 仅支持 headdim=128 "
                             "(flash_api.cpp gfx92a 分支 assert)")
        if not _fp8_pa_arch_ok():
            raise ValueError(
                f"fa_fwd_kvcache: 本机 arch={_arch_raw()} 不支持 FP8 PagedAttention "
                "(需 gfx92a(930) 或 gfx938+) —— 请改用 fp16/bf16 路径或在受支持卡上运行")
    b, sq, h, d = q.shape
    nb, page, hk, _ = kcache.shape
    if vcache.shape[:3] != (nb, page, hk):
        raise ValueError("fa_fwd_kvcache: kcache/vcache 的 (blocks,page,hk) 须一致")
    dv = vcache.shape[3]
    if h % hk:
        raise ValueError("fa_fwd_kvcache: h 必须是 hk 的整数倍 (GQA)")
    if block_table.shape[0] != b or seqlens_k.shape != (b,):
        raise ValueError("fa_fwd_kvcache: block_table 须为 (b,max_blocks), "
                         "seqlens_k 须为 (b,)")
    if block_table.dtype != jnp.int32 or seqlens_k.dtype != jnp.int32:
        raise ValueError("fa_fwd_kvcache: block_table/seqlens_k 须为 int32")
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    sk = int(max_seqlen_k)
    dv_r = _rm(_rm(dv, 8), 32)
    ns, ps = _pa_plan(b, h // hk, sk, page, dv_r, num_splits)
    wl, wr = int(window_size[0]), int(window_size[1])
    if wl >= sk:
        wl = -1                      # torch: window_size_left >= max_seqlen_k -> -1
    if causal and wl < 0 and wr < 0:
        wr = 0                       # torch: causal -> window_size_right = 0
    c = _contract()
    scores_accum = o_accum = dummy = c["dbg"]
    if ns > 1:
        # scores_accum 保存各段最大值与指数和，布局为 (2,ns,b,h,sq)；
        # o_accum 保存逐段归一化输出，布局为 (ns,b,sq,h,dv_r)。
        scores_accum = jnp.zeros((2, ns, b, h, sq), jnp.float32)
        o_accum = jnp.zeros((ns, b, sq, h, dv_r),
                            (jnp.int8 if _int8 else
                             (out_dtype if is_fp8 else q.dtype)))
    if is_fp8:
        # 三个 descale 统一为连续 (b,h)，GQA 的 KV 缩放因子按头组展开；
        # 共享步长 (h,1)，旧式 ABI 通过 total_q/total_k 字段传递该步长。
        def _mat(x, hh):
            x = jnp.asarray(x, jnp.float32)
            if x.shape[-2:] != (b, hh):
                raise ValueError(f"fa_fwd_kvcache: descale 尾两维须为 (b,h)={(b, hh)}")
            return jnp.array(x.reshape(b, hh))
        qd = _mat(q_descale, h)
        kd = _mat(k_descale, hk)
        vd = _mat(v_descale, hk)
        if hk != h:
            kd = jnp.array(jnp.broadcast_to(kd[:, :, None], (b, hk, h // hk))
                           .reshape(b, h))
            vd = jnp.array(jnp.broadcast_to(vd[:, :, None], (b, hk, h // hk))
                           .reshape(b, h))
    else:
        qd = kd = vd = dummy
    if _USE_FFI:
        o = _fa_pa_ffi(q, kcache, vcache, block_table, seqlens_k, scores_accum,
                       o_accum, dummy, qd, kd, vd, scale=float(softmax_scale),
                       causal=bool(causal), layout=1,
                       dtype=int(((1 if out_dtype == jnp.bfloat16 else 0)
                                  if (is_fp8 or _int8)
                                  else (1 if q.dtype == jnp.bfloat16 else 0))),
                       wl=wl, wr=wr, sk=sk, num_splits=ns, partition_size=ps,
                       mtp=sq, ngroups=h // hk,
                       bt_stride=int(block_table.shape[1]),
                       total_q=(h if is_fp8 else 0), total_k=(1 if is_fp8 else 0),
                       mtp_sq=0, is_fp8=(2 if _int8 else int(is_fp8)))
    else:
        o = _pa_p.bind(
            q, kcache, vcache, block_table, seqlens_k, scores_accum, o_accum,
            dummy, qd, kd, vd, b=b, h=h, hk=hk, sq=sq, sk=sk, d=d, dv=dv,
            scale=float(softmax_scale), causal=bool(causal), layout=1,
            dtype=((1 if out_dtype == jnp.bfloat16 else 0)
                   if (is_fp8 or _int8) else (1 if q.dtype == jnp.bfloat16 else 0)),
            wl=wl, wr=wr, page=page,
            num_splits=ns, partition_size=ps, mtp=sq, ngroups=h // hk,
            bt_stride=int(block_table.shape[1]), is_fp8=(2 if _int8 else int(is_fp8)))
    if debug_accum:
        return o, scores_accum, o_accum
    return o

# ===================== varlen 反向 =====================
_fa_bwd_varlen_p = core.Primitive("fa_bwd_v0")
_fa_bwd_varlen_p.multiple_results = True


def _fa_bwd_varlen_abstract(q, k, v, o, do, lse, rng, dbg, sem, alibi, saux,
                           cuq, cuk, *, scale, causal, wl, wr, msq, msk,
                           vbwd_mode=0, lse_unpadded=1):
    if q.ndim != 3:
        raise ValueError("fa_varlen_bwd: q 须为 3-D (total_q,h,d)")
    total_q, h, d = q.shape
    total_k, hk, dv = k.shape[0], k.shape[1], v.shape[2]
    b = cuq.shape[0] - 1
    dq = q.update(shape=(total_q, h, d), weak_type=False)
    dk = core.ShapedArray((total_k, h, d), q.dtype)
    dv_aval = core.ShapedArray((total_k, h, dv), q.dtype)
    dsm = core.ShapedArray((b, h, _rm(msq, 128)), jnp.float32)
    return [dq, dk, dv_aval, dsm]


_fa_bwd_varlen_p.def_abstract_eval(_fa_bwd_varlen_abstract)


def _fa_bwd_varlen_impl(q, k, v, o, do, lse, rng, dbg, sem, alibi, saux, cuq,
                        cuk, scale, causal, wl, wr, msq, msk, vbwd_mode=0,
                        lse_unpadded=1):
    if _USE_FFI:
        tq, h, d = q.shape
        tk, dv = k.shape[0], v.shape[2]
        specs = (jax.ShapeDtypeStruct(q.shape, q.dtype),
                 jax.ShapeDtypeStruct((tk, h, d), q.dtype),
                 jax.ShapeDtypeStruct((tk, h, dv), q.dtype),
                 jax.ShapeDtypeStruct((cuq.shape[0] - 1, h, _rm(msq, 128)), jnp.float32))
        return jax.ffi.ffi_call(BWD_FFI_TARGET, specs, vmap_method="sequential")(
            q, k, v, o, do, lse, rng, dbg, sem, alibi, saux, cuq, cuk,
            **_ffi_attrs(scale=np.float32(scale), causal=np.int32(causal),
                         wl=np.int32(wl), wr=np.int32(wr), has_varlen=np.int32(1),
                         msq=np.int32(msq), msk=np.int32(msk),
                         vbwd_mode=np.int32(vbwd_mode), lse_unpadded=np.int32(lse_unpadded),
                         dtype=np.int32(1 if q.dtype == jnp.bfloat16 else 0)))
    return _fa_bwd_varlen_p.bind(q, k, v, o, do, lse, rng, dbg, sem, alibi, saux,
                                 cuq, cuk, scale=scale, causal=causal, wl=wl,
                                 wr=wr, msq=msq, msk=msk, vbwd_mode=vbwd_mode,
                                 lse_unpadded=lse_unpadded)


_fa_bwd_varlen_p.def_impl(
    functools.partial(dispatch.apply_primitive, _fa_bwd_varlen_p))


def _fa_bwd_varlen_lower(ctx, q, k, v, o, do, lse, rng, dbg, sem, alibi, saux,
                         cuq, cuk, *, scale, causal, wl, wr, msq, msk,
                         vbwd_mode=0, lse_unpadded=1):
    q_aval, k_aval, v_aval = ctx.avals_in[0], ctx.avals_in[1], ctx.avals_in[2]
    total_q, h, d = q_aval.shape
    total_k, hk, dv = k_aval.shape[0], k_aval.shape[1], v_aval.shape[2]
    b = ctx.avals_in[11].shape[0] - 1
    dtype = 1 if q_aval.dtype == jnp.bfloat16 else 0
    opaque = _pack_opaque(b, h, hk, int(msq), int(msk), d, dv, scale, causal, 1,
                          dtype, wl, wr, 0.0, 0, 0, 0, 0, 1, total_q, total_k,
                          vbwd_mode, lse_unpadded, 0.0)

    def layouts(a):
        return mlir.dense_int_array(tuple(range(a.ndim - 1, -1, -1)))

    out = mlir.custom_call(
        BWD_TARGET,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[q, k, v, o, do, lse, rng, dbg, sem, alibi, saux, cuq, cuk],
        backend_config=opaque,
        operand_layouts=[layouts(a) for a in ctx.avals_in],
        result_layouts=[layouts(a) for a in ctx.avals_out],
    )
    return list(out.results)


mlir.register_lowering(_fa_bwd_varlen_p, _fa_bwd_varlen_lower, platform="rocm")
batching.primitive_batchers[_fa_bwd_varlen_p] = functools.partial(
    _sequential_batch, _fa_bwd_varlen_p)


def _seq_ranges(cu, seq_lens, name):
    """每条序列的 [start, stop) 区间 (host 侧)。dense_loop 降级需要具体长度。"""
    if seq_lens is not None:
        try:
            ln = [int(x) for x in np.asarray(seq_lens, np.int64).reshape(-1)]
        except Exception:  # noqa: BLE001
            raise ValueError(
                "dense_loop: seq_lens_q/seq_lens_k 必须是 host 侧值 (Python 列表/"
                "元组或 numpy 数组), 不能是 jit 的 traced 参数 —— 请把长度列表作为"
                "常量传入 (例如闭包捕获的 list/tuple)。") from None
    else:
        try:
            arr = np.asarray(jax.core.concrete_or_error(np.asarray, cu, name))
        except Exception:  # noqa: BLE001
            raise ValueError(
                f"{name}: dense_loop 降级需要 host 侧序列长度, 但该值在 jit 内是 "
                f"traced 的。请在调用处显式传 seq_lens_q=/seq_lens_k= (Python "
                f"列表或 numpy 数组)。") from None
        ln = np.diff(np.asarray(arr, np.int64).reshape(-1)).tolist()
    out, s = [], 0
    for x in ln:
        out.append((s, s + int(x)))
        s += int(x)
    return out


def _varlen_bwd_dense_loop(q, k, v, o, do, lse, cuq, cuk, *, causal,
                           softmax_scale, wl, wr, seq_lens_q, seq_lens_k,
                           deterministic=0):
    """降级实现: 逐序列调用稠密反向再拼回。

    用途: (1) deterministic=True 时保证 dq 累加顺序固定 (跨运行逐位可复现);
    (2) 内核路径的等价对照/回归。**无 padding 浪费**, 代价是 b 次稠密发射
    (发射开销随 b 线性增长), 每条序列独立分块, 长序列的并行度与算子库 varlen
    内核一致。
    仅支持自注意力 (cu_seqlens_q == cu_seqlens_k) 与 unpadded lse (fa_fwd_varlen 的输出)。
    """
    _register()
    c = _contract()
    rq = _seq_ranges(cuq, seq_lens_q, "cu_seqlens_q")
    rk = _seq_ranges(cuk, seq_lens_k, "cu_seqlens_k")
    if len(rq) != len(rk):
        raise ValueError(f"cu_seqlens_q/k 的序列数不一致: {len(rq)} vs {len(rk)}")
    if rq != rk:
        raise ValueError(
            "dense_loop 降级仅支持自注意力 (cu_seqlens_q == cu_seqlens_k); "
            "交叉注意力请用 mode='kernel' (默认)。")
    total_q, h, d = q.shape
    total_k, hk, dv = k.shape[0], k.shape[1], v.shape[2]
    lens_q = tuple(q1 - q0 for q0, q1 in rq)
    lens_k = tuple(k1 - k0 for k0, k1 in rk)
    if len(rq) != cuq.shape[0] - 1:
        raise ValueError("dense_loop host lengths must match the cumulative-length batch")
    if (any(x <= 0 for x in lens_q + lens_k)
            or sum(lens_q) > total_q or sum(lens_k) > total_k):
        raise ValueError("dense_loop lengths must be positive and fit packed capacities")
    # 切片起点使用校验后的累计长度，防止静态长度与设备长度的一致性检查被消除。
    cuq = _varlen_checked_static_cu(cuq, lens_q, total_q, max(lens_q))
    cuk = _varlen_checked_static_cu(cuk, lens_k, total_k, max(lens_k))
    g = h // hk
    scale = float(softmax_scale if softmax_scale is not None else d ** -0.5)
    dq_parts, dk_parts, dv_parts = [], [], []
    for i, (sq_i, sk_i) in enumerate(zip(lens_q, lens_k)):
        def rows(x, cu, count):
            return jax.lax.dynamic_slice_in_dim(x, cu[i], count, axis=0)[None]
        dqi, dki_e, dvi_e, _ = _fa_bwd_cp(
            rows(q, cuq, sq_i), rows(k, cuk, sk_i), rows(v, cuk, sk_i),
            rows(o, cuq, sq_i), rows(do, cuq, sq_i),
            jax.lax.dynamic_slice_in_dim(lse, cuq[i], sq_i, axis=1)[None],
            c["rng"], c["dbg"], c["sem"], _placeholder(), _placeholder(),
            scale, bool(causal), 1, int(wl), int(wr), 0.0, 0, 0, 0, 0, 0.0,
            deterministic=int(deterministic))
        dq_parts.append(dqi.reshape(sq_i, h, d))
        if h != hk:
            dk_parts.append(
                dki_e.reshape(sk_i, hk, g, d).astype(jnp.float32).sum(2).astype(k.dtype))
            dv_parts.append(
                dvi_e.reshape(sk_i, hk, g, dv).astype(jnp.float32).sum(2).astype(v.dtype))
        else:
            dk_parts.append(dki_e.reshape(sk_i, hk, d))
            dv_parts.append(dvi_e.reshape(sk_i, hk, dv))
    dq_parts.append(jnp.zeros((total_q - sum(lens_q), h, d), q.dtype))
    dk_parts.append(jnp.zeros((total_k - sum(lens_k), hk, d), k.dtype))
    dv_parts.append(jnp.zeros((total_k - sum(lens_k), hk, dv), v.dtype))
    return (jnp.concatenate(dq_parts, 0), jnp.concatenate(dk_parts, 0),
            jnp.concatenate(dv_parts, 0))


_VARLEN_BWD_MODE = os.environ.get("FA_VARLEN_BWD", "kernel")


def fa_varlen_bwd(q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                  max_seqlen_k, *, causal: bool = False, softmax_scale=None,
                  window_size=(-1, -1), vbwd_mode: int = 0,
                  lse_unpadded: int = 1, mode: str = None, seq_lens_q=None,
                  seq_lens_k=None, deterministic: bool = False):
    """打包变长注意力反向，输入及累计长度约束同 fa_fwd_varlen。

    o/lse 应来自对应前向，do 与 o 同形状、同类型；
    lse 为 float32 (h,capacity_q)。返回与 q/k/v 同形状的梯度。
    mode="kernel" 使用变长反向内核，默认模式可由 FA_VARLEN_BWD 设置。
    mode="dense_loop" 逐序列调用稠密反向，仅支持逐序列 Q/K 等长；
    在 jit 中无法取得具体累计长度时，须提供静态 seq_lens_q/k。
    deterministic=True 将内核模式切换为 dense_loop。
    仅支持 vbwd_mode=0、lse_unpadded=1。
    """
    max_q, max_k, scale, window = _varlen_metadata(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale, window_size)
    expected_o = (q.shape[0], q.shape[1], v.shape[2])
    if o.shape != expected_o or do.shape != expected_o or o.dtype != q.dtype or do.dtype != q.dtype:
        raise ValueError("fa_varlen_bwd: o/do must match forward output shape and dtype")
    if lse_unpadded != 1 or vbwd_mode != 0:
        raise ValueError("fa_varlen_bwd requires packed vbwd_mode=0 and lse_unpadded=1")
    if lse.dtype != jnp.float32 or lse.shape != (q.shape[1], q.shape[0]):
        raise ValueError("fa_varlen_bwd: lse must be the float32 forward residual")
    mode = _VARLEN_BWD_MODE if mode is None else mode
    if deterministic and mode == "kernel":
        mode = "dense_loop"
    if mode == "dense_loop":
        if lse_unpadded != 1:
            raise ValueError("dense_loop 仅支持 unpadded lse (fa_fwd_varlen 的输出)")
        return _varlen_bwd_dense_loop(
            q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_k, causal=causal,
            softmax_scale=softmax_scale, wl=int(window_size[0]),
            wr=int(window_size[1]), seq_lens_q=seq_lens_q,
            seq_lens_k=seq_lens_k, deterministic=int(bool(deterministic)))
    if mode != "kernel":
        raise ValueError(f"fa_varlen_bwd: 未知 mode={mode!r} (kernel / dense_loop)")
    cuq, cuk = _validated_lengths(
        cu_seqlens_q, cu_seqlens_k, q_capacity=q.shape[0], k_capacity=k.shape[0],
        max_q=max_q, max_k=max_k, require_equal=bool(causal) or window != (-1, -1))
    c = _contract()
    dq, dk_exp, dv_exp, _dsm = _fa_bwd_varlen_impl(
        q, k, v, o, do, lse, c["rng"], c["dbg"], c["sem"], _placeholder(),
        _placeholder(), cuq, cuk, scale=scale, causal=bool(causal),
        wl=window[0], wr=window[1], msq=max_q, msk=max_k,
        vbwd_mode=int(vbwd_mode), lse_unpadded=int(lse_unpadded))
    h, hk = q.shape[1], k.shape[1]
    if h != hk:
        g = h // hk
        dk = dk_exp.reshape(k.shape[0], hk, g, q.shape[2]).astype(jnp.float32).sum(2).astype(k.dtype)
        dv = dv_exp.reshape(k.shape[0], hk, g, v.shape[2]).astype(jnp.float32).sum(2).astype(v.dtype)
    else:
        dk, dv = dk_exp, dv_exp
    return dq, dk, dv

# ===================== attn_mask / padding_mask =====================


_ATTN_LENGTH_REGISTERED = False
_ATTN_LENGTH_TARGET = "fa_attn_length_ffi"


def _register_attn_length():
    global _ATTN_LENGTH_REGISTERED
    if _ATTN_LENGTH_REGISTERED:
        return
    _register()
    targets = _PLUGIN_MOD.registrations().get("ROCM", ())
    if not any(name == _ATTN_LENGTH_TARGET and int(version) == 1
               for name, _capsule, version in targets):
        raise RuntimeError("fa_fwd_attn_length: 请更新配套 fa_attention 插件")
    _ATTN_LENGTH_REGISTERED = True


def _attn_length_call(q, k, v, valid_k, scale, layout):
    _register_attn_length()
    b, h, _, sq, _, _, _ = _shapes(q, k, v, layout)
    results = (jax.ShapeDtypeStruct(q.shape, q.dtype),
               jax.ShapeDtypeStruct((b, h, sq), jnp.float32),
               jax.ShapeDtypeStruct((72,), jnp.int32))
    length = jnp.asarray([valid_k], dtype=jnp.int32)
    fn = jax.ffi.ffi_call(
        _ATTN_LENGTH_TARGET, results, vmap_method="sequential",
        input_layouts=[None] * 4, output_layouts=[None] * 3)
    out, lse, _scratch = fn(q, k, v, length, scale=np.float32(scale),
                           layout=np.int32(layout), valid_k=np.int32(valid_k))
    return out, lse


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _attn_length_static(q, k, v, valid_k, scale, layout):
    return _attn_length_call(q, k, v, valid_k, scale, layout)


def _attn_length_static_fwd(q, k, v, valid_k, scale, layout):
    o, lse = _attn_length_call(q, k, v, valid_k, scale, layout)
    return (o, lse), (q, k, v, o, lse)


def _attn_length_static_bwd(valid_k, scale, layout, residual, cotangents):
    q, k, v, o, lse = residual
    do, _ = cotangents
    if layout == 0:
        q, k, v, o, do = (x.transpose(0, 2, 1, 3) for x in (q, k, v, o, do))
    b, sq, h, d = q.shape
    sk, hk = k.shape[1:3]
    dv = v.shape[-1]
    cuq = jnp.arange(b + 1, dtype=jnp.int32) * sq
    cuk = jnp.arange(b + 1, dtype=jnp.int32) * valid_k
    dq, dk, dv_grad = fa_varlen_bwd(
        q.reshape(b * sq, h, d), k[:, :valid_k].reshape(b * valid_k, hk, d),
        v[:, :valid_k].reshape(b * valid_k, hk, dv),
        o.reshape(b * sq, h, dv), do.reshape(b * sq, h, dv),
        lse.transpose(1, 0, 2).reshape(h, b * sq), cuq, cuk, sq, valid_k,
        softmax_scale=scale, mode="kernel")
    dq = dq.reshape(b, sq, h, d)
    dk = jnp.pad(dk.reshape(b, valid_k, hk, d), ((0, 0), (0, sk - valid_k), (0, 0), (0, 0)))
    dv_grad = jnp.pad(dv_grad.reshape(b, valid_k, hk, dv), ((0, 0), (0, sk - valid_k), (0, 0), (0, 0)))
    if layout == 0:
        dq, dk, dv_grad = (x.transpose(0, 2, 1, 3) for x in (dq, dk, dv_grad))
    return dq, dk, dv_grad


_attn_length_static.defvjp(_attn_length_static_fwd, _attn_length_static_bwd)


def fa_fwd_attn_length(q, k, v, valid_k, *, softmax_scale=None, layout=1):
    """每个查询仅关注对应样本的有效 KV 前缀，返回 (o,lse)。

    q/k/v 同为 fp16/bf16，头维度均为 128；支持 layout=0/1 和 MHA/GQA/MQA。
    valid_k 为静态整数、动态 int32 标量或 int32[b]，满足 1<=valid_k<=sk。
    o 与 q 同形状，lse 为停止梯度的 float32 (b,h,sq)。
    支持 q/k/v 一阶反向，不支持因果、窗口或矩阵掩码。
    静态整数前向使用专用内核，动态长度和反向使用变长路径；
    动态长度存在校验同步和打包开销。
    """
    import operator
    if layout not in (0, 1):
        raise ValueError("fa_fwd_attn_length: layout 必须为 0 或 1")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("fa_fwd_attn_length: q/k/v 必须是 4-D")
    if q.dtype not in (jnp.float16, jnp.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("fa_fwd_attn_length: q/k/v 必须同为 fp16 或 bf16")
    b, h, hk, sq, sk, d, dv = _shapes(q, k, v, layout)
    if d != 128 or k.shape[-1] != 128 or dv != 128:
        raise ValueError("fa_fwd_attn_length: 仅支持 d=dv=128")
    if min(b, h, hk, sq, sk) <= 0 or h % hk != 0 or k.shape != v.shape or k.shape[0] != b:
        raise ValueError("fa_fwd_attn_length: batch/KV shape/head 数不匹配或存在空维度")
    if any(x.size >= 2 ** 31 for x in (q, k, v)):
        raise ValueError("fa_fwd_attn_length: q/k/v 元素数必须小于 2^31")
    scale = d ** -0.5 if softmax_scale is None else float(softmax_scale)
    if not np.isfinite(scale) or not np.isfinite(np.float32(scale)) or scale <= 0:
        raise ValueError("fa_fwd_attn_length: softmax_scale 必须是有限正数")
    if isinstance(valid_k, (bool, np.bool_)):
        raise TypeError("fa_fwd_attn_length: valid_k 必须是整数长度，不是 bool")
    if isinstance(valid_k, (int, np.integer)):
        valid_k = operator.index(valid_k)
        if not 1 <= valid_k <= sk:
            raise ValueError(f"fa_fwd_attn_length: 需要 1 <= valid_k <= {sk}")
        out, lse = _attn_length_static(q, k, v, valid_k, scale, int(layout))
        return out, jax.lax.stop_gradient(lse)
    length = _length_array(valid_k, "fa_fwd_attn_length: valid_k")
    if length.ndim == 0:
        length = jnp.broadcast_to(length, (b,))
    elif length.shape != (b,):
        raise ValueError("fa_fwd_attn_length: valid_k 必须是 int32 标量或 int32[B]")
    return fa_fwd_padded(q, k, v, jnp.full((b,), sq, jnp.int32), length,
                         softmax_scale=scale, layout=layout)


def fa_fwd_padding_length(q, k, v, lengths, *, softmax_scale=None, layout=1):
    """每个样本的 Q/K 共用有效长度，输入布局和返回值同 fa_fwd_padded。

    lengths 为 int32[b]，有效长度为正，Q/K 物理序列长度须相同。
    支持 jit、顺序 vmap 和 q/k/v 一阶反向。
    """
    if layout not in (0, 1) or any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("fa_fwd_padding_length: layout 必须为 0/1，q/k/v 须为 4-D")
    si = 1 if layout == 1 else 2
    if q.shape[si] != k.shape[si]:
        raise ValueError("fa_fwd_padding_length: Q/K 物理长度必须相同")
    return fa_fwd_padded(q, k, v, lengths, lengths,
                         softmax_scale=softmax_scale, layout=layout)


def fa_fwd_attn_mask(q, k, v, attn_mask, **kw):
    """任意矩阵注意力掩码尚未接入，调用即抛出 NotImplementedError。"""
    raise NotImplementedError(
        "当前公开 full attn_mask API 尚未接通；09/21 HIPC 入口只接受 int32[1] "
        "有效 K 长度，不接受 (B,H,Q,K) 矩阵 mask。")


def fa_fwd_padding_mask(q, k, v, padding_mask, **kw):
    """逐元素填充掩码尚未接入，调用即抛出 NotImplementedError。"""
    raise NotImplementedError(
        "当前公开 padding_mask API 尚未接通；09/21 HIPC 入口只接受 int32[B] "
        "每 batch 有效长度，不接受 (B,S) 0/1 token mask。")

# ===================== prefix-prefill 前向 =====================
_prefix_p = core.Primitive("fa_prefix_v0")
_prefix_p.multiple_results = True


def _prefix_abstract(q, kc, vc, bt, seqused, cuq, cuk, sem, *, b, h, hk, sk, d,
                     dv, scale, causal, dtype, wl, wr, page, bt_stride,
                     total_q, total_k, msq):
    if q.dtype not in (jnp.float16, jnp.bfloat16):
        raise TypeError(f"fa_prefix_prefill 仅支持 fp16/bf16, 得到 {q.dtype}")
    o = q.update(shape=(total_q, h, dv), weak_type=False)
    lse = core.ShapedArray((h, total_q), jnp.float32)
    return [o, lse]


_prefix_p.def_abstract_eval(_prefix_abstract)


def _fa_prefix_ffi(q, kc, vc, bt, seqused, cuq, cuk, sem, *, scale, causal, dtype,
                   wl, wr, sk, msq, total_k):
    o_spec = jax.ShapeDtypeStruct((q.shape[0], q.shape[1], vc.shape[3]), q.dtype)
    lse_spec = jax.ShapeDtypeStruct((q.shape[1], q.shape[0]), jnp.float32)
    fn = jax.ffi.ffi_call(PREFIX_FFI_TARGET, (o_spec, lse_spec),
                          vmap_method="sequential")
    return fn(q, kc, vc, bt, seqused, cuq, cuk, sem,
              scale=np.float32(scale), causal=np.int32(int(causal)),
              dtype=np.int32(int(dtype)), wl=np.int32(int(wl)),
              wr=np.int32(int(wr)), sk=np.int32(int(sk)), msq=np.int32(int(msq)),
              total_k=np.int32(int(total_k)))


def _prefix_impl(q, kc, vc, bt, seqused, cuq, cuk, sem, *, b, h, hk, sk, d, dv,
                 scale, causal, dtype, wl, wr, page, bt_stride, total_q,
                 total_k, msq):
    if _USE_FFI:
        return _fa_prefix_ffi(q, kc, vc, bt, seqused, cuq, cuk, sem, scale=scale,
                              causal=causal, dtype=dtype, wl=wl, wr=wr, sk=sk,
                              msq=msq, total_k=total_k)
    return dispatch.apply_primitive(
        _prefix_p, q, kc, vc, bt, seqused, cuq, cuk, sem, b=b, h=h, hk=hk,
        sk=sk, d=d, dv=dv, scale=scale, causal=causal, dtype=dtype, wl=wl,
        wr=wr, page=page, bt_stride=bt_stride, total_q=total_q, total_k=total_k,
        msq=msq)


_prefix_p.def_impl(functools.partial(dispatch.apply_primitive, _prefix_p))


def _prefix_lower(ctx, q, kc, vc, bt, seqused, cuq, cuk, sem, *, b, h, hk, sk,
                  d, dv, scale, causal, dtype, wl, wr, page, bt_stride,
                  total_q, total_k, msq):
    opaque = _pack_pa_opaque(b, h, hk, msq, sk, d, dv, scale, 1, dtype, causal,
                             wl, wr, page, 0, 0, 0, 0, bt_stride,
                             total_q, total_k, msq)

    def layouts(a):
        return mlir.dense_int_array(tuple(range(a.ndim - 1, -1, -1)))

    out = mlir.custom_call(
        PREFIX_TARGET,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[q, kc, vc, bt, seqused, cuq, cuk, sem],
        backend_config=opaque,
        operand_layouts=[layouts(a) for a in ctx.avals_in],
        result_layouts=[layouts(a) for a in ctx.avals_out],
    )
    return list(out.results)


mlir.register_lowering(_prefix_p, _prefix_lower, platform="rocm")
batching.primitive_batchers[_prefix_p] = functools.partial(_sequential_batch, _prefix_p)


def fa_prefix_prefill(q, kcache, vcache, block_table, seqused_k, cu_seqlens_q,
                      max_seqlen_q, max_seqlen_k, *, causal: bool = False,
                      softmax_scale=None, window_size=(-1, -1)):
    """变长查询对分页 KV 缓存的注意力，仅支持前向。

    q 为 fp16/bf16 (total_q,h,128)，缓存为 (num_blocks,page,hk,d/dv)。
    block_table 为 int32[b,max_blocks]，seqused_k 为 int32[b] 实际 KV 长度，
    cu_seqlens_q 为 int32[b+1] 累计查询长度，最大序列长度为静态参数。
    返回 o (total_q,h,dv) 和 float32 LSE (h,total_q)。
    """
    _register()
    if q.ndim != 3:
        raise ValueError("fa_prefix_prefill: q 须为 3-D (total_q,h,d)")
    tq, h, d = q.shape
    nb, page, hk, _ = kcache.shape
    dv = vcache.shape[3]
    b = cu_seqlens_q.shape[0] - 1
    if block_table.shape[0] != b or seqused_k.shape != (b,):
        raise ValueError("fa_prefix_prefill: block_table 须 (b,max_blocks), "
                         "seqused_k 须 (b,)")
    if d != 128:
        raise ValueError("fa_prefix_prefill: 本构建实测仅验证 d=128 "
                         "(源码允许 128/192/256)")
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    c = _contract()
    return _prefix_p.bind(
        q, kcache, vcache, block_table, seqused_k, cu_seqlens_q,
        _cu_placeholder(), c["sem"],
        b=b, h=h, hk=hk, sk=int(max_seqlen_k), d=d, dv=dv,
        scale=float(softmax_scale), causal=bool(causal),
        dtype=1 if q.dtype == jnp.bfloat16 else 0,
        wl=int(window_size[0]), wr=int(window_size[1]), page=page,
        bt_stride=int(block_table.shape[1]), total_q=tq, total_k=0,
        msq=int(max_seqlen_q))

# MLA-prefix retains its opaque layout independently of the retired decode API.
_MLA_FMT = "<5if6i"
assert struct.calcsize(_MLA_FMT) == 48


def fa_mla(q, kcache, block_table, cache_seqlens, max_seqlen_k, *,
           causal: bool = False, softmax_scale=None):
    """MLA 解码不可用，调用即抛出 NotImplementedError。"""
    raise NotImplementedError(
        "fa_mla: MLA decode 不支持 —— 算子组已停止维护，后续将剔除 "
        "run_fwd_flashmla；本接入不再提供该功能。")

# ===================== FP8 (e4m3) 前向 =====================
_fp8_p = core.Primitive("fa_fwd_fp8_v0")
_fp8_p.multiple_results = True


def _fp8_abstract(q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, dq, dk, dv, *,
                  scale, causal, layout, b, h, hk, sq, sk, d, dv_dim, wl, wr,
                  out_bf16):
    if q.dtype != jnp.float8_e4m3fn:
        raise TypeError(f"fa_fwd_fp8: q 须为 float8_e4m3fn, 得到 {q.dtype}")
    od = jnp.bfloat16 if out_bf16 else jnp.float16
    o_shape = (b, sq, h, dv_dim) if layout == 1 else (b, h, sq, dv_dim)
    return [core.ShapedArray(o_shape, od),
            core.ShapedArray((b, h, sq), jnp.float32)]


_fp8_p.def_abstract_eval(_fp8_abstract)


def _fp8_impl(q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, dq, dk, dv, *,
              scale, causal, layout, b, h, hk, sq, sk, d, dv_dim, wl, wr,
              out_bf16):
    return dispatch.apply_primitive(
        _fp8_p, q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, dq, dk, dv,
        scale=scale, causal=causal, layout=layout, b=b, h=h, hk=hk, sq=sq, sk=sk,
        d=d, dv_dim=dv_dim, wl=wl, wr=wr, out_bf16=out_bf16)


_fp8_p.def_impl(functools.partial(dispatch.apply_primitive, _fp8_p))


def _fp8_lower(ctx, q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, dq, dk, dv,
               *, scale, causal, layout, b, h, hk, sq, sk, d, dv_dim, wl, wr,
               out_bf16):
    opaque = _pack_opaque(b, h, hk, sq, sk, d, dv_dim, scale, causal, layout,
                          1 if out_bf16 else 0, wl, wr, 0.0, 0, 0, 0, 0, 0,
                          total_q=h, total_k=1, vbwd_mode=0, lse_unpadded=0,
                          dropout_p=0.0, is_fp8=1)

    def layouts(a):
        return mlir.dense_int_array(tuple(range(a.ndim - 1, -1, -1)))

    out = mlir.custom_call(
        FP8_TARGET,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[q, k, v, rng, dbg, sem, alibi, saux, cuq, cuk, dq, dk, dv],
        backend_config=opaque,
        operand_layouts=[layouts(a) for a in ctx.avals_in],
        result_layouts=[layouts(a) for a in ctx.avals_out],
    )
    return list(out.results)


mlir.register_lowering(_fp8_p, _fp8_lower, platform="rocm")


def fa_fwd_fp8(q, k, v, q_descale, k_descale, v_descale, *, causal: bool = False,
               softmax_scale=None, layout: int = 1, window_size=(-1, -1),
               out_dtype=jnp.bfloat16):
    """FP8 注意力前向，输入布局同 fa_fwd。

    q/k/v 为 float8_e4m3fn，d=dv 且属于 {128,192,256}。
    q_descale 为 float32 (b,h)，k_descale/v_descale 为 float32 (b,hk)。
    返回同布局的 o 和 float32 LSE (b,h,sq)；
    out_dtype 为 bfloat16 或 float16，默认为 bfloat16。
    仅支持前向；硬件须通过支持检查，否则抛出 ValueError。
    """
    _register()
    if q.dtype != jnp.float8_e4m3fn:
        raise ValueError("fa_fwd_fp8: q/k/v 须为 jnp.float8_e4m3fn")
    if k.dtype != jnp.float8_e4m3fn or v.dtype != jnp.float8_e4m3fn:
        raise ValueError("fa_fwd_fp8: k/v 须同为 float8_e4m3fn")
    b, h, hk, sq, sk, d, dv_dim = _shapes(
        jax.core.ShapedArray(q.shape, q.dtype),
        jax.core.ShapedArray(k.shape, k.dtype),
        jax.core.ShapedArray(v.shape, v.dtype), layout)
    if d != dv_dim or d not in (128, 192, 256):
        raise ValueError(f"fa_fwd_fp8: 本构建仅支持 d==dv ∈ {{128,192,256}}, "
                         f"得到 d={d}, dv={dv_dim}")
    if q_descale.dtype != jnp.float32 or q_descale.ndim < 2:
        raise ValueError("fa_fwd_fp8: descale 须为 float32 且 ≥2 维")
    def _mat(x, hh):
        x = jnp.asarray(x, jnp.float32)
        if x.shape[-2:] != (b, hh):
            raise ValueError(f"fa_fwd_fp8: descale 尾两维须为 (b,h)={(b, hh)}, "
                             f"得到 {x.shape}")
        return jnp.array(x.reshape(b, hh))
    qd = _mat(q_descale, h)
    kd = _mat(k_descale, hk)
    vd = _mat(v_descale, hk)
    if hk != h:   # k/v descale 广播到 (b,h), 保证三者 stride 一致
        kd = jnp.array(jnp.broadcast_to(kd[:, :, None], (b, hk, h // hk))
                       .reshape(b, h))
        vd = jnp.array(jnp.broadcast_to(vd[:, :, None], (b, hk, h // hk))
                       .reshape(b, h))
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    if not _fp8_arch_ok():
        raise ValueError(
            "fa_fwd_fp8: 本机 arch 不支持 FP8 内核 (需 gfx938+; 本部署卡为 gfx936) —— "
            "内核会打印 'fp8 is not supported in this arch!' 并输出 NaN。"
            "请在 gfx938 及以后的卡上使用, 或改用 fp16/bf16 路径。")
    c = _contract()
    o, lse = _fp8_p.bind(
        q, k, v, c["rng"], c["dbg"], c["sem"], _placeholder(), _placeholder(),
        _cu_placeholder(), _cu_placeholder(), qd, kd, vd,
        scale=float(softmax_scale), causal=bool(causal), layout=int(layout),
        b=b, h=h, hk=hk, sq=sq, sk=sk, d=d, dv_dim=dv_dim,
        wl=int(window_size[0]), wr=int(window_size[1]),
        out_bf16=(out_dtype == jnp.bfloat16))
    return o, lse

# ================ MLA prefix-prefill (chunked prefill) 前向 =================
_mla_pf_p = core.Primitive("fa_mla_prefix_v0")
_mla_pf_p.multiple_results = True


def _mla_pf_abstract(q, qv, kc, vc, pt, cs, cuq, cukn, sm, *, b, h, hk, tq, msq,
                     scale, causal, dtype, page, bt_stride, is_mtp):
    if q.dtype not in (jnp.float16, jnp.bfloat16):
        raise TypeError(f"fa_mla_prefix 仅支持 fp16/bf16, 得到 {q.dtype}")
    return [core.ShapedArray((tq, h, 512), q.dtype),
            core.ShapedArray((h, tq), jnp.float32)]


_mla_pf_p.def_abstract_eval(_mla_pf_abstract)


def _mla_pf_impl(q, qv, kc, vc, pt, cs, cuq, cukn, sm, *, b, h, hk, tq, msq,
                 scale, causal, dtype, page, bt_stride, is_mtp):
    return dispatch.apply_primitive(
        _mla_pf_p, q, qv, kc, vc, pt, cs, cuq, cukn, sm, b=b, h=h, hk=hk, tq=tq,
        msq=msq, scale=scale, causal=causal, dtype=dtype, page=page,
        bt_stride=bt_stride, is_mtp=is_mtp)


_mla_pf_p.def_impl(functools.partial(dispatch.apply_primitive, _mla_pf_p))


def _mla_pf_lower(ctx, q, qv, kc, vc, pt, cs, cuq, cukn, sm, *, b, h, hk, tq,
                  msq, scale, causal, dtype, page, bt_stride, is_mtp):
    opaque = struct.pack(_MLA_FMT, b, h, hk, tq, msq, float(scale), dtype, causal,
                         page, bt_stride, is_mtp, 0)

    def layouts(a):
        return mlir.dense_int_array(tuple(range(a.ndim - 1, -1, -1)))

    out = mlir.custom_call(
        MLA_PREFIX_TARGET,
        result_types=[mlir.aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[q, qv, kc, vc, pt, cs, cuq, cukn, sm],
        backend_config=opaque,
        operand_layouts=[layouts(a) for a in ctx.avals_in],
        result_layouts=[layouts(a) for a in ctx.avals_out],
    )
    return list(out.results)


mlir.register_lowering(_mla_pf_p, _mla_pf_lower, platform="rocm")


def fa_mla_prefix(q, qv, kcache, vcache, page_table, cache_seqlens,
                  cu_seqlens_q, cu_seqlens_k_new, max_seqlen_q, *,
                  causal: bool = True, softmax_scale=None, is_mtp: bool = False):
    """MLA 前缀预填充不可用，调用即抛出 NotImplementedError。"""
    raise NotImplementedError(
        "fa_mla_prefix: 本构建 (libflash_attention.so.1 09/21) 的 "
        "run_fwd_prefix_prefill_mla 已被算子组停止维护 (2026-09-24), 且实测内核输出无效 "
        "(causal=True 恒全 0, 见 README 附录 W); 请使用算子组后续的新接口。")

    _register()
    if q.ndim != 3 or q.shape[-1] != 576:
        raise ValueError("fa_mla_prefix: q 须为 (total_q,h,576)")
    tq, h, _ = q.shape
    hk = kcache.shape[2]
    if kcache.shape[1] != 128 or vcache.shape[1] != 128:
        raise ValueError("fa_mla_prefix: page_block_size 必须 128")
    if vcache.shape[3] != 512:
        raise ValueError("fa_mla_prefix: vcache 末维须 512")
    b = page_table.shape[0]
    if softmax_scale is None:
        softmax_scale = 576 ** -0.5
    if _arch_raw() and _arch_raw() < 930:
        raise ValueError(f"fa_mla_prefix: arch={_arch_raw()} 不支持 (需 gfx92a 或 >=936)")
    c = _contract()
    sm = jnp.zeros((3, h, tq), jnp.float32)
    if _USE_FFI:
        o_spec = jax.ShapeDtypeStruct((tq, h, 512), q.dtype)
        lse_spec = jax.ShapeDtypeStruct((h, tq), jnp.float32)
        fn = jax.ffi.ffi_call(MLA_PREFIX_FFI_TARGET, (o_spec, lse_spec),
                              vmap_method="sequential")
        return fn(q, qv, kcache, vcache, page_table, cache_seqlens, cu_seqlens_q,
                  cu_seqlens_k_new, sm, scale=np.float32(float(softmax_scale)),
                  causal=np.int32(int(causal)),
                  dtype=np.int32(1 if q.dtype == jnp.bfloat16 else 0),
                  msq=np.int32(int(max_seqlen_q)), is_mtp=np.int32(int(bool(is_mtp))))
    o, lse = _mla_pf_p.bind(
        q, qv, kcache, vcache, page_table, cache_seqlens, cu_seqlens_q,
        cu_seqlens_k_new, sm, b=b, h=h, hk=hk, tq=tq, msq=int(max_seqlen_q),
        scale=float(softmax_scale), causal=bool(causal),
        dtype=1 if q.dtype == jnp.bfloat16 else 0, page=int(kcache.shape[1]),
        bt_stride=int(page_table.shape[1]), is_mtp=int(bool(is_mtp)))
    return o, lse


# ================= standard padded attention / jax.nn adapter ===============
def _padded_attention_options(q, k, v, layout, softmax_scale, window_size):
    """Validate the subset supported without head-dimension padding."""
    import operator

    if layout not in (0, 1):
        raise ValueError("HIPC attention: layout must be 0 or 1")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("HIPC attention: q/k/v must be 4-D")
    if q.dtype not in (jnp.float16, jnp.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("HIPC attention: q/k/v must have the same fp16/bf16 dtype")
    b, h, hk, sq, sk, d, dv = _shapes(q, k, v, layout)
    if min(b, h, hk, sq, sk) <= 0 or h % hk:
        raise ValueError("HIPC attention: positive batch/sequence/head dimensions and H % Hk == 0 required")
    if k.shape != v.shape or k.shape[0] != b or k.shape[-1] != d:
        raise ValueError("HIPC attention: batch/KV shapes must match and d == dv is required")
    if d not in (32, 64, 96, 128, 160, 192, 224, 256):
        raise ValueError("HIPC attention: head dimension must be a multiple of 32 in [32, 256]")
    # int32 偏移检查须覆盖 GQA 展开梯度及补齐后的反向缓冲，不能只检查输入大小。
    counts = [x.size for x in (q, k, v)]
    counts += [b * _rm(sq, 128) * h * d, b * _rm(sk, 128) * h * dv,
               b * h * _rm(sq, 128)]
    if max(counts) >= 2 ** 31:
        raise ValueError("HIPC attention: input and backward buffer sizes must be below 2^31")
    if isinstance(softmax_scale, core.Tracer):
        raise ValueError("HIPC attention: softmax_scale must be a static finite positive scalar")
    scale = d ** -0.5 if softmax_scale is None else float(softmax_scale)
    with np.errstate(over="ignore", under="ignore"):
        scale32 = np.float32(scale)
    if not np.isfinite(scale32) or scale32 <= 0:
        # The current typed shim replaces nonpositive scales with its default.
        raise ValueError("HIPC attention: softmax_scale must be finite and positive in float32")
    if not isinstance(window_size, (tuple, list)) or len(window_size) != 2:
        raise ValueError("HIPC attention: window_size must be a pair of static integers")
    try:
        window = tuple(operator.index(x) for x in window_size)
    except TypeError as e:
        raise ValueError("HIPC attention: window_size must be a pair of static integers") from e
    if min(window) < -1 or max(window) >= 2 ** 31:
        raise ValueError("HIPC attention: window sizes must be int32 values >= -1")
    return b, h, hk, sq, sk, scale, window


def _pack_padded_attention(x, lengths, cu_lengths, active):
    """按序列稳定打包，保留编译期容量 b*s。

    无效散射位置直接丢弃，不得夹到有效行；
    空查询样本使用长度为一的全零占位 Q/K/V。
    """
    b, s, h, d = x.shape
    row = jnp.arange(s, dtype=jnp.int32)[None, :]
    valid = row < lengths[:, None]
    positions = cu_lengths[:-1, None] + row
    positions = jnp.where(valid, positions, b * s).reshape(-1)
    values = jnp.where((valid & active[:, None])[..., None, None], x, 0)
    packed = jnp.zeros((b * s, h, d), x.dtype)
    return packed.at[positions].set(values.reshape(b * s, h, d), mode="drop")


def fa_fwd_padded(q, k, v, query_seq_lengths, key_value_seq_lengths, *,
                  causal=False, softmax_scale=None, layout=1,
                  window_size=(-1, -1)):
    """带有效长度的填充注意力，支持 q/k/v 一阶反向。

    q/k/v 同为 fp16/bf16，采用 layout=1 的 BSHD 或 layout=0 的 BHSD；
    支持 MHA/GQA/MQA，头维度相等且为 [32,256] 内的 32 倍数。
    两组长度为独立 int32[b]，None 表示完整序列；
    查询长度可为零，KV 长度须为正，均不得超过物理容量。
    因果或窗口注意力要求非空查询样本的 Q/K 有效长度相等。
    返回同输入布局的 o 和停止梯度的 float32 LSE (b,h,sq)；
    填充输出行为零，对应 LSE 为 -inf。
    不支持物理空维度，缩放值须为静态且在 float32 中为有限正数。
    运行时长度校验存在主机复制和设备流同步开销。
    """
    q, k, v = (jnp.asarray(x) for x in (q, k, v))
    b, _, _, sq, sk, scale, window = _padded_attention_options(
        q, k, v, layout, softmax_scale, window_size)
    if isinstance(causal, core.Tracer):
        raise ValueError("HIPC attention: causal must be static")
    if layout == 0:
        q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
    try:
        q_lengths = (jnp.full((b,), sq, jnp.int32) if query_seq_lengths is None
                     else _length_array(query_seq_lengths, "query_seq_lengths"))
        k_lengths = (jnp.full((b,), sk, jnp.int32) if key_value_seq_lengths is None
                     else _length_array(key_value_seq_lengths, "key_value_seq_lengths"))
    except TypeError as e:
        # Preserve this API's existing ValueError for invalid length metadata.
        raise ValueError(f"HIPC attention: {e}") from e
    for name, lengths in (("query_seq_lengths", q_lengths),
                          ("key_value_seq_lengths", k_lengths)):
        if lengths.shape != (b,) or lengths.dtype != jnp.int32:
            raise ValueError(f"HIPC attention: {name} must be int32[{b}]")
    q_lengths, k_lengths = _validated_lengths(
        q_lengths, k_lengths, q_capacity=sq, k_capacity=sk,
        max_q=sq, max_k=sk, cumulative=False,
        require_equal=bool(causal) or window != (-1, -1))
    active = q_lengths > 0
    # Every kernel sequence is positive, including causal/local dummy rows.
    effective_q = jnp.where(active, q_lengths, 1)
    effective_k = jnp.where(active, k_lengths, 1)
    cuq = jnp.concatenate((jnp.zeros((1,), jnp.int32),
                           jnp.cumsum(effective_q, dtype=jnp.int32)))
    cuk = jnp.concatenate((jnp.zeros((1,), jnp.int32),
                           jnp.cumsum(effective_k, dtype=jnp.int32)))
    qp = _pack_padded_attention(q, effective_q, cuq, active)
    kp = _pack_padded_attention(k, effective_k, cuk, active)
    vp = _pack_padded_attention(v, effective_k, cuk, active)
    op, lsep = fa_fwd_varlen(
        qp, kp, vp, cuq, cuk, sq, sk, causal=bool(causal),
        softmax_scale=scale, window_size=window)
    rows = jnp.arange(sq, dtype=jnp.int32)[None, :]
    valid = rows < q_lengths[:, None]
    # 仅夹紧未使用的收集索引，再屏蔽填充输出及其梯度。
    # 打包容量不随有效总长度变化，LSE 头步长不能由 cuq[-1] 推断。
    positions = jnp.minimum(cuq[:-1, None] + rows, b * sq - 1)
    out = jnp.where(valid[..., None, None], op[positions], 0)
    lse = jnp.where(valid[:, None, :],
                    lsep[:, positions].transpose(1, 0, 2), -jnp.inf)
    if layout == 0:
        out = out.transpose(0, 2, 1, 3)
    return out, jax.lax.stop_gradient(lse)


def dot_product_attention(q, k, v, bias=None, mask=None,
                          query_seq_lengths=None, key_value_seq_lengths=None, *,
                          scale=None, is_causal=False, local_window_size=None,
                          return_residual=False):
    """标准 dot_product_attention 的 HIPC 后端适配层。

    输入约束同 fa_fwd_padded，使用 BSHD 布局，不支持 bias 或任意矩阵掩码。
    返回注意力输出；return_residual=True 时同时返回停止梯度的残差，
    残差为 BTN 布局、查询类型，填充位置使用与 xla 相同的负值哨兵，
    而非 fa_fwd_padded 的 float32 -inf。
    """
    if bias is not None or mask is not None:
        raise NotImplementedError("HIPC DPA does not support bias or arbitrary attention masks")
    window = (-1, -1) if local_window_size is None else local_window_size
    _, _, _, sq, sk, scale, window = _padded_attention_options(
        q, k, v, 1, scale, window)
    if local_window_size is not None and min(window) < 0:
        raise ValueError("HIPC DPA: local_window_size must contain nonnegative integers")
    if isinstance(is_causal, core.Tracer):
        raise ValueError("HIPC DPA: is_causal must be static")
    use_padding = query_seq_lengths is not None or key_value_seq_lengths is not None
    if use_padding:
        out, lse = fa_fwd_padded(
            q, k, v, query_seq_lengths, key_value_seq_lengths,
            causal=is_causal, softmax_scale=scale, window_size=window)
    else:
        if (is_causal or local_window_size is not None) and sq != sk:
            raise ValueError("HIPC DPA: causal/local attention requires equal Q/K lengths; masks are otherwise right-aligned")
        out, lse = fa_fwd_custom(
            q, k, v, causal=is_causal, softmax_scale=scale,
            layout=1, window_size=window)
    if not return_residual:
        return out
    residual = lse.transpose(0, 2, 1)
    if query_seq_lengths is not None:
        q_lengths = _length_array(query_seq_lengths, "query_seq_lengths")
        valid = jnp.arange(sq)[None, :] < q_lengths[:, None]
        sentinel = jnp.asarray(-0.7 * np.finfo(np.float32).max, jnp.float32)
        residual = jnp.where(valid[..., None], residual, sentinel)
    return out, jax.lax.stop_gradient(residual.astype(q.dtype))
