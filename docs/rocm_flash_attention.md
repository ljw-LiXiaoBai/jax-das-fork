# HIPC FlashAttention for the DTK JAX fork

This module integrates the current vendor `libflash_attention.so.1` into JAX
through the ROCm `fa_attention` plugin. The normal dense path uses typed XLA FFI;
several specialized paths retain legacy custom calls. The current candidate
registers eight typed FFI targets (including length validation) and six legacy
targets after removing MLA-decode registrations. The library is loaded at runtime and is not bundled in JAX or linked
into the plugin at build time.

The fork-specific APIs live under `jax._src.cudnn.fa_attention`. Standard
`jax.nn.dot_product_attention(..., implementation="hipc")` is now an explicit
opt-in for the subset documented below. The default `implementation=None`
route is unchanged. This opt-in does **not** use the old
`implementation="cutlass"` backend or promise compatibility with every standard
DPA option.

**Historical verification (2026-09-29, before MLA retirement):** the supported
subsets below had numerical and gradient coverage on gfx936, in typed FFI and
legacy modes. This is not a post-retirement full rerun. Dense batch/head GQA
partitioning has two-device coverage; this is not a full JAX-suite, wheel-build,
performance, or all-architecture qualification. FP8 numerical cases remain
unverified on gfx936. See the testing section for the exact acceptance scope.

## Installation and library selection

Install matching JAX, jaxlib and ROCm plugin builds that contain this change,
including the standard DPA dispatch update when using `implementation="hipc"`.
The plugin target is `//jaxlib/rocm:fa_attention`, also included in
`//jaxlib/rocm:rocm_gpu_support`. Use this fork's DTK build configuration; a
generic upstream ROCm wheel does not contain this integration.

Deploy the vendor library at:

```text
/opt/libflash_attention.so
```

Deployments that keep the vendor's `libflash_attention.so.1` filename must
either symlink it to the default path above or point `FA_LIB_PATH` at it
**before starting Python**. An alternate library can also be selected:

```bash
export FA_LIB_PATH=/path/to/libflash_attention.so.1
```

When `FA_LIB_PATH` is set, it takes precedence; there is no silent fallback to
another installed copy if that path is wrong. Library handles and symbols are
cached per process. Restart Python after changing the library or its path.
Ensure the matching DTK runtime is available on the dynamic loader search path.

The earlier gfx936 verification used the 09/21 vendor library and header:

```text
libflash_attention.so.1 MD5: 1d8277d0aa0b1b0c2156085073394c31
flash.h SHA256: 475d6d16e9d44bc24ed940bb92072b818e3b0f263ecaef34c160f82a36c74429
```

These identifiers describe that baseline, not acceptance of every expanded path.
The corresponding vendor header is `jaxlib/gpu/flash.h`.
`sizeof(Flash_fwd_params)==0x2a0` is a compile-time sanity check, **not a runtime
ABI version check**. Equal structure sizes, exported symbols or compilation
success do not prove library compatibility or enabled kernel features. A new
library needs matching headers and runtime numerical validation. Never replace
this header with the other CUTLASS backend's parameter definition.

## API overview and boundaries

Unless a particular API says otherwise, Q/K/V have matching fp16 or bf16 dtype.
Use the documented shapes and dimensions; accepted metadata is not evidence that
every parameter combination has completed numerical acceptance.

| API | Purpose and return value | Differentiation / limits |
|---|---|---|
| Standard DPA with `implementation="hipc"` | Dense or length-aware attention; `o`, optionally `(o, residual)` | Explicit subset below; q/k/v first-order reverse mode; residual is stop-gradient |
| `fa_fwd` | Dense forward; `(o, lse)` | Use `fa_fwd_custom` for training |
| `fa_fwd_custom` | Dense forward with custom VJP; `(o, lse)` | q/k/v first-order reverse mode; LSE is auxiliary, not a differentiable loss input |
| `fa_fwd_padded` | Independent per-batch Q/KV valid lengths; `(o, lse)` | q/k/v first-order reverse mode via packed varlen; padded output rows are zero |
| `fa_fwd_varlen` | Packed variable-length attention; `(o, lse)` | Custom VJP for q/k/v; stop-gradient LSE; sinks are forward-only |
| `fa_varlen_bwd` | Explicit packed q/k/v gradients | Kernel mode by default; deterministic self-attention uses dense-loop backward |
| `fa_fwd_attn_length` | Every query attends to a K/V prefix; `(o, lse)` | d=dv=128; static or runtime positive lengths; q/k/v first-order reverse mode |
| `fa_fwd_padding_length` | One shared Q/K length per batch item; `(o, lse)` | Equal physical Q/K sequence capacities; positive lengths; via padded varlen |
| `fa_fwd_kvcache` | Paged KV decoding; `o` | Forward only; no LSE return |
| `fa_prefix_prefill` | Packed Q against paged KV; `(o, lse)` | Forward only; wrapper currently requires d=128 |
| `fa_fwd_fp8` | FP8 dense forward; `(o, lse)` | Experimental/architecture-gated; no numerical acceptance on gfx936 |

First-order reverse mode does **not** imply JVP/forward-mode or higher-order
differentiation support. Batching rules do not make forward-only APIs
trainable. Multi-device support is limited to the candidate rules described in
"Batching and sharding", not arbitrary partitioning or distributed attention.

### Explicitly unavailable

- **softcap and ALiBi:** not supported; non-default requests raise `ValueError`.
  Retained parameter metadata is not an enabled feature, and no vendor build-macro
  state is inferred from the header or symbols.
- **INT8 PA, MLA decode and MLA-prefix:** unsupported; public calls raise
  `NotImplementedError`. The vendor has stopped maintaining MLA decode and will
  remove it; this retirement is not due to a numerical failure. Historical MLA
  passes do not establish current support. The public `fa_mla` signature remains
  only as a compatibility guard: every call immediately raises
  `NotImplementedError` **before plugin registration**. All decode-specific
  Python primitives, lowerings and planners, native handlers/getters, the
  `FaLib` `flashmla` symbol lookup, and typed/legacy plugin registrations are
  removed. The existing guarded MLA-prefix native target is unchanged;
  standard PA and prefix prefill are unaffected.
- **Additive bias and arbitrary matrix masks:** not supported by HIPC DPA.
  Supplying `bias` or `mask` raises `NotImplementedError`; no mathematical
  attention fallback is substituted.
- **Old standalone mask APIs:** `fa_fwd_attn_mask` and `fa_fwd_padding_mask`
  still reject calls. In particular, a `(B,S)` token/padding mask or
  `(B,H,Q,K)` matrix is not a length vector. Use the explicit length APIs below;
  they do not enable those old mask entry points.
These unsupported entry points are operator-library limitations rather than
JAX-side gaps; raise new requirements with the vendor team if needed.
- **Zero valid KV length:** rejected, including runtime `int32` zero values.
  The standard XLA path's unusual all-masked/zero-K behavior is not emulated.
  Zero valid Q length is supported only through the padded adapter's dummy path;
  physically zero-sized sequence dimensions remain unsupported.

## Standard DPA opt-in

The Python examples below share the imports in this first block and are intended
to run in document order. Later blocks explicitly replace their inputs where
needed. All six blocks were executed in fresh processes against the candidate
plugin; the dependent second block included the first block's setup.

```python
import jax
import jax.numpy as jnp
from jax._src.cudnn import fa_attention as fa

q = jnp.ones((2, 5, 4, 64), jnp.float16)
k = jnp.ones((2, 7, 2, 64), jnp.float16)
v = jnp.ones_like(k)
ql = jnp.array([5, 0], jnp.int32)
kl = jnp.array([3, 4], jnp.int32)

def attention(q, k, v, ql, kl):
    return jax.nn.dot_product_attention(
        q, k, v, query_seq_lengths=ql, key_value_seq_lengths=kl,
        implementation="hipc", return_residual=True)

o, residual = jax.jit(attention)(q, k, v, ql, kl)
# o: (B,T,N,D) = (2,5,4,64), input dtype
# residual: (B,T,N) = (2,5,4), input dtype, stop-gradient
# ql[1] == 0: all output rows and q/k/v gradients for that item are zero.

def loss(q, k, v):
    return attention(q, k, v, ql, kl)[0].astype(jnp.float32).sum()

dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
```

The supported standard-DPA subset is:

- Batched Q `(B,T,N,D)`, K/V `(B,S,K,D)` with positive physical dimensions,
  matching fp16/bf16 dtype and `N % K == 0` (MHA/GQA/MQA).
- Equal Q/K/V feature dimensions, `d == dv`, positive multiples of 32 up to 256.
  No implicit head-dimension padding or fp32 input support.
- `scale=None` means `1/sqrt(D)`. An explicit scale must be static, finite and
  positive in float32; a traced, nonpositive, overflowing or underflow-to-zero
  scale is not accepted.
- Optional independent `query_seq_lengths` and `key_value_seq_lengths`, each
  `int32[B]`. An omitted vector means the corresponding full physical capacity.
  Q lengths range from zero to T; KV lengths range from one to S. Existing host
  arrays must already have int32 dtype; int64 arrays are rejected before JAX
  can narrow them. Python integer sequences are checked for int32 range first.
- Full noncausal attention permits unequal valid Q/K lengths. `is_causal=True`
  or a local window requires equal valid Q/K lengths for each nonempty-Q item.
  Without length vectors, the physical Q/K lengths must then be equal. This
  prevents HIPC's right-aligned masking from silently differing from standard
  top-left masking. Empty-Q items use the dummy path instead.
- `local_window_size` follows the standard entry point's normalization (an
  integer or a pair); the resulting pair must be static and nonnegative.
  `is_causal` must also be static. Bias and arbitrary masks are rejected.

Without length vectors, the adapter uses the dense custom-VJP path. If either
length vector is supplied, it uses `fa_fwd_padded` and packed-varlen kernels.
`return_residual=True` returns standard **BTN** order in the **input dtype**, not
native `(B,N,T)` fp32 LSE. The residual is explicitly stop-gradient. Padded query
positions use XLA's large-negative fp32 sentinel before conversion to the input
dtype (which may convert it to `-inf`); this is distinct from the lower-level
padded API's fp32 `-inf` LSE convention.

## Independent padded Q/KV lengths

```python
# Reuse q/k/v and length vectors from the preceding example.
o, lse = jax.jit(lambda q, k, v, ql, kl: fa.fa_fwd_padded(
    q, k, v, ql, kl))(q, k, v, ql, kl)
# o: input layout/dtype; lse: (B,H,Q) fp32, stop-gradient
```

`fa_fwd_padded(q, k, v, query_seq_lengths, key_value_seq_lengths, *,
causal=False, softmax_scale=None, layout=1, window_size=(-1,-1))` shares the
DPA dtype, feature-dimension, scale and length restrictions above. It accepts
layout 1 `(B,S,H,D)` or layout 0 `(B,H,S,D)`; either length argument can be `None`
for full capacity. The lower-level `window_size=(-1,-1)` disables the window.
Causal or enabled local attention requires equal nonempty-Q valid lengths.

Valid rows are compacted into fixed-capacity `(B*Q,H,D)` and `(B*K,Hk,D)`
buffers, processed by varlen kernels, then gathered back to padded output.
Shapes and allocation capacities stay static even as runtime totals change.
Padded output rows are zero; their fp32 LSE is `-inf`. Their input gradients are
masked out. For an item with Q length zero, the adapter uses zero-valued,
length-one dummy Q/K/V, then masks out its outputs and all input gradients.
The original KV length must still be positive and valid. This is not support
for zero-length sequences in the native packed-varlen API.

## Dense forward and training

```python
q = jnp.ones((2, 128, 4, 64), jnp.float16)
k = jnp.ones((2, 128, 2, 64), jnp.float16)
v = jnp.ones_like(k)

o, lse = jax.jit(lambda q, k, v: fa.fa_fwd(q, k, v, causal=True))(q, k, v)
# o: (2,128,4,64), input dtype; lse: (2,4,128), fp32

def loss(q, k, v):
    o, _ = fa.fa_fwd_custom(q, k, v, causal=True)
    return o.astype(jnp.float32).sum()

dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
```

Common parameters:

- `layout=1`: q `(B,Q,H,D)`, k `(B,K,Hk,D)`, v `(B,K,Hk,Dv)`.
- `layout=0`: q `(B,H,Q,D)`, k `(B,Hk,K,D)`, v `(B,Hk,K,Dv)`.
- `H % Hk == 0` for GQA; gradients are reduced to the KV head count.
- `softmax_scale=None` selects `1/sqrt(D)`.
- `window_size=(-1,-1)` disables the local window. With causal attention,
  the right window is intersected with the causal restriction.
- `dropout_p=0.0`, `seed=0`: dense dropout parameters. The seed is a static
  Python value; forward and backward reuse its RNG state.
- `deterministic=True` is a `fa_fwd_custom` option for deterministic backward.
- `sinks` is a forward option in `fa_fwd`. Training sink values, sink plus
  dropout, and sink-specific gradients are not supported contracts.

Keep layout, causal, window and dropout configuration static under `jit`.
The candidate regressions exercise dense causal, GQA, sliding-window, dropout,
deterministic reverse mode and noncausal cross-attention. Passing these selected
cases does not establish every combination of their options.

## Packed variable-length attention and automatic reverse mode

```python
lengths = (64, 96)
cu = jnp.array([0, 64, 160], jnp.int32)
q = jnp.ones((192, 2, 64), jnp.float16)  # fixed capacity, total used = 160
k = jnp.ones((192, 1, 64), jnp.float16)
v = jnp.ones_like(k)

o, lse = fa.fa_fwd_varlen(q, k, v, cu, cu, 96, 96)
dq, dk, dv = jax.jit(jax.grad(
    lambda q, k, v: fa.fa_fwd_varlen(q, k, v, cu, cu, 96, 96)[0]
        .astype(jnp.float32).sum(), argnums=(0, 1, 2)))(q, k, v)

# Explicit backward remains available; pass the exact forward residuals.
dq, dk, dv = fa.fa_varlen_bwd(
    q, k, v, o, jnp.ones_like(o), lse, cu, cu, 96, 96, mode="kernel")

# Deterministic automatic backward: equal, explicit static host lengths.
f = jax.jit(lambda q, k, v, cu: fa.fa_fwd_varlen(
    q, k, v, cu, cu, 96, 96, deterministic=True,
    seq_lens_q=lengths, seq_lens_k=lengths))
o, lse = f(q, k, v, cu)
```

Contract:

- Q `(capacity_q,H,D)`, K `(capacity_k,Hk,D)`, V `(capacity_k,Hk,Dv)` are
  matching fp16/bf16, with positive capacities and `H % Hk == 0`. D and Dv
  must each be positive multiples of 32 no greater than 256.
- Cumulative offsets are matching `int32[B+1]` vectors. Each starts at zero,
  strictly increases, ends at or below the corresponding buffer capacity, and
  has positive differences bounded by its static `max_seqlen_q/k`. Maxima must
  be positive static int32 integers and cannot exceed the respective capacity.
  Device values are checked at runtime, not assumed valid from shape alone.
- Noncausal full attention permits independent Q/K lengths. Causal/window
  attention requires equal per-sequence Q/K lengths.
- Output shape is `(capacity_q,H,Dv)`; LSE is `(H,capacity_q)` fp32 and explicitly
  stop-gradient. The cumulative endpoint is a used-token count, **not** a new
  allocation size or LSE head stride. Unused capacity is not an extra sequence.
- `fa_fwd_varlen` now has a custom VJP for q/k/v first-order reverse mode,
  preserving exact forward O/LSE, checked lengths and static options. Default
  automatic backward calls `fa_varlen_bwd(mode="kernel")`, including GQA
  gradient reduction. Varlen dropout is not exposed.
- `sinks` remains forward-only. Attempting automatic differentiation with sinks
  raises rather than silently omitting gradients. LSE and integer metadata are
  not differentiable loss inputs; JVP and higher-order AD are unsupported.
- `deterministic=True` uses the self-attention dense-loop backward. Supply equal
  explicit static host `seq_lens_q` and `seq_lens_k` tuples/lists/arrays, including
  under `jit`. They must be positive, fit capacities/maxima and agree with the
  actual device cumulative offsets; equality is validated before use. Omit
  `seq_lens_*` for the ordinary automatic kernel path.

The explicit `fa_varlen_bwd` API retains `mode="kernel"` and `mode="dense_loop"`.
Its default can be configured through `FA_VARLEN_BWD`; the automatic VJP passes
its mode explicitly. Dense-loop backward launches per-sequence dense calls and
is limited to equal-length self-attention, not general cross-attention.

## K-prefix and shared padding-length attention

### `fa_fwd_attn_length`

```python
q = jnp.ones((2, 5, 4, 128), jnp.float16)
k = jnp.ones((2, 7, 2, 128), jnp.float16)
v = jnp.ones_like(k)

# Static Python integer: direct native length forward.
f_static = jax.jit(fa.fa_fwd_attn_length, static_argnames=("valid_k",))
o, lse = f_static(q, k, v, valid_k=3)

# Runtime scalar or per-batch int32 lengths: padded-varlen adapter.
f_dynamic = jax.jit(fa.fa_fwd_attn_length)
o, lse = f_dynamic(q, k, v, jnp.array(3, jnp.int32))
o, lse = f_dynamic(q, k, v, jnp.array([3, 6], jnp.int32))

dq, dk, dv = jax.grad(lambda q, k, v: fa.fa_fwd_attn_length(
    q, k, v, 3)[0].astype(jnp.float32).sum(), argnums=(0, 1, 2))(q, k, v)
```

Every query attends to the selected K/V prefix; Q is not trimmed and there is
no causal triangle. `valid_k` is either a static Python/NumPy integer, a runtime
`int32` scalar shared across the batch, or `int32[B]` for per-item K lengths.
Each value must satisfy `1 <= valid_k <= K`. Physical Q and K lengths may differ.

This API remains fp16/bf16, d=dv=128, MHA/GQA/MQA, layout 0 or 1. Output has Q's
shape/dtype; LSE is `(B,H,Q)` fp32 and stop-gradient. Static integer forward
keeps the direct native length fast path using typed FFI; its custom VJP uses
varlen backward and zero-pads unused K/V gradients. Dynamic scalar/vector
lengths go through `fa_fwd_padded` with all Q rows valid. Both routes provide
q/k/v first-order reverse mode, not JVP or higher-order AD. Zero K length,
causal/window options, dropout, bias and matrix masks are not exposed here.

The static forward remains typed FFI even with `FA_LEGACY=1`; dynamic forward
and backward use the applicable varlen path plus typed length validation.
Sequential `vmap` is not a fused length-attention batch kernel.

### `fa_fwd_padding_length`

```python
q = jnp.ones((2, 7, 4, 64), jnp.float16)
k = jnp.ones((2, 7, 2, 64), jnp.float16)
v = jnp.ones_like(k)
lengths = jnp.array([3, 6], jnp.int32)
o, lse = jax.jit(fa.fa_fwd_padding_length)(q, k, v, lengths)
```

`fa_fwd_padding_length(q, k, v, lengths, *, softmax_scale=None, layout=1)` uses
one `int32[B]` vector as both Q and KV valid lengths. Physical Q/K sequence
capacities must be equal. Lengths must be positive because they also specify KV
lengths. It delegates to the padded-varlen path and inherits its fp16/bf16,
d=dv multiples-of-32-through-256 subset, layout options, stop-gradient LSE and
q/k/v first-order reverse mode. Padded output rows are zero. It is **not** the
old `fa_fwd_padding_mask` API and accepts neither token masks nor matrix masks.

## Runtime validation, scratch ownership and cost

The new native `fa_validate_lengths_ffi` validator copies device length vectors
to the host, synchronizes **the same execution stream**, validates their values,
then returns checked device copies. Attention and packing/slicing consume those
copies, creating a required dependency: validation must complete **before the
vendor attention kernel reads the lengths**. This applies under `jit` and
sequential `vmap`, including varlen paths selected by `FA_LEGACY=1`. There is no
Python-side conversion of traced device values, but there is real native D2H
traffic and synchronization; deterministic host-length checks can add checks.

Native private workspace/accumulation scratch is scoped by device, stream and
host thread, rather than shared writable process-global storage. Each thread's
cache is bounded; eviction/thread exit synchronizes the owning device before
freeing retained allocations. This cleanup can stall that device. The static
length forward separately retains its own XLA-owned result scratch buffer; it
does not borrow mutable input constants. Dense dropout RNG handling now uses
HIP's actual device-to-device and device-to-host copy directions rather than
incorrect numeric direction constants, with a same-stream read/synchronize.
The candidate also corrects native padded head strides for dense BHSD backward
on non-aligned sequence lengths. These implementation fixes are not a numerical,
performance or concurrent-execution acceptance result by themselves.

Padded compaction/unpacking adds scatter/gather bandwidth and fixed-capacity
storage. Validation adds synchronization/copy costs; deterministic dense-loop
backward and sequential `vmap` add launches. The static length fast path is an
implementation choice, not a measured speedup. No zero-overhead, universal
speedup or throughput guarantee is made by these correctness-oriented changes.

## Batching and sharding

Sequential batching rules now cover dense forward/backward, packed varlen,
PA and prefix paths, including mixed mapped/unmapped operands and nonleading
mapped axes in the candidate regressions. Mapping executes complete kernel
calls; packed offsets remain local to each mapped item, and static options
remain static. Varlen and length-aware q/k/v gradient compositions have dedicated
regressions. Actual PA/prefix sequential-vmap numerical cases were compared
with explicit per-instance kernel calls in both ABI modes.
These are scoped transform rules, not a claim that every API, option, nested
transform or forward-only path supports differentiation.

Dense batch/head GQA partitioning was tested on GPU 4/5 (gfx936), including
both layouts and gradients under Shardy and the legacy partitioner.
Candidate rules require whole batches and whole KV head groups on local shards,
aligned Q/K/V ownership, and corresponding `(batch,head,replicated sequence)`
LSE ownership. True local SPMD partitioning is scoped to dense batch/head
attention with replicated sequence/features; there are no PA, prefix or length-
API local-partition guarantees. Sharded dropout and sinks are unsupported;
unsharded dense dropout remains available. Packed-varlen sequence ownership/
rebased offsets are not implemented as distributed sharding.

There is **no distributed sequence-attention implementation**, nor supported
feature-dimension partitioning. Shardy may all-gather, reshard or replicate
unsupported input shards before a local call, so an accepted input sharding or
successful compile does not prove distributed execution or guarantee an error
at the API boundary.
Acceptance must inspect lowered/compiled sharding and collectives as well as
numerical outputs and gradients. A single-device run, shape-only test, or
replicated fallback cannot validate true multi-device kernel partitioning.

## Other forward APIs

- `fa_fwd_kvcache(q, kcache, vcache, block_table, seqlens_k, max_seqlen_k, ...)`:
  q `(B,Q,H,D)`; caches `(blocks,page,Hk,D/Dv)`; block table int32
  `(B,max_blocks)`; KV lengths int32 `(B,)`, **not cumulative**. Returns o only.
  Static `max_seqlen_k` must bound every actual length. `num_splits=None` uses
  the existing split-KV heuristic. Sequential batching does not add backward.
- `fa_prefix_prefill(q, kcache, vcache, block_table, seqused_k, cu_seqlens_q,
  max_seqlen_q, max_seqlen_k, ...)`: q `(total_q,H,128)`, paged caches,
  per-batch KV lengths and cumulative Q offsets. Returns o and `(H,total_q)`
  LSE. Causal positions are right-aligned against the existing KV sequence.
  This forward-only prefix API is distinct from unsupported MLA-prefix.

These inference APIs must receive valid block indices and lengths. The new
padded/varlen validator is not a blanket promise of synchronous validation for
every PA/prefix device-side index or length.

## FP8 status

`fa_fwd_fp8` uses e4m3 inputs, fp32 descale arrays and fp16/bf16 outputs. Its
wrapper requires d=dv in `{128,192,256}` and a vendor-accepted architecture
(`arch >= 938`). FP8 PA uses `fa_fwd_kvcache` with descales, requires head
dimension 128, and accepts vendor arch 930 or >=938.

On gfx936, real FP8 numerical cases are skipped while guard/plumbing tests run.
`FA_DEBUG=5` deliberately suppresses vendor kernel execution; success in that
mode is **not numerical acceptance**. Existing architecture gates remain in
place. Do not infer FP8 production support from this capability expansion.

## Candidate tests and acceptance commands

Run from the candidate JAX checkout after installing its matching plugin and
staging the test files. These are acceptance commands, **not recorded passes**.
Do not use the HOST metadata directory or an unrelated temporary test fallback
as the installed candidate checkout.

```bash
export HIP_VISIBLE_DEVICES=4             # use the assigned device slot
export FA_REQUIRE_TESTS=1                # fail on missing prerequisites
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset FA_DEBUG                           # never suppress numerical kernels

for mode in 0 1; do
  FA_LEGACY=$mode python tests/fa_attention_test.py
  FA_LEGACY=$mode python tests/fa_attn_length_test.py
  FA_LEGACY=$mode python tests/fa_varlen_ad_test.py
  FA_LEGACY=$mode python tests/fa_dpa_test.py
  FA_LEGACY=$mode python tests/fa_length_extended_test.py
  FA_LEGACY=$mode python tests/fa_fp8_test.py
done

# Separate multi-device acceptance; do not silently substitute a single GPU.
for mode in 0 1; do
  HIP_VISIBLE_DEVICES=4,5 FA_RUN_BATCHING_GPU_TESTS=1 FA_LEGACY=$mode \
    python tests/fa_batching_test.py HIPCTransformAcceptanceTest
done
```

Candidate manual GPU target names (verify BUILD wiring before invoking them):

| Script | Bazel target | Focus |
|---|---|---|
| `fa_attention_test.py` | `//tests:fa_attention_test_gpu` | Dense/supported specialized API regressions; MLA positives replaced by unsupported-guard/registration regressions |
| `fa_attn_length_test.py` | `//tests:fa_attn_length_test_gpu` | Static length baseline, dynamic scalar reuse and gradient regression |
| `fa_varlen_ad_test.py` | `//tests:fa_varlen_ad_test_gpu` | Automatic/explicit reverse mode, capacity tails, runtime lengths, deterministic validation |
| `fa_dpa_test.py` | `//tests:fa_dpa_test_gpu` | Padded adapter, standard DPA subset, residual convention, native numerical checks |
| `fa_batching_test.py` | `//tests:fa_batching_test_gpu` | Sequential mapping and actual multi-device dense GQA partitioning |
| `fa_length_extended_test.py` | `//tests:fa_length_extended_test_gpu` | Static/dynamic prefix gradients and per-batch padding-length expansion |
| `fa_fp8_test.py` | `//tests:fa_fp8_test_gpu` | Architecture guards, plumbing and eligible numerical cases |

Use the fork's ROCm/DTK Bazel configuration and pass required variables through
`--test_env`. For example, after target integration:

```bash
bazel test --config=rocm \
  --test_env=HIP_VISIBLE_DEVICES=4 --test_env=FA_REQUIRE_TESTS=1 \
  --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false --test_env=FA_LEGACY=0 \
  //tests:fa_varlen_ad_test_gpu //tests:fa_dpa_test_gpu \
  //tests:fa_length_extended_test_gpu

bazel test --config=rocm \
  --test_env=HIP_VISIBLE_DEVICES=4,5 --test_env=FA_REQUIRE_TESTS=1 \
  --test_env=FA_RUN_BATCHING_GPU_TESTS=1 \
  --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false --test_env=FA_LEGACY=0 \
  //tests:fa_batching_test_gpu
# Repeat with --test_env=FA_LEGACY=1 and the same fork-specific build options.
```

Without `FA_REQUIRE_TESTS=1`, missing ROCm hardware, an absent default vendor
library, or an absent plugin can be explicit skips in prerequisite-aware tests.
An explicitly configured nonexistent `FA_LIB_PATH`, broken plugin imports,
registration failures and execution errors are failures, not fallback triggers.
The real batching acceptance class additionally requires
`FA_RUN_BATCHING_GPU_TESTS=1` and two visible devices. Architecture-specific FP8
skips must remain visible. Mocked adapter/shape checks and debug-suppressed
kernels must be reported separately from real native numerical execution.

The MLA-retirement plugin was rebuilt and exposes eight typed / six legacy
targets. Eleven CPU unsupported-guard tests passed in each ABI mode with the
vendor library unavailable, including eager/jit/grad/jit-grad/vmap and calls
without arrays; rejection occurs before plugin registration. The revised main
suite replaces two MLA-positive tests with one registration regression and these
eleven guards: 64 total. Typed FFI and legacy each passed 62 tests with two
multi-device skips. Two additional standard-DPA tests per ABI passed against the
rebuilt plugin, covering forward/backward/residuals and native HIPC lowering.

**Historical acceptance below predates MLA retirement; it is not a full rerun
of the revised suites.** Historical MLA passes do not imply current support.
The earlier plugin was built with Bazel 7.7.0/nanobind and loaded from the
development checkout. All seven manual GPU test targets passed Bazel dependency
analysis (`build --nobuild`); this is not an executed `bazel test` or a complete
wheel build.

The earlier gfx936 acceptance covered the original dense/inference regressions, packed
varlen AD, independent padded lengths, static/dynamic/per-batch K prefixes, and
the explicit DPA route in both ABI modes. The DPA matrix exercised d=32..256 in
steps of 32, fp16/bf16, dense and padded forward/gradients against fp32 references.
The multi-device suite additionally covered batch/head GQA partitioning in both
layouts, non-aligned BHSD backward, and sequential dense/PA/prefix batching.
Cold-process documentation examples passed. CPU tests separately exercised
metadata guards, transform tracing and unchanged default/XLA routing.

Single-card executions explicitly skipped the two original head-sharding cases;
the separate two-device suite exercised partitioning. Each FP8 suite retained
eight architecture-dependent numerical skips and four guard/plumbing passes.
No full JAX suite, other GPU architecture, benchmark, or unrestricted transform
combination is claimed. Keep actual exit status, source/plugin hashes and skipped
cases with each deployment's acceptance record.

## Troubleshooting and scope

- A missing `fa_attention_plugin` means the installed ROCm plugin lacks this
  module. Installing a vendor `.so` alone does not install the JAX FFI plugin.
- A vendor `dlopen` error requires checking `FA_LIB_PATH`, permissions and DTK
  runtime dependencies. Prefer typed FFI for readable native errors; legacy
  handlers do not uniformly provide native status errors.
- Do not set `FA_DEBUG=5` for numerical or performance runs. Uninitialized output
  from a suppressed kernel is not a valid attention result.
- Do not infer matrix masks from the HIPC field named `attn_mask`, or verified
  compile-time feature macros from ABI metadata.
- Bias/matrix-mask experiments, vendor binaries, generated `.so`/`.o`, raw probe
  logs and temporary reproduction scripts are outside this candidate's scope.
