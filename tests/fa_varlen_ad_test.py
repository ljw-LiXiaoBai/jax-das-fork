# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""打包变长注意力的一阶反向验收。

自动微分与显式反向均对照独立 fp32 参考，不互为真值。
参考实现将 GQA 的 KV 复制置于求导图内，以累加共享头梯度。
"""
import os
from unittest import mock

from absl.testing import absltest, parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu

config.parse_flags_with_absl()


def _load_fa():
  required = os.environ.get("FA_REQUIRE_TESTS", "0") == "1"

  def unavailable(reason):
    if required:
      raise RuntimeError(reason)
    raise absltest.SkipTest(reason)

  if not jtu.is_device_rocm():
    unavailable("ROCm GPU required for varlen AD tests")
  path = os.environ.get("FA_LIB_PATH", "/opt/libflash_attention.so")
  if not os.path.isfile(path):
    if "FA_LIB_PATH" in os.environ:
      raise FileNotFoundError(f"FA_LIB_PATH points to a missing file: {path}")
    unavailable(f"FlashAttention operator library not deployed: {path}")
  from jax._src.cudnn import fa_attention
  try:
    fa_attention._register()
  except ModuleNotFoundError as error:
    if error.name != "fa_attention_plugin":
      raise
    unavailable(str(error))
  return fa_attention


def _cu(lengths):
  return jnp.asarray(np.concatenate(([0], np.cumsum(lengths))), jnp.int32)


def _reference(q, k, v, lengths_q, lengths_k, *, causal=False,
               window=(-1, -1), scale=None):
  """Pure JAX fp32 packed attention; repeat KV inside the differentiated graph."""
  q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
  scale = q.shape[-1] ** -0.5 if scale is None else scale
  groups = q.shape[1] // k.shape[1]
  output = jnp.zeros((q.shape[0], q.shape[1], v.shape[2]), jnp.float32)
  lse = jnp.zeros((q.shape[1], q.shape[0]), jnp.float32)
  q0 = k0 = 0
  for sq, sk in zip(lengths_q, lengths_k):
    qq = q[q0:q0 + sq]
    kk = jnp.repeat(k[k0:k0 + sk], groups, axis=1)
    vv = jnp.repeat(v[k0:k0 + sk], groups, axis=1)
    logits = jnp.einsum("qhd,khd->hqk", qq, kk,
                        precision=jax.lax.Precision.HIGHEST) * scale
    qi = jnp.arange(sq)[:, None]
    ki = jnp.arange(sk)[None, :]
    mask = jnp.ones((sq, sk), dtype=bool)
    if causal:
      mask &= ki <= qi
    if window[0] >= 0:
      mask &= ki >= qi - window[0]
    if window[1] >= 0:
      mask &= ki <= qi + window[1]
    logits = jnp.where(mask[None], logits, -jnp.inf)
    ll = jax.scipy.special.logsumexp(logits, axis=-1)
    pp = jax.nn.softmax(logits, axis=-1)
    oo = jnp.einsum("hqk,khd->qhd", pp, vv,
                    precision=jax.lax.Precision.HIGHEST)
    output = output.at[q0:q0 + sq].set(oo)
    lse = lse.at[:, q0:q0 + sq].set(ll)
    q0 += sq
    k0 += sk
  return output, lse


def _inputs(lengths_q, lengths_k, dtype, *, heads=4, kv_heads=2,
            dim=64, value_dim=None, capacity_tail=0, seed=1):
  value_dim = dim if value_dim is None else value_dim
  rng = np.random.default_rng(seed)
  tq, tk = sum(lengths_q) + capacity_tail, sum(lengths_k) + capacity_tail
  shapes = ((tq, heads, dim), (tk, kv_heads, dim),
            (tk, kv_heads, value_dim), (tq, heads, value_dim))
  return tuple(jnp.asarray(rng.normal(0, .3 if i < 3 else .5, shape), dtype)
               for i, shape in enumerate(shapes))


class FaVarlenMetadataTest(absltest.TestCase):
  """Metadata and tracing checks allocate no device arrays or native buffers."""

  def test_length_conversion_rejects_narrowing_before_native_calls(self):
    from jax._src.cudnn import fa_attention as fa
    q = jax.ShapeDtypeStruct((1, 2, 1, 128), jnp.float16)
    k = jax.ShapeDtypeStruct((1, 3, 1, 128), jnp.float16)
    calls = {
        "prefix": lambda a, b, lengths: fa.fa_fwd_attn_length(a, b, b, lengths),
        "padded_q": lambda a, b, lengths: fa.fa_fwd_padded(a, b, b, lengths, None),
        "padded_k": lambda a, b, lengths: fa.fa_fwd_padded(a, b, b, None, lengths),
        "standard_q": lambda a, b, lengths: jax.nn.dot_product_attention(
            a, b, b, query_seq_lengths=lengths, implementation="hipc"),
        "standard_k": lambda a, b, lengths: jax.nn.dot_product_attention(
            a, b, b, key_value_seq_lengths=lengths, implementation="hipc"),
    }
    invalid = (np.array([1], np.int64), np.array([2 ** 32 + 1], np.int64),
               np.array(1, np.int64), np.array(2 ** 32 + 1, np.int64),
               np.array([1], np.uint32), np.array([True]),
               np.array([1.], np.float32), [True], [1.],
               [2 ** 31], [-(2 ** 31) - 1], (2 ** 32 + 1,),
               [np.bool_(True)], [np.float32(1.)])
    # 非法类型或超出 int32 范围的长度须在窄化转换和插件注册前被拒绝。
    with mock.patch.object(fa, "_register") as register, \
         mock.patch.object(fa, "_register_attn_length") as register_length:
      for x64 in (False, True):
        with jax.enable_x64(x64):
          for name, call in calls.items():
            for lengths in invalid:
              with self.subTest(x64=x64, route=name, lengths=repr(lengths)):
                with self.assertRaisesRegex((TypeError, ValueError), "int32"):
                  jax.make_jaxpr(lambda a, b: call(a, b, lengths))(q, k)
      register.assert_not_called()
      register_length.assert_not_called()

  def test_length_conversion_preserves_int32_and_other_backends(self):
    from jax._src.cudnn import fa_attention as fa
    from jax._src.nn import functions as nn_functions
    with jax.enable_x64(False):
      for lengths in ([1, 2], (1, 2), [np.int64(1), np.int32(2)],
                      [-(2 ** 31), 2 ** 31 - 1],
                      np.array([1, 2], np.int32), np.array(1, np.int32)):
        result = jax.eval_shape(lambda: fa._length_array(lengths, "lengths"))
        self.assertEqual(result.dtype, jnp.int32)
        self.assertEqual(result.shape, np.shape(lengths))
      for dtype in (jnp.int32, jnp.int64, jnp.uint32, jnp.bool_, jnp.float32):
        with jax.enable_x64(True):
          arg = jax.ShapeDtypeStruct((2,), dtype)
          if dtype == jnp.int32:
            result = jax.eval_shape(lambda x: fa._length_array(x, "lengths"), arg)
            self.assertEqual(result.dtype, jnp.int32)
          else:
            with self.assertRaisesRegex(TypeError, "int32"):
              jax.eval_shape(lambda x: fa._length_array(x, "lengths"), arg)
      scalar = jax.ShapeDtypeStruct((), jnp.int32)
      result = jax.eval_shape(lambda x: fa._length_array(x, "lengths"),
                              [scalar, scalar])
      self.assertEqual(result.shape, (2,))
      self.assertEqual(result.dtype, jnp.int32)

      q = jax.ShapeDtypeStruct((1, 2, 1, 128), jnp.float16)
      k = jax.ShapeDtypeStruct((1, 3, 1, 128), jnp.float16)
      def fake_static(a, b, c, valid_k, scale, layout):
        return jnp.zeros_like(a), jnp.zeros((1, 1, 2), jnp.float32)
      with mock.patch.object(fa, "_attn_length_static", side_effect=fake_static) as fast:
        for length in (1, np.int64(1)):
          jax.eval_shape(lambda a, b: fa.fa_fwd_attn_length(a, b, b, length), q, k)
        self.assertEqual(fast.call_count, 2)

      def fake_adapter(a, b, c, bias, mask, ql, kl, **kwargs):
        self.assertEqual(ql.dtype, jnp.int32)
        self.assertEqual(kl.dtype, jnp.int32)
        return jnp.zeros_like(a)
      with mock.patch.object(fa, "dot_product_attention", side_effect=fake_adapter) as adapter:
        for ql, kl in (([1], (2,)), (np.array([1], np.int32), np.array([2], np.int32))):
          jax.eval_shape(lambda a, b: jax.nn.dot_product_attention(
              a, b, b, query_seq_lengths=ql, key_value_seq_lengths=kl,
              implementation="hipc"), q, k)
        self.assertEqual(adapter.call_count, 2)

      def fake_xla(a, b, c, *args, **kwargs):
        self.assertEqual(kwargs["kv_seqlen"].dtype, jnp.int32)
        return jnp.zeros_like(a)
      with mock.patch.object(fa, "_length_array", side_effect=AssertionError("HIPC only")), \
           mock.patch.object(nn_functions, "_dot_product_attention_xla", side_effect=fake_xla):
        for backend in ("xla", None):
          jax.eval_shape(lambda a, b: jax.nn.dot_product_attention(
              a, b, b, key_value_seq_lengths=np.array([1], np.int64),
              implementation=backend), q, k)

  def test_native_index_products_and_float32_scale(self):
    from jax._src.cudnn import fa_attention as fa
    def array(shape, dtype=jnp.float16):
      return jax.ShapeDtypeStruct(shape, dtype)
    cu = array((2,), jnp.int32)
    q, k, v = (array((33, 4, 64)), array((33, 1, 64)), array((33, 1, 64)))
    for scale in (float("inf"), float("nan"), 1e300, 1e-300, 0., -1.):
      with self.assertRaisesRegex(ValueError, "float32"):
        fa._varlen_metadata(q, k, v, cu, cu, 33, 33, scale, (-1, -1))
    huge_q = array((1 << 25, 4, 64))
    huge_k = array((1 << 25, 1, 64))
    for qq, kk, vv in ((huge_q, k, v), (q, huge_k, huge_k)):
      with self.assertRaisesRegex(ValueError, "int32"):
        fa._varlen_metadata(qq, kk, vv, cu, cu, 33, 33, None, (-1, -1))
    many_cu = array((1 << 24,), jnp.int32)
    with self.assertRaisesRegex(ValueError, "int32"):
      fa._varlen_metadata(q, k, v, many_cu, many_cu, 33, 33, None, (-1, -1))
    with self.assertRaisesRegex(ValueError, "capacity"):
      fa._varlen_metadata(q, k, v, cu, cu, 34, 33, None, (-1, -1))

  def test_backward_rejects_experimental_layouts_and_malformed_residuals(self):
    from jax._src.cudnn import fa_attention as fa
    q = jax.ShapeDtypeStruct((33, 4, 64), jnp.float16)
    k = jax.ShapeDtypeStruct((33, 1, 64), jnp.float16)
    cu = jax.ShapeDtypeStruct((2,), jnp.int32)
    lse = jax.ShapeDtypeStruct((4, 33), jnp.float32)
    args = (q, k, k, q, q, lse, cu, cu, 33, 33)
    for opts in (dict(lse_unpadded=0), dict(lse_unpadded=2), dict(vbwd_mode=1)):
      with self.assertRaisesRegex(ValueError, "packed"):
        fa.fa_varlen_bwd(*args, **opts)
    for bad in (jax.ShapeDtypeStruct((4, 32), jnp.float32),
                jax.ShapeDtypeStruct((4, 33), jnp.float16)):
      with self.assertRaisesRegex(ValueError, "residual"):
        fa.fa_varlen_bwd(*args[:5], bad, *args[6:])


class FaVarlenADTest(jtu.JaxTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.fa = _load_fa()

  def assertClose(self, actual, expected, dtype):
    tol = 3e-3 if dtype == jnp.float16 else 1.5e-2
    self.assertArraysAllClose(actual, expected, rtol=tol, atol=tol,
                              check_dtypes=False)

  def assertGradients(self, actual, expected, inputs, dtype):
    for got, want, primal in zip(actual, expected, inputs):
      self.assertEqual(got.shape, primal.shape)
      self.assertEqual(got.dtype, primal.dtype)
      self.assertTrue(np.isfinite(np.asarray(got, np.float32)).all())
      self.assertClose(got, want, dtype)

  def _check_case(self, dtype, kv_heads, lengths_q, lengths_k, causal, window,
                  *, dim=64, capacity_tail=0):
    q, k, v, do = _inputs(lengths_q, lengths_k, dtype, kv_heads=kv_heads,
                           dim=dim, capacity_tail=capacity_tail)
    cuq, cuk = _cu(lengths_q), _cu(lengths_k)
    # Nondefault scale verifies that AD reuses the exact forward options.
    scale = .173
    opts = dict(causal=causal, softmax_scale=scale, window_size=window)

    def forward(a, b, c, cq, ck):
      return self.fa.fa_fwd_varlen(a, b, c, cq, ck,
                                   max(lengths_q), max(lengths_k), **opts)

    def loss(a, b, c, cq, ck):
      oo, ll = forward(a, b, c, cq, ck)
      return (oo.astype(jnp.float32) * do.astype(jnp.float32)).sum(), ll

    def ref_loss(a, b, c):
      oo, _ = _reference(a, b, c, lengths_q, lengths_k,
                          causal=causal, window=window, scale=scale)
      return (oo * do.astype(jnp.float32)).sum()

    primals = (q, k, v)
    fp32 = tuple(x.astype(jnp.float32) for x in primals)
    expected = jax.grad(ref_loss, argnums=(0, 1, 2))(*fp32)
    ref_o, ref_lse = _reference(*fp32, lengths_q, lengths_k,
                               causal=causal, window=window, scale=scale)
    o, lse = forward(*primals, cuq, cuk)
    self.assertClose(o, ref_o, dtype)
    self.assertArraysAllClose(lse[:, :sum(lengths_q)], ref_lse[:, :sum(lengths_q)],
                              rtol=2e-4, atol=2e-4)
    differentiated = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)
    for run in (differentiated, jax.jit(differentiated)):
      (value, aux), gradients = run(*primals, cuq, cuk)
      self.assertClose(value, ref_loss(*fp32), dtype)
      self.assertArraysAllClose(aux, lse, rtol=2e-4, atol=2e-4)
      self.assertGradients(gradients, expected, primals, dtype)
      if capacity_tail:
        for grad, used in zip(gradients, (sum(lengths_q), sum(lengths_k), sum(lengths_k))):
          self.assertArraysEqual(grad[used:], jnp.zeros_like(grad[used:]))
    # Explicit kernel mode is checked independently against the fp32 oracle too.
    explicit = self.fa.fa_varlen_bwd(
        q, k, v, o, do, lse, cuq, cuk, max(lengths_q), max(lengths_k),
        mode="kernel", **opts)
    self.assertGradients(explicit, expected, primals, dtype)

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16),
      kv_heads=(1, 4),
      case=(
          ((33, 65), (33, 65), False, (-1, -1)),
          ((33, 65), (33, 65), True, (-1, -1)),
          ((33, 65), (33, 65), False, (16, 24)),
          ((33, 65), (33, 65), True, (16, 24)),
          ((33, 49), (97, 65), False, (-1, -1)),
      ))
  def test_automatic_reverse_and_explicit_kernel(self, dtype, kv_heads, case):
    self._check_case(dtype, kv_heads, *case)

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16))
  def test_dim128_and_capacity_tail(self, dtype):
    self._check_case(dtype, 2, (33, 49), (97, 65), False, (-1, -1),
                     dim=128, capacity_tail=17)

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16), shared_kv=(False, True))
  def test_vmap_grad_and_grad_vmap(self, dtype, shared_kv):
    # Same capacities/maxima, different batch boundaries, and non-leading mapped Q.
    lq = ((33, 49), (41, 41))
    lk = ((97, 65), (81, 81))
    samples = [_inputs(lq[i], lk[i], dtype, seed=21 + i) for i in range(2)]
    qq = jnp.stack([x[0] for x in samples], axis=1)
    kk = samples[0][1] if shared_kv else jnp.stack([x[1] for x in samples])
    vv = samples[0][2] if shared_kv else jnp.stack([x[2] for x in samples])
    dd = jnp.stack([x[3] for x in samples])
    cq = jnp.stack([_cu(x) for x in lq])
    ck = _cu(lk[0]) if shared_kv else jnp.stack([_cu(x) for x in lk])
    axes = (1, None if shared_kv else 0, None if shared_kv else 0,
            0, 0, None if shared_kv else 0)

    def loss(a, b, c, d, aq, ak):
      o, _ = self.fa.fa_fwd_varlen(a, b, c, aq, ak, 49, 97)
      return (o.astype(jnp.float32) * d.astype(jnp.float32)).sum()

    args = (qq, kk, vv, dd, cq, ck)
    mapped_grad = jax.vmap(jax.grad(loss, argnums=(0, 1, 2)), in_axes=axes)
    expected = []
    for i in range(2):
      a, b, c, d = samples[i]
      if shared_kv:
        b, c = kk, vv
      def ref(aa, bb, cc):
        oo, _ = _reference(aa, bb, cc, lq[i], lk[0] if shared_kv else lk[i])
        return (oo * d.astype(jnp.float32)).sum()
      expected.append(jax.grad(ref, argnums=(0, 1, 2))(
          *(x.astype(jnp.float32) for x in (a, b, c))))
    stacked = tuple(jnp.stack([g[i] for g in expected]) for i in range(3))
    for run in (mapped_grad, jax.jit(mapped_grad)):
      for got, want in zip(run(*args), stacked):
        self.assertClose(got, want, dtype)
    summed_loss = lambda *xs: jax.vmap(loss, in_axes=axes)(*xs).sum()
    gradients = jax.jit(jax.grad(summed_loss, argnums=(0, 1, 2)))(*args)
    self.assertClose(gradients[0], jnp.moveaxis(stacked[0], 0, 1), dtype)
    for got, want in zip(gradients[1:], stacked[1:]):
      self.assertClose(got, want.sum(0) if shared_kv else want, dtype)

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16))
  def test_lse_is_explicitly_stop_gradient_auxiliary(self, dtype):
    lengths = (33, 49)
    q, k, v, do = _inputs(lengths, lengths, dtype)
    cu = _cu(lengths)
    def loss(a, b, c, lse_weight):
      o, lse = self.fa.fa_fwd_varlen(a, b, c, cu, cu, 49, 49)
      return (o.astype(jnp.float32) * do.astype(jnp.float32)).sum() + lse_weight * lse.sum()
    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    ordinary = grad(q, k, v, 0.)
    with_aux = grad(q, k, v, 7.)
    for got, expected in zip(with_aux, ordinary):
      self.assertClose(got, expected, dtype)
    lse_only = jax.jit(jax.grad(
        lambda a: self.fa.fa_fwd_varlen(a, k, v, cu, cu, 49, 49)[1].sum()))(q)
    self.assertArraysEqual(lse_only, jnp.zeros_like(q))

  def test_sinks_ad_is_rejected_not_silently_dropped(self):
    lengths = (33,)
    q, k, v, _ = _inputs(lengths, lengths, jnp.float16)
    cu = _cu(lengths)
    sinks = jnp.zeros((q.shape[1],), jnp.float32)
    def loss(a, s):
      return self.fa.fa_fwd_varlen(a, k, v, cu, cu, 33, 33, sinks=s)[0].sum()
    for argnums in (0, 1):
      with self.assertRaisesRegex(ValueError, "sinks"):
        jax.jit(jax.grad(loss, argnums=argnums))(q, sinks)

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16))
  def test_deterministic_host_lengths_and_capacity(self, dtype):
    lengths = (33, 49)
    q, k, v, do = _inputs(lengths, lengths, dtype, capacity_tail=9)
    cu = _cu(lengths)
    def loss(a, b, c, cq, ck):
      o, _ = self.fa.fa_fwd_varlen(
          a, b, c, cq, ck, 49, 49, deterministic=True,
          seq_lens_q=lengths, seq_lens_k=lengths)
      return (o.astype(jnp.float32) * do.astype(jnp.float32)).sum()
    def ref(a, b, c):
      o, _ = _reference(a, b, c, lengths, lengths)
      return (o * do.astype(jnp.float32)).sum()
    expected = jax.grad(ref, argnums=(0, 1, 2))(
        *(x.astype(jnp.float32) for x in (q, k, v)))
    run = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    first = run(q, k, v, cu, cu)
    second = run(q, k, v, cu, cu)
    self.assertGradients(first, expected, (q, k, v), dtype)
    for a, b in zip(first, second):
      self.assertArraysEqual(a, b)
      self.assertArraysEqual(a[sum(lengths):], jnp.zeros_like(a[sum(lengths):]))

  def test_deterministic_device_and_host_lengths_must_agree(self):
    q, k, v, _ = _inputs((33, 49), (33, 49), jnp.float16)
    def run(cu):
      return self.fa.fa_fwd_varlen(
          q, k, v, cu, cu, 49, 49, deterministic=True,
          seq_lens_q=(33, 49), seq_lens_k=(33, 49))[0]
    with self.assertRaisesRegex(Exception, "length|equal|cumulative|seqlen|sequence"):
      jax.jit(run)(_cu((41, 41))).block_until_ready()

  @parameterized.parameters(
      (1, 33), (0, 0), (0, 34), (0, -1), (0, 17, 16))
  def test_invalid_cumulative_lengths_fail_before_attention(self, *values):
    q, k, v, _ = _inputs((33,), (33,), jnp.float16)
    cu = jnp.asarray(values, jnp.int32)
    def run(cq):
      return self.fa.fa_fwd_varlen(q, k, v, cq, cq, 33, 33)[0]
    with self.assertRaisesRegex(Exception, "length|capacity|cumulative|seqlen|sequence"):
      jax.jit(run)(cu).block_until_ready()

  def test_masked_cross_lengths_rejected(self):
    q, k, v, _ = _inputs((33,), (97,), jnp.float16)
    for opts in (dict(causal=True), dict(window_size=(16, 16))):
      run = jax.jit(lambda cq, ck: self.fa.fa_fwd_varlen(
          q, k, v, cq, ck, 33, 97, **opts)[0])
      with self.assertRaisesRegex(Exception, "length|equal|seqlen|sequence"):
        run(_cu((33,)), _cu((97,))).block_until_ready()

  def test_static_metadata_validation(self):
    q, k, v, _ = _inputs((33,), (33,), jnp.float16)
    cu = _cu((33,))
    cases = (
        (q.astype(jnp.float32), k, v, cu, cu, 33, 33),
        (q, k.astype(jnp.bfloat16), v, cu, cu, 33, 33),
        (q[:, :3], k, v, cu, cu, 33, 33),
        (q[..., :63], k[..., :63], v, cu, cu, 33, 33),
        (q, k, v, cu.astype(jnp.float32), cu, 33, 33),
        (q, k, v, cu, jnp.array([0, 16, 33], jnp.int32), 33, 33),
        (q, k, v, cu, cu, 0, 33),
    )
    for args in cases:
      with self.assertRaises((ValueError, TypeError)):
        jax.eval_shape(lambda *arrays: self.fa.fa_fwd_varlen(
            *arrays, max_seqlen_q=args[5], max_seqlen_k=args[6]), *args[:5])
    with self.assertRaisesRegex(ValueError, "static host lengths"):
      self.fa.fa_fwd_varlen(q, k, v, cu, cu, 33, 33, deterministic=True)


if __name__ == "__main__":
  absltest.main()
