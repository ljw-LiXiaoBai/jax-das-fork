# Copyright 2026 The JAX Authors.
#
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
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

"""DPA 适配测试。

CPU 用例模拟长度校验与内核，仅验证打包和路由，不作为生产回退。
ROCm 用例另行验证原生执行。
"""
import os
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
import numpy as np

config.parse_flags_with_absl()


def _inputs(b=3, sq=5, sk=7, h=4, hk=2, d=32, dtype=jnp.float16):
  rng = np.random.default_rng(840)
  return tuple(jnp.asarray(rng.normal(0, .3, shape), dtype) for shape in
               ((b, sq, h, d), (b, sk, hk, d), (b, sk, hk, d)))


def _packed_reference(q, k, v, cuq, cuk, msq, msk, *, causal=False,
                      softmax_scale=None, window_size=(-1, -1)):
  del msq, msk
  qr = jnp.arange(q.shape[0])
  kr = jnp.arange(k.shape[0])
  qb = jnp.minimum(jnp.sum(qr[:, None] >= cuq[None, 1:], axis=1), cuq.size - 2)
  kb = jnp.minimum(jnp.sum(kr[:, None] >= cuk[None, 1:], axis=1), cuk.size - 2)
  qpos = qr - cuq[qb]
  kpos = kr - cuk[kb]
  delta = (cuk[qb + 1] - cuk[qb]) - (cuq[qb + 1] - cuq[qb])
  center = qpos + delta
  mask = ((qb[:, None] == kb[None, :]) & (qr[:, None] < cuq[-1]) &
          (kr[None, :] < cuk[-1]))
  if causal:
    mask &= kpos[None, :] <= center[:, None]
  left, right = window_size
  if left >= 0:
    mask &= kpos[None, :] >= center[:, None] - left
  if right >= 0:
    mask &= kpos[None, :] <= center[:, None] + right
  groups = q.shape[1] // k.shape[1]
  kk = jnp.repeat(k.astype(jnp.float32), groups, axis=1)
  vv = jnp.repeat(v.astype(jnp.float32), groups, axis=1)
  scale = q.shape[-1] ** -.5 if softmax_scale is None else softmax_scale
  logits = jnp.sum(q.astype(jnp.float32)[:, None] * kk[None], axis=-1).transpose(2, 0, 1) * scale
  logits = jnp.where(mask[None], logits, -1e30)
  maximum = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
  weights = jnp.exp(logits - maximum)
  probs = weights / jnp.sum(weights, axis=-1, keepdims=True)
  out = jnp.sum(probs.transpose(1, 2, 0)[..., None] * vv[None], axis=1).astype(q.dtype)
  out = jnp.where((qr < cuq[-1])[:, None, None], out, 0)
  lse = jax.nn.logsumexp(logits, axis=-1)
  lse = jnp.where((qr < cuq[-1])[None], lse, 0)
  return out, jax.lax.stop_gradient(lse)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PaddedAdapterTest(jtu.JaxTestCase):
  """Pure packing/routing tests; no plugin or GPU is required."""

  def setUp(self):
    super().setUp()
    from jax._src.cudnn import fa_attention
    self.fa = fa_attention
    self.validation = mock.patch.object(
        self.fa, "_validated_lengths", side_effect=lambda q, k, **kw: (q, k),
        create=True)
    self.validation_mock = self.validation.start()
    self.addCleanup(self.validation.stop)
    self.kernel = mock.patch.object(self.fa, "fa_fwd_varlen",
                                    side_effect=_packed_reference)
    self.kernel_mock = self.kernel.start()
    self.addCleanup(self.kernel.stop)

  def test_pack_static_capacity_and_zero_dummy(self):
    x = jnp.arange(3 * 4 * 2, dtype=jnp.float32).reshape(3, 4, 2, 1)
    lengths = jnp.array([2, 1, 3], jnp.int32)
    cu = jnp.array([0, 2, 3, 6], jnp.int32)
    active = jnp.array([True, False, True])
    actual = jax.jit(self.fa._pack_padded_attention)(x, lengths, cu, active)
    expected = np.zeros((12, 2, 1), np.float32)
    expected[:2] = x[0, :2]
    expected[3:6] = x[2, :3]
    np.testing.assert_array_equal(actual, expected)
    grad = jax.grad(lambda a: self.fa._pack_padded_attention(a, lengths, cu, active).sum())(x)
    expected_grad = np.zeros(x.shape, np.float32)
    expected_grad[0, :2] = 1
    expected_grad[2, :3] = 1
    np.testing.assert_array_equal(grad, expected_grad)

  def test_independent_lengths_eager_jit_and_grad(self):
    q, k, v = _inputs()
    ql = jnp.array([0, 3, 1], jnp.int32)
    kl = jnp.array([4, 5, 2], jnp.int32)
    fn = lambda a, b, c, lq, lk: self.fa.fa_fwd_padded(a, b, c, lq, lk)
    ref = lambda a, b, c: jax.nn.dot_product_attention(
        a, b, c, query_seq_lengths=ql, key_value_seq_lengths=kl,
        implementation="xla")
    expected = ref(q, k, v)
    for actual in (fn(q, k, v, ql, kl), jax.jit(fn)(q, k, v, ql, kl)):
      np.testing.assert_allclose(actual[0], expected, rtol=3e-3, atol=3e-3)
      self.assertEqual(actual[1].shape, (3, 4, 5))
      self.assertEqual(actual[1].dtype, jnp.float32)
      np.testing.assert_array_equal(actual[0][0], 0)
      self.assertTrue(np.isneginf(np.asarray(actual[1][0])).all())
    actual_grad = jax.jit(jax.grad(
        lambda a, b, c: fn(a, b, c, ql, kl)[0].astype(jnp.float32).sum(),
        argnums=(0, 1, 2)))(q, k, v)
    expected_grad = jax.grad(lambda a, b, c: ref(a, b, c).astype(jnp.float32).sum(),
                             argnums=(0, 1, 2))(q, k, v)
    for actual, expected in zip(actual_grad, expected_grad):
      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=3e-3)
      np.testing.assert_array_equal(actual[0], 0)
    for i, (nq, nk) in enumerate(zip(np.asarray(ql), np.asarray(kl))):
      np.testing.assert_array_equal(actual_grad[0][i, nq:], 0)
      np.testing.assert_array_equal(actual_grad[1][i, nk:], 0)
      np.testing.assert_array_equal(actual_grad[2][i, nk:], 0)

  def test_single_missing_length_layout_and_runtime_reuse(self):
    q, k, v = _inputs()
    fn = jax.jit(lambda a, b, c, lk: self.fa.fa_fwd_padded(
        a, b, c, None, lk, layout=0))
    for lengths in ([1, 3, 6], [4, 2, 7]):
      kl = jnp.array(lengths, jnp.int32)
      actual, _ = fn(*(x.transpose(0, 2, 1, 3) for x in (q, k, v)), kl)
      expected = jax.nn.dot_product_attention(q, k, v,
          key_value_seq_lengths=kl, implementation="xla")
      np.testing.assert_allclose(actual.transpose(0, 2, 1, 3), expected,
                                  rtol=3e-3, atol=3e-3)
    actual, _ = self.fa.fa_fwd_padded(q, k, v, jnp.array([2, 1, 5], jnp.int32), None)
    self.assertEqual(actual.shape, q.shape)

  def test_equal_lengths_causal_window_and_zero_query(self):
    q, k, v = _inputs()
    ql = jnp.array([0, 3, 5], jnp.int32)
    kl = jnp.array([7, 3, 5], jnp.int32)
    for causal, window in ((True, (-1, -1)), (False, (0, 0)),
                           (False, (2, 1)), (True, (2, 1))):
      with self.subTest(causal=causal, window=window):
        actual, _ = self.fa.fa_fwd_padded(q, k, v, ql, kl,
                                         causal=causal, window_size=window)
        expected = jax.nn.dot_product_attention(q, k, v,
            query_seq_lengths=ql, key_value_seq_lengths=kl,
            is_causal=causal,
            local_window_size=None if window == (-1, -1) else window,
            implementation="xla")
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)
        self.assertTrue(self.validation_mock.call_args.kwargs["require_equal"])

  def test_all_empty_queries_ignore_nan_inputs(self):
    q, k, v = (jnp.full(x.shape, jnp.nan, x.dtype) for x in _inputs())
    ql, kl = jnp.zeros((3,), jnp.int32), jnp.array([2, 3, 4], jnp.int32)
    fn = lambda a, b, c: self.fa.fa_fwd_padded(a, b, c, ql, kl)[0]
    actual = jax.jit(fn)(q, k, v)
    np.testing.assert_array_equal(actual, 0)
    for grad in jax.grad(lambda a, b, c: fn(a, b, c).sum(), argnums=(0, 1, 2))(q, k, v):
      np.testing.assert_array_equal(grad, 0)

  def test_residual_dtype_padding_and_stop_gradient(self):
    for dtype in (jnp.float16, jnp.bfloat16):
      q, k, v = _inputs(dtype=dtype)
      ql, kl = jnp.array([0, 2, 4], jnp.int32), jnp.array([2, 3, 5], jnp.int32)
      fn = lambda a: self.fa.dot_product_attention(
          a, k, v, query_seq_lengths=ql, key_value_seq_lengths=kl,
          return_residual=True)
      actual = fn(q)
      expected = jax.nn.dot_product_attention(q, k, v,
          query_seq_lengths=ql, key_value_seq_lengths=kl,
          return_residual=True, implementation="xla")
      self.assertEqual(actual[1].dtype, dtype)
      self.assertEqual(actual[1].shape, q.shape[:-1])
      np.testing.assert_allclose(actual[1].astype(jnp.float32),
                                 expected[1].astype(jnp.float32), rtol=.02, atol=.02)
      np.testing.assert_array_equal(jax.grad(lambda a: fn(a)[1].astype(jnp.float32).sum())(q), 0)

  def test_vmap_padded_and_grad(self):
    q, k, v = _inputs()
    ql = jnp.array([[0, 2, 3], [1, 4, 5]], jnp.int32)
    kl = jnp.array([[1, 4, 6], [2, 5, 7]], jnp.int32)
    qs, ks, vs = (jnp.stack((x, x * .5)) for x in (q, k, v))
    fn = lambda a, b, c, lq, lk: self.fa.fa_fwd_padded(a, b, c, lq, lk)[0]
    actual = jax.jit(jax.vmap(fn))(qs, ks, vs, ql, kl)
    for i in range(2):
      expected = jax.nn.dot_product_attention(qs[i], ks[i], vs[i],
          query_seq_lengths=ql[i], key_value_seq_lengths=kl[i], implementation="xla")
      np.testing.assert_allclose(actual[i], expected, rtol=3e-3, atol=3e-3)
    gradient = jax.jit(jax.grad(lambda a: jax.vmap(fn)(a, ks, vs, ql, kl).sum()))(qs)
    np.testing.assert_array_equal(gradient[0, 0], 0)

  def test_standard_dispatch_and_unbatched_result_shapes(self):
    q, k, v = _inputs(b=1)
    with mock.patch.object(self.fa, "dot_product_attention", return_value=(
        jnp.zeros_like(q), jnp.zeros(q.shape[:-1], q.dtype))) as adapter:
      actual = jax.nn.dot_product_attention(q[0], k[0], v[0],
          implementation="hipc", return_residual=True, local_window_size=2)
    self.assertEqual(actual[0].shape, q[0].shape)
    self.assertEqual(actual[1].shape, q[0].shape[:-1])
    self.assertEqual(adapter.call_args.kwargs["local_window_size"], (2, 2))

  def test_dense_uses_custom_vjp_entry(self):
    q, k, v = _inputs()
    with mock.patch.object(self.fa, "fa_fwd_custom", return_value=(
        jnp.zeros_like(q), jnp.zeros((3, 4, 5), jnp.float32))) as dense:
      actual = self.fa.dot_product_attention(q, k, v)
    self.assertEqual(actual.shape, q.shape)
    dense.assert_called_once()

  def test_subset_guards_before_kernel(self):
    q, k, v = _inputs()
    for scale in (0., -1., np.inf, np.nan, 1e-100):
      with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "positive"):
        self.fa.dot_product_attention(q, k, v, scale=scale)
    for kw in ({"bias": jnp.zeros((1, 1, 1, 1))},
               {"mask": jnp.ones((1, 1, 1, 1), bool)}):
      with self.assertRaises(NotImplementedError):
        self.fa.dot_product_attention(q, k, v, **kw)
    for kw in ({"is_causal": True}, {"local_window_size": (1, 2)}):
      with self.assertRaisesRegex(ValueError, "equal Q/K"):
        self.fa.dot_product_attention(q, k, v, **kw)
    with self.assertRaisesRegex(ValueError, "nonnegative"):
      self.fa.dot_product_attention(q, k, v, local_window_size=(-1, -1))
    with self.assertRaisesRegex(TypeError, "fp16/bf16"):
      self.fa.dot_product_attention(q.astype(jnp.float32), k, v)
    with self.assertRaisesRegex(ValueError, "head dimension"):
      self.fa.dot_product_attention(q[..., :24], k[..., :24], v[..., :24])
    with self.assertRaisesRegex(ValueError, "int32"):
      self.fa.fa_fwd_padded(q, k, v, jnp.ones((3,), jnp.float32), None)
    self.kernel_mock.assert_not_called()


def _load_rocm():
  required = os.environ.get("FA_REQUIRE_TESTS", "0") == "1"
  def unavailable(reason):
    if required:
      raise RuntimeError(reason)
    raise absltest.SkipTest(reason)
  if not jtu.is_device_rocm():
    unavailable("ROCm GPU required")
  path = os.environ.get("FA_LIB_PATH", "/opt/libflash_attention.so")
  if not os.path.isfile(path):
    if "FA_LIB_PATH" in os.environ:
      raise FileNotFoundError(path)
    unavailable(f"HIPC library unavailable: {path}")
  from jax._src.cudnn import fa_attention
  try:
    fa_attention._register()
  except ModuleNotFoundError as e:
    if e.name != "fa_attention_plugin":
      raise
    unavailable(str(e))
  return fa_attention


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class DpaRocmTest(jtu.JaxTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.fa = _load_rocm()

  def test_dpa_forward_backward_and_residual(self):
    for dtype in (jnp.float16, jnp.bfloat16):
      q, k, v = _inputs(sq=17, sk=33, d=128, dtype=dtype)
      for ql, kl, causal, window in (
          (None, None, False, None),
          ([0, 13, 7], [5, 19, 11], False, None),
          ([0, 13, 7], [5, 13, 7], True, (2, 1)),
          ([0, 13, 7], [5, 13, 7], False, (0, 0))):
        ql = None if ql is None else jnp.array(ql, jnp.int32)
        kl = None if kl is None else jnp.array(kl, jnp.int32)
        options = dict(query_seq_lengths=ql, key_value_seq_lengths=kl,
                       is_causal=causal, local_window_size=window)
        fn = lambda a, b, c: jax.nn.dot_product_attention(
            a, b, c, implementation="hipc", return_residual=True, **options)
        def ref(a, b, c):
          out, residual = jax.nn.dot_product_attention(
              a.astype(jnp.float32), b.astype(jnp.float32), c.astype(jnp.float32),
              implementation="xla", return_residual=True, **options)
          return out, residual.astype(a.dtype)
        actual, expected = jax.jit(fn)(q, k, v), ref(q, k, v)
        tol = .03 if dtype == jnp.bfloat16 else .005
        for a, e in zip(actual, expected):
          np.testing.assert_allclose(a.astype(jnp.float32), e.astype(jnp.float32), rtol=tol, atol=tol)
        grad = lambda f: jax.jit(jax.grad(
            lambda a, b, c: f(a, b, c)[0].astype(jnp.float32).sum(), argnums=(0, 1, 2)))(q, k, v)
        actual_grad, expected_grad = grad(fn), grad(ref)
        for a, e in zip(actual_grad, expected_grad):
          np.testing.assert_allclose(a.astype(jnp.float32), e.astype(jnp.float32), rtol=tol, atol=tol)
        # 填充位置的梯度必须严格为零；空查询批次的 Q/K/V 梯度也必须为零。
        if ql is not None:
          np.testing.assert_array_equal(actual[0][0], 0)
          for a in actual_grad:
            np.testing.assert_array_equal(a[0], 0)
          for i, (nq, nk) in enumerate(zip(np.asarray(ql), np.asarray(kl))):
            np.testing.assert_array_equal(actual_grad[0][i, nq:], 0)
            np.testing.assert_array_equal(actual_grad[1][i, nk:], 0)
            np.testing.assert_array_equal(actual_grad[2][i, nk:], 0)

  def test_advertised_head_dimensions_forward_backward(self):
    for dtype in (jnp.float16, jnp.bfloat16):
      for dim in (32, 64, 96, 128, 160, 192, 224, 256):
        q, k, v = _inputs(b=1, sq=17, sk=33, h=2, hk=1, d=dim, dtype=dtype)
        for padded in (False, True):
          with self.subTest(dtype=str(dtype), dim=dim, padded=padded):
            options = (dict(query_seq_lengths=jnp.array([13], jnp.int32),
                            key_value_seq_lengths=jnp.array([23], jnp.int32))
                       if padded else {})
            def loss(a, b, c):
              out = jax.nn.dot_product_attention(a, b, c, implementation="hipc", **options)
              return out.astype(jnp.float32).sum(), out
            def reference(a, b, c):
              out = jax.nn.dot_product_attention(
                  a.astype(jnp.float32), b.astype(jnp.float32), c.astype(jnp.float32),
                  implementation="xla", **options)
              return out.sum(), out
            (_, actual), actual_grads = jax.jit(jax.value_and_grad(
                loss, argnums=(0, 1, 2), has_aux=True))(q, k, v)
            (_, expected), expected_grads = jax.jit(jax.value_and_grad(
                reference, argnums=(0, 1, 2), has_aux=True))(q, k, v)
            tol = .03 if dtype == jnp.bfloat16 else .005
            for a, e in zip((actual,) + actual_grads, (expected,) + expected_grads):
              self.assertTrue(np.isfinite(np.asarray(a)).all())
              np.testing.assert_allclose(a.astype(jnp.float32), e.astype(jnp.float32),
                                         rtol=tol, atol=tol)
            if padded:
              np.testing.assert_array_equal(actual[:, 13:], 0)
              np.testing.assert_array_equal(actual_grads[0][:, 13:], 0)
              np.testing.assert_array_equal(actual_grads[1][:, 23:], 0)
              np.testing.assert_array_equal(actual_grads[2][:, 23:], 0)

  def test_all_empty_query_batches_real_kernel(self):
    q, k, v = (jnp.full(x.shape, jnp.nan, x.dtype) for x in _inputs(d=128))
    ql, kl = jnp.zeros((3,), jnp.int32), jnp.array([1, 3, 7], jnp.int32)
    fn = lambda a, b, c: jax.nn.dot_product_attention(
        a, b, c, query_seq_lengths=ql, key_value_seq_lengths=kl,
        implementation="hipc", is_causal=True, local_window_size=(0, 0))
    np.testing.assert_array_equal(jax.jit(fn)(q, k, v), 0)
    for grad in jax.jit(jax.grad(lambda a, b, c: fn(a, b, c).astype(jnp.float32).sum(),
                                 argnums=(0, 1, 2)))(q, k, v):
      np.testing.assert_array_equal(grad, 0)

  def test_padded_vmap_backward_real_kernel(self):
    q, k, v = _inputs(sq=5, sk=7, d=128)
    qs = jnp.stack((q, q * .5))
    qls = jnp.array([[0, 2, 4], [1, 3, 5]], jnp.int32)
    kls = jnp.array([[1, 3, 6], [2, 4, 7]], jnp.int32)
    fn = lambda a, lq, lk: jax.nn.dot_product_attention(
        a, k, v, query_seq_lengths=lq, key_value_seq_lengths=lk,
        implementation="hipc")
    def ref(a, lq, lk):
      return jax.nn.dot_product_attention(
          a.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32),
          query_seq_lengths=lq, key_value_seq_lengths=lk, implementation="xla")
    actual = jax.jit(jax.vmap(fn))(qs, qls, kls)
    expected = jax.vmap(ref)(qs, qls, kls)
    np.testing.assert_allclose(actual, expected, rtol=.005, atol=.005)
    actual_grad = jax.jit(jax.grad(lambda a: jax.vmap(fn)(a, qls, kls).astype(jnp.float32).sum()))(qs)
    expected_grad = jax.grad(lambda a: jax.vmap(ref)(a, qls, kls).sum())(qs)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=.005, atol=.005)
    np.testing.assert_array_equal(actual_grad[0, 0], 0)

  def test_single_query_causal_standard_semantics(self):
    q, k, v = _inputs(b=1, sq=1, sk=7, d=128)
    for fn in (lambda: jax.nn.dot_product_attention(
                   q, k, v, implementation="hipc", is_causal=True),
               lambda: jax.jit(lambda a, b, c: jax.nn.dot_product_attention(
                   a, b, c, implementation="hipc", is_causal=True))(q, k, v)):
      with self.assertRaisesRegex(ValueError, "equal Q/K"):
        fn()
    actual = jax.jit(lambda a, b, c: jax.nn.dot_product_attention(
        a, b, c, implementation="hipc", is_causal=True))(q, k[:, :1], v[:, :1])
    expected = jnp.repeat(v[:, :1], q.shape[2] // v.shape[2], axis=2)
    np.testing.assert_array_equal(actual, expected)

  def test_runtime_length_validation(self):
    q, k, v = _inputs(d=128)
    fn = jax.jit(lambda lq, lk: jax.nn.dot_product_attention(q, k, v,
        query_seq_lengths=lq, key_value_seq_lengths=lk, implementation="hipc"))
    for lq, lk in (([-1, 2, 3], [1, 2, 3]), ([6, 2, 3], [1, 2, 3]),
                   ([1, 2, 3], [0, 2, 3]), ([1, 2, 3], [8, 2, 3])):
      with self.subTest(lq=lq, lk=lk), self.assertRaises(Exception):
        jax.block_until_ready(fn(jnp.array(lq, jnp.int32), jnp.array(lk, jnp.int32)))
    causal = jax.jit(lambda lq, lk: self.fa.fa_fwd_padded(q, k, v, lq, lk, causal=True))
    with self.assertRaises(Exception):
      jax.block_until_ready(causal(jnp.array([1, 2, 3], jnp.int32),
                                   jnp.array([2, 2, 3], jnp.int32)))

  def test_lowering_is_hipc_not_matmul(self):
    q, k, v = _inputs(d=128)
    ql, kl = jnp.array([0, 2, 4], jnp.int32), jnp.array([1, 3, 6], jnp.int32)
    fn = jax.jit(lambda a, b, c, lq, lk: jax.nn.dot_product_attention(
        a, b, c, query_seq_lengths=lq, key_value_seq_lengths=lk, implementation="hipc"))
    text = fn.lower(q, k, v, ql, kl).as_text()
    self.assertIn("fa_fwd_ffi" if self.fa._USE_FFI else "fa_fwd_v0", text)
    self.assertIn("fa_validate_lengths_ffi", text)
    self.assertNotIn("dot_general", text)
    first = fn(q, k, v, ql, kl)
    second = fn(q, k, v, jnp.array([1, 3, 5], jnp.int32), jnp.array([2, 4, 7], jnp.int32))
    self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
  absltest.main()
