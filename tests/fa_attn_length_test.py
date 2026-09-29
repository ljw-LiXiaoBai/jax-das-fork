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

"""有效 KV 前缀长度的原生集成测试；数学参考仅用于对照，不作为回退。"""
import functools
import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
import numpy as np

config.parse_flags_with_absl()

def _load_fa():
  required = os.environ.get("FA_REQUIRE_TESTS", "0") == "1"
  def unavailable(reason):
    if required:
      raise RuntimeError(reason)
    raise absltest.SkipTest(reason)
  if not jtu.is_device_rocm():
    unavailable("需要 ROCm GPU")
  path = os.environ.get("FA_LIB_PATH", "/opt/libflash_attention.so")
  if not os.path.isfile(path):
    if "FA_LIB_PATH" in os.environ:
      raise FileNotFoundError(f"FA_LIB_PATH 指向不存在的文件: {path}")
    unavailable(f"未部署算子库: {path}")
  from jax._src.cudnn import fa_attention
  try:
    fa_attention._register()
  except ModuleNotFoundError as e:
    if e.name != "fa_attention_plugin":
      raise
    unavailable(str(e))
  return fa_attention


def prefix_reference(q, k, v, length, layout, scale):
  """独立 fp32 参考：仅截取 KV 前缀，保留全部查询行。"""
  q, k, v = (np.asarray(x, np.float32) for x in (q, k, v))
  if layout == 0:
    q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
  groups = q.shape[2] // k.shape[2]
  k = np.repeat(k[:, :length], groups, axis=2)
  v = np.repeat(v[:, :length], groups, axis=2)
  logits = np.einsum("bqhd,bkhd->bhqk", q, k) * np.float32(scale)
  maximum = logits.max(-1, keepdims=True)
  weights = np.exp(logits - maximum)
  denominator = weights.sum(-1, keepdims=True)
  o = np.einsum("bhqk,bkhd->bqhd", weights / denominator, v)
  lse = (maximum + np.log(denominator))[..., 0]
  return (o if layout else o.transpose(0, 2, 1, 3)), lse


class AttnLengthTest(jtu.JaxTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.fa = _load_fa()

  def inputs(self, b=2, sq=5, sk=7, h=2, hk=2, dtype=jnp.float16, layout=1):
    rng = np.random.default_rng(20 + sq)
    arrays = [jnp.asarray(rng.normal(0, .4, shape), dtype)
              for shape in ((b, sq, h, 128), (b, sk, hk, 128), (b, sk, hk, 128))]
    if layout == 0:
      arrays = [x.transpose(0, 2, 1, 3) for x in arrays]
    return arrays

  def test_numeric_eager_jit_matrix(self):
    # (batch, q_length, k_length, heads, kv_heads, valid_k)
    cases = [(1, 5, 7, 2, 2, 3), (2, 128, 128, 2, 2, 128),
             (2, 96, 160, 4, 2, 37), (2, 128, 256, 4, 1, 160),
             (1, 1, 128, 2, 1, 1), (2, 33, 256, 2, 2, 3),
             (2, 257, 193, 4, 2, 192), (1, 17, 65, 2, 2, 65)]
    for dtype in (jnp.float16, jnp.bfloat16):
      for layout in (0, 1):
        for case in cases:
          b, sq, sk, h, hk, length = case
          with self.subTest(dtype=str(dtype), layout=layout, case=case):
            q, k, v = self.inputs(b, sq, sk, h, hk, dtype, layout)
            before = [np.asarray(x).copy() for x in (q, k, v)]
            scale = .07
            fn = functools.partial(self.fa.fa_fwd_attn_length, valid_k=length,
                                   softmax_scale=scale, layout=layout)
            eager = fn(q, k, v)
            compiled = jax.jit(fn)(q, k, v)
            expected = prefix_reference(q, k, v, length, layout, scale)
            tol = 1e-2 if dtype == jnp.bfloat16 else 2e-3
            self.assertEqual(eager[0].shape, q.shape)
            self.assertEqual(eager[1].shape, (b, h, sq))
            self.assertEqual(eager[0].dtype, q.dtype)
            self.assertEqual(eager[1].dtype, jnp.float32)
            for actual in (eager, compiled):
              np.testing.assert_allclose(actual[0], expected[0], rtol=tol, atol=tol)
              np.testing.assert_allclose(actual[1], expected[1], rtol=1e-4, atol=1e-4)
              self.assertTrue(np.isfinite(np.asarray(actual[0])).all())
              self.assertTrue(np.isfinite(np.asarray(actual[1])).all())
            for a, b_ in zip(eager, compiled):
              np.testing.assert_array_equal(a, b_)
            for a, b_ in zip(before, (q, k, v)):
              np.testing.assert_array_equal(a, b_)

  def test_masked_kv_changes_do_not_affect_output(self):
    q, k, v = self.inputs(sk=128)
    fn = jax.jit(lambda q, k, v: self.fa.fa_fwd_attn_length(q, k, v, 3))
    base = fn(q, k, v)
    changed = fn(q, k.at[:, 3:].set(50), v.at[:, 3:].set(-50))
    for a, b in zip(base, changed):
      np.testing.assert_array_equal(a, b)

  def test_length_one_is_first_value_for_all_queries(self):
    q, k, v = self.inputs()
    o, _ = self.fa.fa_fwd_attn_length(q, k, v, 1)
    np.testing.assert_array_equal(o, np.broadcast_to(np.asarray(v[:, :1]), q.shape))

  def test_static_length_and_numpy_integer(self):
    q, k, v = self.inputs()
    fn = jax.jit(self.fa.fa_fwd_attn_length, static_argnames=("valid_k",))
    out3, lse3 = fn(q, k, v, valid_k=3)
    out5, _ = fn(q, k, v, valid_k=np.int32(5))
    self.assertFalse(np.array_equal(out3, out5))
    expected = prefix_reference(q, k, v, 3, 1, 128 ** -.5)
    np.testing.assert_allclose(out3, expected[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(lse3, expected[1], rtol=1e-4, atol=1e-4)

  def test_dynamic_scalar_length_reuses_executable(self):
    q, k, v = self.inputs()
    fn = jax.jit(self.fa.fa_fwd_attn_length).lower(q, k, v, jnp.int32(3)).compile()
    for length in (3, 5, 7):
      actual = fn(q, k, v, jnp.int32(length))
      expected = prefix_reference(q, k, v, length, 1, 128 ** -.5)
      np.testing.assert_allclose(actual[0], expected[0], rtol=2e-3, atol=2e-3)
      np.testing.assert_allclose(actual[1], expected[1], rtol=1e-4, atol=1e-4)

  def test_invalid_lengths(self):
    q, k, v = self.inputs()
    for length in (0, -1, 8):
      with self.subTest(length=length), self.assertRaisesRegex(ValueError, "valid_k"):
        self.fa.fa_fwd_attn_length(q, k, v, length)
    for length in (True, 3.0, jnp.array([3, 4], jnp.float32)):
      with self.subTest(length=str(length)), self.assertRaisesRegex(TypeError, "valid_k"):
        self.fa.fa_fwd_attn_length(q, k, v, length)
    for length in ([3], np.ones((2, 2), np.int32), jnp.array([3], jnp.int32)):
      with self.subTest(length=str(length)), self.assertRaisesRegex(ValueError, "valid_k"):
        self.fa.fa_fwd_attn_length(q, k, v, length)

  def test_invalid_shape_dtype_layout(self):
    q, k, v = self.inputs()
    with self.assertRaisesRegex(ValueError, "layout"):
      self.fa.fa_fwd_attn_length(q, k, v, 3, layout=2)
    with self.assertRaisesRegex(ValueError, "4-D"):
      self.fa.fa_fwd_attn_length(q[0], k, v, 3)
    with self.assertRaisesRegex(ValueError, "128"):
      self.fa.fa_fwd_attn_length(q[..., :64], k[..., :64], v[..., :64], 3)
    with self.assertRaisesRegex(TypeError, "fp16"):
      self.fa.fa_fwd_attn_length(q.astype(jnp.float32), k, v, 3)
    with self.assertRaisesRegex(ValueError, "不匹配"):
      self.fa.fa_fwd_attn_length(q, k, v[:, :6], 3)
    with self.assertRaisesRegex(ValueError, "不匹配"):
      self.fa.fa_fwd_attn_length(q, k[:1], v[:1], 3)
    for scale in (0, -1, float("nan"), float("inf")):
      with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "有限正数"):
        self.fa.fa_fwd_attn_length(q, k, v, 3, softmax_scale=scale)

  def test_grad_length_one_exact_value_path(self):
    q, k, v = self.inputs()
    do = jnp.asarray(np.random.default_rng(51).normal(size=q.shape), q.dtype)
    loss = lambda a, b, c: (self.fa.fa_fwd_attn_length(a, b, c, 1)[0].astype(jnp.float32)
                            * do.astype(jnp.float32)).sum()
    grad = jax.grad(loss, argnums=(0, 1, 2))
    expected_dv = np.zeros(v.shape, np.float32)
    expected_dv[:, 0] = np.asarray(do, np.float32).sum(axis=1)
    for fn in (grad, jax.jit(grad)):
      dq, dk, dv = fn(q, k, v)
      np.testing.assert_allclose(dq, 0, rtol=0, atol=3e-3)
      np.testing.assert_allclose(dk, 0, rtol=0, atol=3e-3)
      np.testing.assert_allclose(dv, expected_dv, rtol=3e-3, atol=3e-3)
      np.testing.assert_array_equal(dk[:, 1:], np.zeros_like(np.asarray(dk[:, 1:])))
      np.testing.assert_array_equal(dv[:, 1:], np.zeros_like(np.asarray(dv[:, 1:])))

  def test_full_mask_guard_unchanged(self):
    q, k, v = self.inputs()
    with self.assertRaisesRegex(NotImplementedError, "full attn_mask"):
      self.fa.fa_fwd_attn_mask(q, k, v, jnp.ones((2, 2, 5, 7), bool))
    with self.assertRaisesRegex(NotImplementedError, "padding_mask"):
      self.fa.fa_fwd_padding_mask(q, k, v, jnp.array([5, 5], jnp.int32))

  def test_hlo_calls_length_target(self):
    q, k, v = self.inputs()
    text = jax.jit(lambda a, b, c: self.fa.fa_fwd_attn_length(a, b, c, 3)).lower(q, k, v).as_text()
    self.assertIn("fa_attn_length_ffi", text)
    self.assertNotIn("dot_general", text)

  def test_vmap_sequential(self):
    q, k, v = self.inputs()
    qs, ks, vs = (jnp.stack([x, x * .5]) for x in (q, k, v))
    fn = lambda a, b, c: self.fa.fa_fwd_attn_length(a, b, c, 3)
    actual = jax.jit(jax.vmap(fn))(qs, ks, vs)
    for i in range(2):
      expected = prefix_reference(qs[i], ks[i], vs[i], 3, 1, 128 ** -.5)
      np.testing.assert_allclose(actual[0][i], expected[0], rtol=2e-3, atol=2e-3)
      np.testing.assert_allclose(actual[1][i], expected[1], rtol=1e-4, atol=1e-4)

  def test_ffi_rejects_malformed_output(self):
    q, k, v = self.inputs()
    self.fa._register_attn_length()
    wrong = (jax.ShapeDtypeStruct((2, 7, 2, 128), q.dtype),
             jax.ShapeDtypeStruct((2, 2, 5), jnp.float32),
             jax.ShapeDtypeStruct((72,), jnp.int32))
    fn = jax.ffi.ffi_call("fa_attn_length_ffi", wrong, vmap_method="sequential")
    with self.assertRaisesRegex(Exception, "o shape"):
      output = fn(q, k, v, jnp.array([3], jnp.int32),
                  scale=np.float32(.07), layout=np.int32(1), valid_k=np.int32(3))
      jax.block_until_ready(output)

  def test_unsupported_options_rejected(self):
    q, k, v = self.inputs()
    for kw in ({"causal": True}, {"window_size": (3, 0)}, {"dropout_p": .1}):
      with self.subTest(kw=kw), self.assertRaises(TypeError):
        self.fa.fa_fwd_attn_length(q, k, v, 3, **kw)

  def test_existing_dense_baseline(self):
    q, k, v = self.inputs(b=1, sq=128, sk=128)
    expected = self.fa.fa_fwd(q, k, v)
    actual = self.fa.fa_fwd_attn_length(q, k, v, 128)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
