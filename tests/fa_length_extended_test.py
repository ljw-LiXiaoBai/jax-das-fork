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

"""长度注意力验收，参考值由独立 fp32 运算及对 fp32 输入求导获得。

填充位置梯度必须严格为零；改变长度值时复用同一已编译对象。
FA_REQUIRE_TESTS=1 时，缺少运行前置条件须失败而非跳过。
"""
import functools
import os

from absl.testing import absltest, parameterized
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
    unavailable("ROCm GPU required for extended length attention tests")
  path = os.environ.get("FA_LIB_PATH", "/opt/libflash_attention.so")
  if not os.path.isfile(path):
    if "FA_LIB_PATH" in os.environ:
      raise FileNotFoundError(f"FA_LIB_PATH points to a missing file: {path}")
    unavailable(f"HIPC library unavailable: {path}")
  from jax._src.cudnn import fa_attention
  try:
    fa_attention._register()
  except ModuleNotFoundError as error:
    if error.name != "fa_attention_plugin":
      raise
    unavailable(str(error))
  return fa_attention


def length_reference(q, k, v, lengths, *, layout=1, scale=None,
                     padding=False):
  """Independent all-fp32 masked attention, including GQA and natural-log LSE."""
  q, k, v = (jnp.asarray(x, jnp.float32) for x in (q, k, v))
  if layout == 0:
    q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
  batch, sq, heads, dim = q.shape
  sk, kv_heads = k.shape[1:3]
  lengths = jnp.broadcast_to(jnp.asarray(lengths, jnp.int32), (batch,))
  groups = heads // kv_heads
  k, v = (jnp.repeat(x, groups, axis=2) for x in (k, v))
  scale = jnp.asarray(dim ** -.5 if scale is None else scale, jnp.float32)
  # Explicit fp32 multiply/reduce also avoids this JAX fork's CPU contraction
  # transpose bug. Keep this small-fixture oracle independent of fused kernels.
  logits = jnp.sum(q[:, :, None, :, :] * k[:, None, :, :, :], axis=-1)
  logits = logits.transpose(0, 3, 1, 2) * scale
  valid_k = jnp.arange(sk)[None, :] < lengths[:, None]
  logits = jnp.where(valid_k[:, None, None, :], logits, -jnp.inf)
  maximum = jnp.max(logits, axis=-1, keepdims=True)
  weights = jnp.exp(logits - maximum)
  denominator = jnp.sum(weights, axis=-1, keepdims=True)
  probabilities = (weights / denominator).transpose(0, 2, 3, 1)
  out = jnp.sum(probabilities[..., None] * v[:, None, :, :, :], axis=2)
  lse = (maximum + jnp.log(denominator))[..., 0]
  if padding:
    valid_q = jnp.arange(sq)[None, :] < lengths[:, None]
    out = jnp.where(valid_q[..., None, None], out, 0)
    lse = jnp.where(valid_q[:, None, :], lse, -jnp.inf)
  if layout == 0:
    out = out.transpose(0, 2, 1, 3)
  return out, lse


def reference_vjp(q, k, v, do, lengths, **options):
  # Differentiate fp32 inputs, not low-precision casts of reference gradients.
  fn = lambda a, b, c: length_reference(a, b, c, lengths, **options)[0]
  _, pullback = jax.vjp(fn, *(x.astype(jnp.float32) for x in (q, k, v)))
  return pullback(do.astype(jnp.float32))


def output_vjp(call, q, k, v, do):
  _, pullback = jax.vjp(lambda a, b, c: call(a, b, c)[0], q, k, v)
  return pullback(do)


def _inputs(dtype=jnp.float16, layout=1, gqa=True, sq=33, sk=97):
  rng = np.random.default_rng(917 + sq + sk)
  heads, kv_heads = (4, 2) if gqa else (2, 2)
  shapes = ((2, sq, heads, 128), (2, sk, kv_heads, 128),
            (2, sk, kv_heads, 128), (2, sq, heads, 128))
  arrays = tuple(jnp.asarray(rng.normal(0, .4, shape), dtype)
                 for shape in shapes)
  return arrays if layout else tuple(x.transpose(0, 2, 1, 3) for x in arrays)


def _fp32(x):
  return np.asarray(x, dtype=np.float32)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class LengthExtendedTest(jtu.JaxTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.fa = _load_fa()

  def assert_forward(self, actual, expected, q, layout, padding=False):
    out, lse = actual
    batch, heads, sq = (q.shape[0], q.shape[2], q.shape[1]) if layout else q.shape[:3]
    self.assertEqual(out.shape, q.shape)
    self.assertEqual(out.dtype, q.dtype)
    self.assertEqual(lse.shape, (batch, heads, sq))
    self.assertEqual(lse.dtype, jnp.float32)
    tol = 1e-2 if q.dtype == jnp.bfloat16 else 2e-3
    np.testing.assert_allclose(_fp32(out), _fp32(expected[0]), rtol=tol, atol=tol)
    np.testing.assert_allclose(lse, expected[1], rtol=1e-4, atol=1e-4)
    self.assertTrue(np.isfinite(_fp32(out)).all())
    if not padding:
      self.assertTrue(np.isfinite(np.asarray(lse)).all())

  def assert_gradients(self, actual, expected, inputs, lengths, layout,
                       padding=False):
    tol = 1.5e-2 if inputs[0].dtype == jnp.bfloat16 else 3e-3
    lens = np.broadcast_to(np.asarray(lengths), (inputs[0].shape[0],))
    for index, (got, ref, original) in enumerate(zip(actual, expected, inputs)):
      self.assertEqual(got.shape, original.shape)
      self.assertEqual(got.dtype, original.dtype)
      self.assertTrue(np.isfinite(_fp32(got)).all())
      np.testing.assert_allclose(_fp32(got), _fp32(ref), rtol=tol, atol=tol)
      if index > 0 or padding:
        canonical = _fp32(got) if layout else _fp32(got).transpose(0, 2, 1, 3)
        for batch, length in enumerate(lens):
          # This is an exact-zero contract, independent of numeric tolerances.
          tail = canonical[batch, int(length):]
          np.testing.assert_array_equal(tail, np.zeros_like(tail))

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16), layout=(0, 1),
                         gqa=(False, True), kind=("static", "scalar", "vector"))
  def test_cross_attention_forward_and_arbitrary_vjp(self, dtype, layout, gqa, kind):
    q, k, v, do = _inputs(dtype, layout, gqa)
    lengths = {"static": 37, "scalar": jnp.int32(37),
               "vector": jnp.array([1, 61], jnp.int32)}[kind]
    call = functools.partial(self.fa.fa_fwd_attn_length, valid_k=lengths,
                             softmax_scale=.07, layout=layout)
    ref = length_reference(q, k, v, lengths, layout=layout, scale=.07)
    ref_grad = reference_vjp(q, k, v, do, lengths, layout=layout, scale=.07)
    before = tuple(np.asarray(x).copy() for x in (q, k, v))
    grad = functools.partial(output_vjp, call)
    for compiled in (False, True):
      with self.subTest(compiled=compiled):
        actual = jax.jit(call)(q, k, v) if compiled else call(q, k, v)
        got_grad = jax.jit(grad)(q, k, v, do) if compiled else grad(q, k, v, do)
        self.assert_forward(actual, ref, q, layout)
        self.assert_gradients(got_grad, ref_grad, (q, k, v), lengths, layout)
    for original, current in zip(before, (q, k, v)):
      np.testing.assert_array_equal(original, current)

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16), layout=(0, 1),
                         kind=("scalar", "vector"))
  def test_runtime_lengths_reuse_same_forward_backward_executable(self, dtype, layout, kind):
    q, k, v, do = _inputs(dtype, layout)

    def run(a, b, c, cotangent, lengths):
      call = functools.partial(self.fa.fa_fwd_attn_length, valid_k=lengths,
                               layout=layout)
      return call(a, b, c), output_vjp(call, a, b, c, cotangent)

    values = (1, 97, 37) if kind == "scalar" else ([1, 61], [97, 2], [37, 97])
    first_length = jnp.asarray(values[0], jnp.int32)
    # Calling this AOT object with changed *values* cannot silently retrace.
    executable = jax.jit(run).lower(q, k, v, do, first_length).compile()
    previous = None
    for value in values:
      lengths = jnp.asarray(value, jnp.int32)
      actual, gradients = executable(q, k, v, do, lengths)
      ref = length_reference(q, k, v, lengths, layout=layout)
      ref_grad = reference_vjp(q, k, v, do, lengths, layout=layout)
      self.assert_forward(actual, ref, q, layout)
      self.assert_gradients(gradients, ref_grad, (q, k, v), lengths, layout)
      if previous is not None:
        self.assertFalse(np.array_equal(previous, actual[0]))
      previous = np.asarray(actual[0])

  @parameterized.product(layout=(0, 1), kind=("static", "scalar", "vector", "padding"))
  def test_lse_is_stop_gradient_even_with_nonzero_cotangent(self, layout, kind):
    q, k, v, do = _inputs(layout=layout, sq=33, sk=33 if kind == "padding" else 97)
    lengths = (17 if kind == "static" else jnp.int32(17)
               if kind == "scalar" else jnp.array([3, 17], jnp.int32))
    api = self.fa.fa_fwd_padding_length if kind == "padding" else self.fa.fa_fwd_attn_length
    call = lambda a, b, c: api(a, b, c, lengths, layout=layout)

    def gradients(a, b, c, cotangent):
      (out, lse), pullback = jax.vjp(call, a, b, c)
      dlse = jnp.arange(lse.size, dtype=jnp.float32).reshape(lse.shape) * .01 + .3
      return (pullback((cotangent, dlse)),
              pullback((jnp.zeros_like(out), dlse)))

    actual, lse_only = jax.jit(gradients)(q, k, v, do)
    expected = reference_vjp(q, k, v, do, lengths, layout=layout,
                             padding=kind == "padding")
    self.assert_gradients(actual, expected, (q, k, v), lengths, layout,
                          padding=kind == "padding")
    for gradient in lse_only:
      np.testing.assert_array_equal(_fp32(gradient), np.zeros(gradient.shape, np.float32))

  @parameterized.parameters(0, 1)
  def test_per_batch_kv_tail_changes_leave_all_query_rows_unchanged(self, layout):
    q, k, v, do = _inputs(layout=layout)
    lengths = jnp.array([1, 61], jnp.int32)
    valid = jnp.arange(97)[None, :] < lengths[:, None]
    valid = valid[..., None, None] if layout else valid[:, None, :, None]
    changed_k, changed_v = jnp.where(valid, k, 20), jnp.where(valid, v, -20)
    call = functools.partial(self.fa.fa_fwd_attn_length, valid_k=lengths, layout=layout)
    base = jax.jit(call)(q, k, v)
    changed = jax.jit(call)(q, changed_k, changed_v)
    for original, modified in zip(base, changed):
      np.testing.assert_array_equal(original, modified)
    # A K length of one does not mean a Q length of one.
    value = np.asarray(v[0, :1] if layout else v[0, :, :1])
    value = np.repeat(value, 2, axis=1 if layout else 0)  # GQA head groups.
    np.testing.assert_array_equal(base[0][0], np.broadcast_to(value, q[0].shape))
    changed_grad = jax.jit(functools.partial(output_vjp, call))(q, changed_k, changed_v, do)
    ref_grad = reference_vjp(q, k, v, do, lengths, layout=layout)
    self.assert_gradients(changed_grad, ref_grad, (q, k, v), lengths, layout)

  @parameterized.product(layout=(0, 1), kind=("static", "scalar", "vector"))
  def test_vmap_per_instance_lengths_and_broadcast_kv_reverse(self, layout, kind):
    q, k, v, do = _inputs(layout=layout, sq=5, sk=17)
    qs, dos = jnp.stack((q, q * .5)), jnp.stack((do, -do * .75))
    lengths = (jnp.array([3, 11], jnp.int32) if kind != "vector"
               else jnp.array([[1, 7], [13, 3]], jnp.int32))
    if kind == "static":
      single = lambda a, b, c: self.fa.fa_fwd_attn_length(a, b, c, 7, layout=layout)
      mapped = jax.vmap(single, in_axes=(0, None, None))
      actual = jax.jit(mapped)(qs, k, v)
      mapped_grad = jax.jit(jax.vmap(functools.partial(output_vjp, single),
                                   in_axes=(0, None, None, 0)))(qs, k, v, dos)
    else:
      single = lambda a, b, c, length: self.fa.fa_fwd_attn_length(a, b, c, length, layout=layout)
      mapped = lambda a, b, c: jax.vmap(single, in_axes=(0, None, None, 0))(a, b, c, lengths)
      actual = jax.jit(mapped)(qs, k, v)
      per_grad = lambda a, b, c, cotangent, length: output_vjp(
          lambda x, y, z: single(x, y, z, length), a, b, c, cotangent)
      mapped_grad = jax.jit(jax.vmap(per_grad, in_axes=(0, None, None, 0, 0)))(
          qs, k, v, dos, lengths)
    references = []
    for index in range(2):
      length = 7 if kind == "static" else lengths[index]
      ref = length_reference(qs[index], k, v, length, layout=layout)
      ref_grad = reference_vjp(qs[index], k, v, dos[index], length, layout=layout)
      self.assert_forward(tuple(x[index] for x in actual), ref, q, layout)
      self.assert_gradients(tuple(x[index] for x in mapped_grad), ref_grad,
                            (q, k, v), length, layout)
      references.append(ref_grad)
    # grad(vmap) must sum, rather than retain an extra map axis on shared K/V.
    total_loss = lambda a, b, c: jnp.sum(mapped(a, b, c)[0].astype(jnp.float32) * dos.astype(jnp.float32))
    total_grad = jax.jit(jax.grad(total_loss, argnums=(0, 1, 2)))(qs, k, v)
    expected = (jnp.stack((references[0][0], references[1][0])),
                references[0][1] + references[1][1],
                references[0][2] + references[1][2])
    for got, ref in zip(total_grad, expected):
      self.assertEqual(got.shape, ref.shape)
      np.testing.assert_allclose(_fp32(got), _fp32(ref), rtol=3e-3, atol=3e-3)

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16), layout=(0, 1),
                         gqa=(False, True))
  def test_padding_forward_backward_and_changed_lengths(self, dtype, layout, gqa):
    q, k, v, do = _inputs(dtype, layout, gqa, sq=33, sk=33)

    def run(a, b, c, cotangent, lengths):
      call = functools.partial(self.fa.fa_fwd_padding_length, lengths=lengths,
                               softmax_scale=.07, layout=layout)
      return call(a, b, c), output_vjp(call, a, b, c, cotangent)

    executable = jax.jit(run).lower(q, k, v, do, jnp.array([1, 17], jnp.int32)).compile()
    for values in ([1, 17], [33, 3]):
      lengths = jnp.array(values, jnp.int32)
      ref = length_reference(q, k, v, lengths, layout=layout, scale=.07, padding=True)
      ref_grad = reference_vjp(q, k, v, do, lengths, layout=layout, scale=.07, padding=True)
      for execute in (run, executable):
        actual, gradients = execute(q, k, v, do, lengths)
        self.assert_forward(actual, ref, q, layout, padding=True)
        self.assert_gradients(gradients, ref_grad, (q, k, v), lengths, layout, padding=True)
        output = _fp32(actual[0]) if layout else _fp32(actual[0]).transpose(0, 2, 1, 3)
        for batch, length in enumerate(values):
          np.testing.assert_array_equal(output[batch, length:], np.zeros_like(output[batch, length:]))
          self.assertTrue(np.isneginf(np.asarray(actual[1])[batch, :, length:]).all())

  @parameterized.parameters(0, 1)
  def test_padding_vmap_with_per_instance_lengths(self, layout):
    q, k, v, do = _inputs(layout=layout, sq=17, sk=17)
    qs, ks, vs, dos = (jnp.stack((x, x * .5)) for x in (q, k, v, do))
    lengths = jnp.array([[1, 13], [7, 17]], jnp.int32)

    def run(a, b, c, cotangent, lens):
      call = lambda x, y, z: self.fa.fa_fwd_padding_length(x, y, z, lens, layout=layout)
      return call(a, b, c), output_vjp(call, a, b, c, cotangent)

    actual, gradients = jax.jit(jax.vmap(run))(qs, ks, vs, dos, lengths)
    for index in range(2):
      ref = length_reference(qs[index], ks[index], vs[index], lengths[index],
                              layout=layout, padding=True)
      ref_grad = reference_vjp(qs[index], ks[index], vs[index], dos[index], lengths[index],
                               layout=layout, padding=True)
      self.assert_forward(tuple(x[index] for x in actual), ref, q, layout, padding=True)
      self.assert_gradients(tuple(x[index] for x in gradients), ref_grad, (q, k, v),
                            lengths[index], layout, padding=True)

  @parameterized.parameters("scalar", "vector", "padding")
  def test_invalid_device_lengths_fail_readably_and_executable_recovers(self, kind):
    q, k, v, _ = _inputs(sq=5, sk=5 if kind == "padding" else 7)
    capacity = k.shape[1]
    api = self.fa.fa_fwd_padding_length if kind == "padding" else self.fa.fa_fwd_attn_length
    call = lambda a, b, c, lens: api(a, b, c, lens)
    valid = jnp.asarray(3 if kind == "scalar" else [3, 2], jnp.int32)
    executable = jax.jit(call).lower(q, k, v, valid).compile()
    for bad in (0, -1, capacity + 1):
      lens = jnp.asarray(bad if kind == "scalar" else [2, bad], jnp.int32)
      for execute in (call, executable):
        with self.subTest(bad=bad, compiled=execute is executable):
          with self.assertRaisesRegex(Exception, "(?i)(length|valid_k|capacity|positive|长度)"):
            jax.block_until_ready(execute(q, k, v, lens))
      # 每次非法长度报错后，用合法长度复用同一可执行对象，确认执行流仍可用。
      actual = jax.block_until_ready(executable(q, k, v, valid))
      expected = length_reference(q, k, v, valid, padding=kind == "padding")
      self.assert_forward(actual, expected, q, 1, padding=kind == "padding")

  @parameterized.parameters(False, True)
  def test_bad_length_dtype_rank_and_batch_shape(self, padding):
    q, k, v, _ = _inputs(sq=5, sk=5)
    api = self.fa.fa_fwd_padding_length if padding else self.fa.fa_fwd_attn_length
    invalid = (jnp.array([1., 2.], jnp.float32), jnp.array([1, 2], jnp.int16),
               jnp.array([1, 2], jnp.uint32), jnp.array([True, False]),
               jnp.ones((2, 1), jnp.int32), jnp.ones((1,), jnp.int32),
               jnp.ones((3,), jnp.int32), jnp.empty((0,), jnp.int32))
    if padding:
      invalid += (jnp.int32(3),)
    else:
      invalid += (True, 3., jnp.float32(3))
    for lens in invalid:
      for call in (api, jax.jit(api)):
        with self.subTest(length=str(lens), compiled=call is not api):
          with self.assertRaisesRegex((TypeError, ValueError), "(int32|valid_k|length|bool)"):
            jax.block_until_ready(call(q, k, v, lens))

  def test_padding_rejects_different_physical_query_key_lengths(self):
    q, k, v, _ = _inputs(sq=5, sk=7)
    for call in (self.fa.fa_fwd_padding_length, jax.jit(self.fa.fa_fwd_padding_length)):
      with self.assertRaisesRegex(ValueError, "(Q/K|physical|物理)"):
        call(q, k, v, jnp.array([3, 2], jnp.int32))

  def test_static_and_dynamic_lowerings_use_native_forward_and_backward(self):
    q, k, v, do = _inputs(sq=5, sk=7)
    static = lambda a, b, c: self.fa.fa_fwd_attn_length(a, b, c, 3)
    text = jax.jit(static).lower(q, k, v).as_text()
    self.assertIn("fa_attn_length_ffi", text)
    self.assertNotIn("dot_general", text)
    text = jax.jit(functools.partial(output_vjp, static)).lower(q, k, v, do).as_text()
    self.assertIn("fa_attn_length_ffi", text)
    self.assertRegex(text, "fa_bwd_(ffi|v0)")
    # Static reverse must consume the native length forward O/LSE, not rerun
    # another forward kernel and silently use different residuals.
    self.assertNotRegex(text, "fa_fwd_(ffi|v0)")
    self.assertNotIn("dot_general", text)

    def dynamic(a, b, c, cotangent, lengths):
      call = lambda x, y, z: self.fa.fa_fwd_attn_length(x, y, z, lengths)
      return call(a, b, c), output_vjp(call, a, b, c, cotangent)

    for lengths in (jnp.int32(3), jnp.array([3, 5], jnp.int32)):
      text = jax.jit(dynamic).lower(q, k, v, do, lengths).as_text()
      self.assertIn("fa_validate_lengths_ffi", text)
      self.assertRegex(text, "fa_fwd_(ffi|v0)")
      self.assertRegex(text, "fa_bwd_(ffi|v0)")
      self.assertNotIn("dot_general", text)


if __name__ == "__main__":
  absltest.main()
