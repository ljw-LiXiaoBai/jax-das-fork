# Copyright 2026 The JAX Authors.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""批处理变换与分片测试。

形状追踪和模拟内核编译不验证厂商数值。
原生验收由 FA_RUN_BATCHING_GPU_TESTS=1 启用，多卡用例不足两卡即失败。
"""
import functools
import os
from unittest import mock

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()


def _spec(shape, dtype=jnp.float16):
  return jax.ShapeDtypeStruct(shape, dtype)


class BatchingTracingTest(parameterized.TestCase):
  """No kernel execution: test all argument/result ranks and AD composition."""

  def setUp(self):
    super().setUp()
    from jax._src.cudnn import fa_attention
    self.fa = fa_attention
    self.enterContext(mock.patch.object(self.fa, "_register"))
    self.enterContext(mock.patch.object(self.fa, "_register_attn_length"))

  @parameterized.product(ffi=(False, True), layout=(0, 1))
  def test_dense_mixed_nonleading_nested_and_grad(self, ffi, layout):
    qshape = (2, 96, 4, 64) if layout else (2, 4, 96, 64)
    kshape = (2, 96, 2, 64) if layout else (2, 2, 96, 64)
    q, k = _spec(qshape), _spec(kshape)
    call = functools.partial(self.fa.fa_fwd_custom, layout=layout)
    with mock.patch.object(self.fa, "_USE_FFI", ffi):
      # An extra map dimension is not the kernel's existing batch dimension.
      qs = _spec((3,) + qshape)
      out, lse = jax.eval_shape(jax.vmap(call, in_axes=(0, None, None)), qs, k, k)
      self.assertEqual(out.shape, (3,) + qshape)
      self.assertEqual(lse.shape, (3, 2, 4, 96))
      nonleading = _spec(qshape[:1] + (3,) + qshape[1:])
      out, _ = jax.eval_shape(jax.vmap(call, in_axes=(1, None, None)), nonleading, k, k)
      self.assertEqual(out.shape, (3,) + qshape)
      nested = jax.vmap(jax.vmap(call, in_axes=(0, None, None)), in_axes=(0, None, None))
      out, _ = jax.eval_shape(nested, _spec((2, 3) + qshape), k, k)
      self.assertEqual(out.shape, (2, 3) + qshape)
      loss = lambda a, b, c: call(a, b, c)[0].astype(jnp.float32).sum()
      grad = jax.vmap(jax.grad(loss, argnums=(0, 1, 2)), in_axes=(0, None, None))
      got = jax.eval_shape(grad, qs, k, k)
      self.assertEqual([x.shape for x in got], [(3,) + qshape, (3,) + kshape, (3,) + kshape])
      # grad(vmap) must reduce broadcast K/V cotangents back to their original ranks.
      total = lambda a, b, c: jax.vmap(call, in_axes=(0, None, None))(a, b, c)[0].astype(jnp.float32).sum()
      got = jax.eval_shape(jax.grad(total, argnums=(0, 1, 2)), qs, k, k)
      self.assertEqual([x.shape for x in got], [(3,) + qshape, kshape, kshape])

  @parameterized.parameters(False, True)
  def test_packed_varlen_maps_own_offsets_and_all_backward_results(self, ffi):
    q = _spec((3, 160, 4, 64))
    k = _spec((160, 2, 64))
    cu = _spec((3, 3), jnp.int32)
    call = lambda a, b, c, cq, ck: self.fa.fa_fwd_varlen(a, b, c, cq, ck, 96, 96)
    with mock.patch.object(self.fa, "_USE_FFI", ffi):
      out, lse = jax.eval_shape(jax.vmap(call, in_axes=(0, None, None, 0, 0)), q, k, k, cu, cu)
    self.assertEqual(out.shape, q.shape)
    self.assertEqual(lse.shape, (3, 4, 160))
    bwd = lambda a, b, c, o, do, l, cq, ck: self.fa.fa_varlen_bwd(a, b, c, o, do, l, cq, ck, 96, 96, mode="kernel")
    got = jax.eval_shape(jax.vmap(bwd, in_axes=(0, None, None, 0, 0, 0, 0, 0)), q, k, k, out, out, lse, cu, cu)
    self.assertEqual([x.shape for x in got], [q.shape, (3,) + k.shape, (3,) + k.shape])

  @parameterized.parameters(False, True)
  def test_pa_and_prefix_sequential_mixed_operands(self, ffi):
    q = _spec((3, 2, 1, 4, 128))
    cache = _spec((8, 64, 2, 128))
    bt = _spec((3, 2, 4), jnp.int32)
    lengths = _spec((3, 2), jnp.int32)
    pa = lambda a, b, c, table, lens: self.fa.fa_fwd_kvcache(a, b, c, table, lens, 128, num_splits=1)
    with mock.patch.object(self.fa, "_USE_FFI", ffi):
      out = jax.eval_shape(jax.vmap(pa, in_axes=(0, None, None, 0, 0)), q, cache, cache, bt, lengths)
      self.assertEqual(out.shape, q.shape)
      prefix = lambda a, b, c, table, lens, cu: self.fa.fa_prefix_prefill(a, b, c, table, lens, cu, 4, 128)
      out, lse = jax.eval_shape(jax.vmap(prefix, in_axes=(0, None, None, 0, 0, 0)), _spec((3, 7, 4, 128)), cache, cache, bt, lengths, _spec((3, 3), jnp.int32))
      self.assertEqual(out.shape, (3, 7, 4, 128))
      self.assertEqual(lse.shape, (3, 4, 7))
      # Neither paged decode nor prefix exposes a native training contract.
      loss = lambda a, c, table, lens: pa(a, c, c, table, lens).sum()
      with self.assertRaises((ValueError, NotImplementedError)):
        jax.eval_shape(jax.grad(loss), _spec((2, 1, 4, 128)), cache,
                       _spec((2, 4), jnp.int32), _spec((2,), jnp.int32))
      loss_prefix = lambda a, c, table, lens, cu: prefix(a, c, c, table, lens, cu)[0].sum()
      with self.assertRaises((ValueError, NotImplementedError)):
        jax.eval_shape(jax.grad(loss_prefix), _spec((7, 4, 128)), cache,
                       _spec((2, 4), jnp.int32), _spec((2,), jnp.int32),
                       _spec((3,), jnp.int32))

  def test_length_static_dynamic_sequential_and_grad(self):
    q = _spec((3, 2, 5, 4, 128))
    k = _spec((2, 7, 2, 128))
    call = functools.partial(self.fa.fa_fwd_attn_length, valid_k=3)
    out, lse = jax.eval_shape(jax.vmap(call, in_axes=(0, None, None)), q, k, k)
    self.assertEqual(out.shape, q.shape)
    self.assertEqual(lse.shape, (3, 2, 4, 5))
    dynamic = jax.vmap(self.fa.fa_fwd_attn_length, in_axes=(0, None, None, 0))
    out, lse = jax.eval_shape(dynamic, q, k, k, _spec((3,), jnp.int32))
    self.assertEqual(out.shape, q.shape)
    loss = lambda a, b, c, lengths: dynamic(a, b, c, lengths)[0].astype(jnp.float32).sum()
    got = jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), q, k, k, _spec((3,), jnp.int32))
    self.assertEqual([x.shape for x in got], [q.shape, k.shape, k.shape])


class PartitionRuleTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    from jax._src.cudnn import fa_attention
    self.fa = fa_attention
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("requires two devices (CPU devices are sufficient)")
    self.mesh = Mesh(np.asarray(devices[:2]), ("x",))

  def arg(self, shape, spec):
    return jax.ShapeDtypeStruct(shape, jnp.float16, sharding=NamedSharding(self.mesh, spec))

  @parameterized.product(layout=(0, 1), axis=("batch", "head"), shardy=(False, True))
  def test_dense_gqa_specs_and_local_shapes(self, layout, axis, shardy):
    qshape = (4, 96, 8, 64) if layout else (4, 8, 96, 64)
    kshape = (4, 160, 4, 64) if layout else (4, 4, 160, 64)
    spec = P("x", None, None, None) if axis == "batch" else (P(None, None, "x", None) if layout else P(None, "x", None, None))
    args = [self.arg(qshape, spec), self.arg(kshape, spec), self.arg(kshape, spec)]
    _, _, lse = self.fa._dense_partition_specs(self.mesh, args, layout)
    self.assertEqual(lse, ("x", None, None) if axis == "batch" else (None, "x", None))
    with mock.patch.object(self.fa, "_register"), mock.patch.object(self.fa, "_USE_FFI", True):
      def fake_ffi(q, k, v, *unused):
        b, h, _, sq, _, _, _ = self.fa._shapes(q, k, v, layout)
        return q, jnp.zeros((b, h, sq), jnp.float32)
      def fake_bwd(q, k, v, *unused):
        b, h, _, sq, sk, d, dv = self.fa._shapes(q, k, v, layout)
        ks = (b, sk, h, d) if layout else (b, h, sk, d)
        vs = ks[:-1] + (dv,)
        return (jnp.zeros_like(q), jnp.zeros(ks, q.dtype), jnp.zeros(vs, q.dtype),
                jnp.zeros((b, h, self.fa._rm(sq, 128)), jnp.float32))
      # CPU-compatible mocks test actual CP lowering, not vendor numerics.
      with (mock.patch.object(self.fa, "_fa_fwd_ffi", fake_ffi),
            mock.patch.object(self.fa, "_fa_bwd_ffi", fake_bwd),
            config.use_shardy_partitioner(shardy)):
        call = functools.partial(self.fa.fa_fwd_custom, layout=layout)
        sh = NamedSharding(self.mesh, spec)
        ls = NamedSharding(self.mesh, P(*lse))
        lowered = jax.jit(call, in_shardings=(sh, sh, sh), out_shardings=(sh, ls)).lower(*args)
        lowered.compile()
        loss = lambda a, b, c: call(a, b, c)[0].astype(jnp.float32).sum()
        grad = jax.grad(loss, argnums=(0, 1, 2))
        jax.jit(grad, in_shardings=(sh, sh, sh), out_shardings=(sh, sh, sh)).lower(*args).compile()

  def test_packed_replicated_cp_compiles(self):
    args = [self.arg((160, 4, 64), P()), self.arg((160, 2, 64), P()),
            self.arg((160, 2, 64), P()),
            jax.ShapeDtypeStruct((3,), jnp.int32, sharding=NamedSharding(self.mesh, P()))]
    def fake_ffi(q, k, v, *unused):
      return q, jnp.zeros((q.shape[1], q.shape[0]), jnp.float32)
    with (mock.patch.object(self.fa, "_register"),
          mock.patch.object(self.fa, "_USE_FFI", True),
          mock.patch.object(self.fa, "_fa_fwd_ffi", fake_ffi)):
      sh = NamedSharding(self.mesh, P())
      call = lambda q, k, v, cu: self.fa.fa_fwd_varlen(q, k, v, cu, cu, 96, 96)
      jax.jit(call, in_shardings=(sh,) * 4, out_shardings=(sh, sh)).lower(*args).compile()

  def test_rejects_sequence_feature_mismatched_v_and_partial_gqa(self):
    qshape, kshape = (4, 96, 8, 64), (4, 160, 4, 64)
    for spec in (P(None, "x", None, None), P(None, None, None, "x")):
      args = [self.arg(qshape, spec), self.arg(kshape, spec), self.arg(kshape, spec)]
      with self.assertRaisesRegex(ValueError, "sequence/feature"):
        self.fa._dense_partition_specs(self.mesh, args, 1)
    sh = P(None, None, "x", None)
    args = [self.arg(qshape, sh), self.arg(kshape, sh), self.arg(kshape, P())]
    with self.assertRaisesRegex(ValueError, "must match"):
      self.fa._dense_partition_specs(self.mesh, args, 1)
    args = [self.arg(qshape, sh), self.arg((4, 160, 1, 64), sh), self.arg((4, 160, 1, 64), sh)]
    with self.assertRaisesRegex(ValueError, "whole KV"):
      self.fa._dense_partition_specs(self.mesh, args, 1)

  @parameterized.parameters("dropout", "sinks")
  def test_sharded_dropout_and_sinks_rejected(self, feature):
    sh = P(None, None, "x", None)
    args = [self.arg((4, 96, 8, 64), sh), self.arg((4, 96, 4, 64), sh),
            self.arg((4, 96, 4, 64), sh)]
    args += [self.arg((n,), P()) for n in (4, 1, 64, 1, 8 if feature == "sinks" else 1, 2, 2)]
    with self.assertRaisesRegex(ValueError, feature):
      self.fa._fa_fwd_infer_sharding(.125, False, 1, -1, -1, 0., 0, 0,
          int(feature == "sinks"), 1, 0, 0, 0, .1 if feature == "dropout" else 0.,
          self.mesh, args, ())

  def test_packed_sharding_has_explicit_error(self):
    args = [self.arg((160, 4, 64), P("x", None, None))] * 3
    with self.assertRaisesRegex(ValueError, "packed sharding is unsupported"):
      self.fa._fa_fwd_infer_sharding(.125, False, 1, -1, -1, 0., 0, 0, 0, 0, 1, 96, 96, 0., self.mesh, args, ())


class HIPCTransformAcceptanceTest(jtu.JaxTestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if os.environ.get("FA_RUN_BATCHING_GPU_TESTS") != "1":
      message = "set FA_RUN_BATCHING_GPU_TESTS=1 in an assigned GPU slot"
      if os.environ.get("FA_REQUIRE_TESTS") == "1":
        raise RuntimeError(message)
      raise absltest.SkipTest(message)
    if not jtu.is_device_rocm():
      raise RuntimeError("HIPC acceptance requires real ROCm devices")
    from jax._src.cudnn import fa_attention
    cls.fa = fa_attention
    cls.fa._register()

  def arrays(self, layout=1):
    rng = np.random.default_rng(43)
    arrays = [jnp.asarray(rng.normal(0, .3, shape), jnp.float16)
              for shape in ((4, 96, 8, 64), (4, 96, 4, 64), (4, 96, 4, 64))]
    return arrays if layout else [x.transpose(0, 2, 1, 3) for x in arrays]

  @parameterized.product(layout=(0, 1), axis=("batch", "head"))
  def test_real_dense_gqa_forward_backward_partitioning(self, layout, axis):
    devices = jax.devices()
    if len(devices) < 2:
      self.fail("multi-GPU acceptance must not silently skip with one visible GPU")
    mesh = Mesh(np.asarray(devices[:2]), ("x",))
    q, k, v = self.arrays(layout)
    spec = P("x", None, None, None) if axis == "batch" else (P(None, None, "x", None) if layout else P(None, "x", None, None))
    ls = P("x", None, None) if axis == "batch" else P(None, "x", None)
    sh, lsh = NamedSharding(mesh, spec), NamedSharding(mesh, ls)
    call = functools.partial(self.fa.fa_fwd_custom, layout=layout)
    ref = call(q, k, v)
    compiled = jax.jit(call, in_shardings=(sh, sh, sh), out_shardings=(sh, lsh)).lower(q, k, v).compile()
    got = compiled(q, k, v)
    for actual, expected in zip(got, ref):
      self.assertArraysAllClose(actual, expected, rtol=3e-3, atol=3e-3)
    self.assertEqual(got[0].sharding, sh)
    self.assertEqual(got[1].sharding, lsh)
    # 前向编译结果须保留原生调用且不含 all-gather，避免以全量复制冒充分片。
    text = compiled.as_text()
    self.assertIn("fa_fwd_ffi" if self.fa._USE_FFI else "fa_fwd_v0", text)
    self.assertNotIn("all-gather(", text.lower())
    loss = lambda a, b, c: call(a, b, c)[0].astype(jnp.float32).sum()
    grad = jax.grad(loss, argnums=(0, 1, 2))
    ref_grad = grad(q, k, v)
    got_grad = jax.jit(grad, in_shardings=(sh, sh, sh), out_shardings=(sh, sh, sh))(q, k, v)
    for actual, expected in zip(got_grad, ref_grad):
      self.assertArraysAllClose(actual, expected, rtol=4e-3, atol=4e-3)

  def test_real_bhsd_unaligned_backward_against_math(self):
    q, k, v = self.arrays(layout=0)
    q, k, v = q[:2], k[:2], v[:2]
    def reference(a, b, c):
      a, b, c = (x.astype(jnp.float32) for x in (a, b, c))
      b, c = (jnp.repeat(x, a.shape[1] // b.shape[1], axis=1) for x in (b, c))
      logits = jnp.einsum("bhqd,bhkd->bhqk", a, b) * np.float32(.125)
      return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, axis=-1), c).sum()
    loss = lambda a, b, c: self.fa.fa_fwd_custom(a, b, c, layout=0)[0].astype(jnp.float32).sum()
    actual = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    expected = jax.grad(reference, argnums=(0, 1, 2))(q, k, v)
    for got, ref in zip(actual, expected):
      self.assertArraysAllClose(got, ref, rtol=4e-3, atol=4e-3)

  @parameterized.parameters("pa", "prefix")
  def test_real_sequential_paged_vmap(self, kind):
    rng = np.random.default_rng(56)
    cache = jnp.asarray(rng.normal(0, .3, (8, 64, 2, 128)), jnp.float16)
    values = jnp.asarray(rng.normal(0, .3, cache.shape), jnp.float16)
    tables = jnp.asarray([[[0, 1, 2, 3], [4, 5, 6, 7]],
                          [[4, 5, 6, 7], [0, 1, 2, 3]]], jnp.int32)
    lengths = jnp.asarray([[130, 200], [180, 129]], jnp.int32)
    if kind == "pa":
      q = jnp.asarray(rng.normal(0, .3, (2, 2, 1, 4, 128)), jnp.float16)
      def call(a, table, lens):
        return self.fa.fa_fwd_kvcache(a, cache, values, table, lens, 200,
                                      causal=True, num_splits=1)
      got = jax.jit(jax.vmap(call))(q, tables, lengths)
      expected = jnp.stack([call(q[i], tables[i], lengths[i]) for i in range(2)])
      self.assertArraysAllClose(got, expected, rtol=5e-3, atol=5e-3)
    else:
      q = jnp.asarray(rng.normal(0, .3, (2, 7, 4, 128)), jnp.float16)
      cu = jnp.asarray([[0, 4, 7], [0, 3, 7]], jnp.int32)
      def call(a, table, lens, offsets):
        return self.fa.fa_prefix_prefill(a, cache, values, table, lens, offsets,
                                         4, 200, causal=True)
      got = jax.jit(jax.vmap(call))(q, tables, lengths, cu)
      expected = jax.tree.map(lambda *xs: jnp.stack(xs),
                              *(call(q[i], tables[i], lengths[i], cu[i]) for i in range(2)))
      for actual, reference in zip(got, expected):
        self.assertArraysAllClose(actual, reference, rtol=5e-3, atol=5e-3)

  def test_real_sequential_dense_vmap(self):
    q, k, v = self.arrays()
    qs = jnp.stack([q, q * .75])
    call = self.fa.fa_fwd_custom
    got = jax.jit(jax.vmap(call, in_axes=(0, None, None)))(qs, k, v)
    expected = jax.tree.map(lambda *xs: jnp.stack(xs), *(call(x, k, v) for x in qs))
    for actual, reference in zip(got, expected):
      self.assertArraysAllClose(actual, reference, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
  absltest.main()
