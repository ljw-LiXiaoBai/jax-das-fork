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

"""FlashAttention 验收，数值结果与梯度对照独立 fp32 数学参考。

原生验收缺少前置条件时跳过；FA_REQUIRE_TESTS=1 时失败。
显式库路径无效、插件损坏及执行错误不得作为跳过处理。
"""
import os
import subprocess
import sys
from unittest import mock

from absl.testing import absltest, parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding, PartitionSpec

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


TOL = {jnp.float16: (2e-3, 2e-3), jnp.bfloat16: (1e-2, 1e-2)}
TOL_BWD = {jnp.float16: (3e-3, 3e-3), jnp.bfloat16: (1.5e-2, 1.5e-2)}


class FaDiscoveryTest(absltest.TestCase):

  def test_missing_plugin_has_distinct_error(self):
    from jax._src.cudnn import fa_attention
    with mock.patch("importlib.util.find_spec", return_value=None):
      with self.assertRaises(ModuleNotFoundError) as caught:
        fa_attention._load_plugin()
    self.assertEqual(caught.exception.name, "fa_attention_plugin")

  def test_broken_plugin_import_is_not_hidden(self):
    from jax._src.cudnn import fa_attention
    with mock.patch("importlib.util.find_spec", return_value=object()):
      with mock.patch("importlib.import_module", side_effect=ImportError("broken extension")):
        with self.assertRaisesRegex(ImportError, "broken extension"):
          fa_attention._load_plugin()

  def test_explicit_missing_library_is_not_skipped(self):
    with mock.patch.object(jtu, "is_device_rocm", return_value=True):
      with mock.patch.dict(os.environ, {"FA_LIB_PATH": "/missing/fa/library.so"}):
        with mock.patch("os.path.isfile", return_value=False):
          with self.assertRaisesRegex(FileNotFoundError, "FA_LIB_PATH"):
            _load_fa()


class FaMlaUnsupportedTest(parameterized.TestCase):

  @parameterized.product(legacy=(False, True),
                         transform=("eager", "jit", "grad", "jit_grad", "vmap"))
  def test_decode_rejected_before_registration(self, legacy, transform):
    from jax._src.cudnn import fa_attention
    q = np.zeros((1, 2, 4, 576), np.float16)
    cache = np.zeros((2, 128, 1, 576), np.float16)
    table = np.array([[0, 1]], np.int32)
    lengths = np.array([256], np.int32)

    def call(x):
      return fa_attention.fa_mla(x, cache, table, lengths, 256, causal=True)

    if transform == "jit":
      call = jax.jit(call)
    elif transform in ("grad", "jit_grad"):
      forward = call
      call = jax.grad(lambda x: forward(x).astype(jnp.float32).sum())
      if transform == "jit_grad":
        call = jax.jit(call)
    elif transform == "vmap":
      call = jax.vmap(call)
      q = q[None, ...]
    with mock.patch.object(fa_attention, "_USE_FFI", not legacy):
      with mock.patch.object(fa_attention, "_register") as register:
        with self.assertRaisesRegex(
            NotImplementedError, "MLA decode.*停止维护.*剔除.*run_fwd_flashmla"):
          call(q)
        register.assert_not_called()

  def test_decode_rejected_without_arrays_or_plugin(self):
    from jax._src.cudnn import fa_attention
    with mock.patch.object(fa_attention, "_load_plugin") as load:
      with self.assertRaisesRegex(NotImplementedError, "MLA decode"):
        fa_attention.fa_mla(None, None, None, None, None)
      load.assert_not_called()


class FaAttentionTest(jtu.JaxTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.fa = _load_fa()

  @staticmethod
  def _wmask(s, causal, wl, wr):
    """厂商窗口语义 (flash_api.cpp): 掩码 = causal ∩ [i-wl, i+wr]; -1 = 无限。

    causal=True 时厂商入口无条件令 wr=0, 即 causal 与显式右窗取交 —— 所以
    causal=1 + window=(64,64) 与 (64,0) 等价 (都是 [i-64, i]); 这里按同一语义建模。
    """
    i = np.arange(s)[:, None]
    j = np.arange(s)[None, :]
    m = np.ones((s, s), bool)
    if causal:
      m &= (j <= i)
    if wl >= 0:
      m &= (i - j <= wl)
    if wr >= 0:
      m &= (j - i <= wr)
    return m

  def _dense_ref(self, q, k, v, do, causal, scale, layout, wl=-1, wr=-1):
    """稠密参考: (o, lse, dq, dk, dv), 与输入同 layout。"""
    def to_bhsd(x):
      x = x.astype(jnp.float32)
      return x.transpose(0, 2, 1, 3) if layout == 1 else x

    def from_bhsd(x):
      return x.transpose(0, 2, 1, 3) if layout == 1 else x

    qf, kf, vf, dof = (to_bhsd(x) for x in (q, k, v, do))
    g = qf.shape[1] // kf.shape[1]
    kf2 = jnp.repeat(kf, g, 1) if g > 1 else kf
    vf2 = jnp.repeat(vf, g, 1) if g > 1 else vf
    wm = jnp.asarray(self._wmask(qf.shape[2], causal, wl, wr))[None, None]

    def f(a, bb, cc):
      sc = jnp.einsum("bhqd,bhkd->bhqk", a, bb,
                      preferred_element_type=jnp.float32) * scale
      sc = jnp.where(wm, sc, -jnp.inf)
      mx = sc.max(-1, keepdims=True)
      p = jnp.exp(sc - mx)
      p = p / p.sum(-1, keepdims=True)
      return jnp.sum(jnp.einsum("bhqk,bhkd->bhqd", p, cc,
                                preferred_element_type=jnp.float32) * dof)

    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(qf, kf2, vf2)
    if g > 1:  # bhsd: (b, h=hk*g, s, d) → (b, hk, g, s, d).sum(2)
      dk = dk.reshape(dk.shape[0], kf.shape[1], g, dk.shape[2], dk.shape[3]).sum(2)
      dv = dv.reshape(dv.shape[0], kf.shape[1], g, dv.shape[2], dv.shape[3]).sum(2)

    sc = jnp.einsum("bhqd,bhkd->bhqk", qf, kf2,
                    preferred_element_type=jnp.float32) * scale
    sc = jnp.where(wm, sc, -jnp.inf)
    mx = sc.max(-1, keepdims=True)
    lse = (mx + jnp.log(jnp.exp(sc - mx).sum(-1, keepdims=True))).squeeze(-1)
    p = jnp.exp(sc - mx)
    p = p / p.sum(-1, keepdims=True)
    o = jnp.einsum("bhqk,bhkd->bhqd", p, vf2, preferred_element_type=jnp.float32)
    lse_out = lse  # 接入层两种 layout 下 lse 都是 (b,h,s)
    return from_bhsd(o), lse_out, from_bhsd(dq), from_bhsd(dk), from_bhsd(dv)

  def _varlen_ref(self, q, k, v, do, cu, causal, scale, wl=-1, wr=-1):
    """变长参考 (3-D 打包): (o, lse, dq, dk, dv), lse 为 (h,total_q)。"""
    qf, kf, vf, dof = (jnp.asarray(x, jnp.float32) for x in (q, k, v, do))
    cu = np.asarray(cu)
    g = qf.shape[1] // kf.shape[1]
    masks = [jnp.asarray(self._wmask(int(cu[i + 1]) - int(cu[i]), causal, wl, wr))
             for i in range(len(cu) - 1)]

    def f(a, bb, cc):
      tot = jnp.zeros_like(dof)
      for i in range(len(cu) - 1):
        s0, s1 = int(cu[i]), int(cu[i + 1])
        kk = jnp.repeat(bb[s0:s1], g, 1)
        cc2 = jnp.repeat(cc[s0:s1], g, 1)
        sc = jnp.einsum("shd,lhd->hsl", a[s0:s1], kk) * scale
        sc = jnp.where(masks[i][None], sc, -jnp.inf)
        mx = sc.max(-1, keepdims=True)
        p = jnp.exp(sc - mx)
        p = p / p.sum(-1, keepdims=True)
        tot = tot.at[s0:s1].set(jnp.einsum("hsl,lhd->shd", p, cc2))
      return (tot * dof).sum()

    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(qf, kf, vf)
    o = jnp.zeros_like(qf)
    lse = jnp.zeros((qf.shape[1], qf.shape[0]), jnp.float32)
    for i in range(len(cu) - 1):
      s0, s1 = int(cu[i]), int(cu[i + 1])
      kk = jnp.repeat(kf[s0:s1], g, 1)
      sc = jnp.einsum("shd,lhd->hsl", qf[s0:s1], kk) * scale
      sc = jnp.where(masks[i][None], sc, -jnp.inf)
      mx = sc.max(-1, keepdims=True)
      p = jnp.exp(sc - mx)
      p = p / p.sum(-1, keepdims=True)
      o = o.at[s0:s1].set(jnp.einsum("hsl,lhd->shd", p,
                                     jnp.repeat(vf[s0:s1], g, 1)))
      lse = lse.at[:, s0:s1].set(
          (mx + jnp.log(jnp.exp(sc - mx).sum(-1, keepdims=True))).squeeze(-1))
    return o, lse, dq, dk, dv

  def _pa_ref(self, q, kc, vc, bt, seqlens_k, scale, causal):
    """分页解码参考 (numpy fp32): 按 block_table gather 成连续 KV 再算 (右对齐因果)。"""
    b, sq, h, d = q.shape
    page, hk = kc.shape[1], kc.shape[2]
    dv = vc.shape[3]
    g = h // hk
    outs = []
    for i in range(b):
      L = int(seqlens_k[i])
      idx = np.asarray(bt[i, : (L + page - 1) // page])
      k_i = np.asarray(kc[idx]).reshape(-1, hk, d)[:L]
      v_i = np.asarray(vc[idx]).reshape(-1, hk, dv)[:L]
      q_i = np.asarray(q[i], np.float32)
      sc = np.einsum("shd,lhd->hsl", q_i,
                     np.asarray(np.repeat(k_i, g, 1), np.float32)) * scale
      if causal:
        kv = np.arange(L)
        pos = np.arange(L - sq, L)
        sc = np.where(kv[None, :] <= pos[:, None], sc, -np.inf)
      mx = sc.max(-1, keepdims=True)
      p = np.exp(sc - mx)
      p = p / p.sum(-1, keepdims=True)
      outs.append(np.einsum("hsl,lhd->shd", p,
                            np.asarray(np.repeat(v_i, g, 1), np.float32)))
    return np.stack(outs)

  def _prefix_ref(self, q, kc, vc, bt, seqlens_k, cu_q, scale, causal):
    """prefix-prefill 参考 (numpy fp32): 分页 KV + 逐序列右对齐因果。"""
    page, hk = kc.shape[1], kc.shape[2]
    h, dv = q.shape[1], vc.shape[3]
    g = h // hk
    outs = []
    for i in range(len(cu_q) - 1):
      L = int(seqlens_k[i])
      idx = np.asarray(bt[i, : (L + page - 1) // page])
      ki = np.asarray(kc[idx]).reshape(-1, hk, kc.shape[3])[:L].astype(np.float32)
      vi = np.asarray(vc[idx]).reshape(-1, hk, dv)[:L].astype(np.float32)
      s0, s1 = int(cu_q[i]), int(cu_q[i + 1])
      qi = np.asarray(q[s0:s1], np.float32)
      sq = s1 - s0
      o_i = np.zeros((sq, h, dv), np.float32)
      for hh in range(h):
        sc = qi[:, hh] @ ki[:, hh // g].T * scale
        if causal:
          pos = np.arange(L - sq, L)
          sc = np.where(np.arange(L)[None, :] <= pos[:, None], sc, -np.inf)
        mx = sc.max(-1, keepdims=True)
        p = np.exp(sc - mx)
        p = p / p.sum(-1, keepdims=True)
        o_i[:, hh] = p @ vi[:, hh // g]
      outs.append(o_i)
    return np.concatenate(outs, 0)

  @jtu.sample_product(dtype=(jnp.float16, jnp.bfloat16), causal=(False, True),
                      layout=(1, 0))
  def test_dense_forward(self, dtype, causal, layout):
    b, s, h, d = 2, 128, 2, 64
    scale = d ** -0.5
    shape = (b, s, h, d) if layout == 1 else (b, h, s, d)
    q = (jax.random.normal(jax.random.key(0), shape) * .3).astype(dtype)
    k = (jax.random.normal(jax.random.key(1), shape) * .3).astype(dtype)
    v = (jax.random.normal(jax.random.key(2), shape) * .3).astype(dtype)
    o, lse = self.fa.fa_fwd(q, k, v, causal=causal, softmax_scale=scale,
                            layout=layout)
    ref_o, ref_lse, *_ = self._dense_ref(q, k, v, jnp.zeros_like(q), causal,
                                         scale, layout)
    rtol, atol = TOL[dtype]
    self.assertArraysAllClose(o, ref_o, rtol=rtol, atol=atol, check_dtypes=False)
    self.assertArraysAllClose(lse, ref_lse, rtol=1e-4, atol=1e-4)

  @parameterized.product(dtype=(jnp.float16, jnp.bfloat16), layout=(0, 1),
                         lengths=((64, 128), (128, 64), (33, 97)))
  def test_dense_forward_cross_attention(self, dtype, layout, lengths):
    sq, sk = lengths
    b, h, hk, d = 2, 4, 2, 64
    rng = np.random.default_rng(47)
    q, k, v = [jnp.asarray(rng.normal(0, .3, shape), dtype)
               for shape in ((b, sq, h, d), (b, sk, hk, d), (b, sk, hk, d))]
    qf, kf, vf = (np.asarray(x, np.float32) for x in (q, k, v))
    kf, vf = np.repeat(kf, h // hk, 2), np.repeat(vf, h // hk, 2)
    sc = np.einsum("bqhd,bkhd->bhqk", qf, kf) * np.float32(.07)
    maximum = sc.max(-1, keepdims=True)
    exps = np.exp(sc - maximum)
    denominator = exps.sum(-1, keepdims=True)
    ref_o = np.einsum("bhqk,bkhd->bqhd", exps / denominator, vf)
    ref_lse = (maximum + np.log(denominator))[..., 0]
    if layout == 0:
      q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
      ref_o = ref_o.transpose(0, 2, 1, 3)
    def call(a, b_, c):
      return self.fa.fa_fwd(a, b_, c, layout=layout, softmax_scale=.07)
    for out, lse in (call(q, k, v), jax.jit(call)(q, k, v)):
      self.assertEqual(out.shape, q.shape)
      self.assertEqual(lse.shape, (b, h, sq))
      self.assertArraysAllClose(out, ref_o, rtol=TOL[dtype][0], atol=TOL[dtype][1],
                                check_dtypes=False)
      self.assertArraysAllClose(lse, ref_lse, rtol=1e-4, atol=1e-4)

  @jtu.sample_product(dtype=(jnp.float16, jnp.bfloat16), causal=(False, True),
                      gqa=(False, True))
  def test_dense_backward(self, dtype, causal, gqa):
    b, s, h, hk, d = 2, 128, 4 if gqa else 2, 2, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(dtype)
    k = (jax.random.normal(jax.random.key(1), (b, s, hk, d)) * .3).astype(dtype)
    v = (jax.random.normal(jax.random.key(2), (b, s, hk, d)) * .3).astype(dtype)
    do = (jax.random.normal(jax.random.key(3), (b, s, h, d)) * .5).astype(dtype)

    def loss(qq, kk, vv):
      oo, _ = self.fa.fa_fwd_custom(qq, kk, vv, causal=causal, softmax_scale=scale)
      return (oo.astype(jnp.float32) * jnp.asarray(do, jnp.float32)).sum()

    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    _, _, ref_dq, ref_dk, ref_dv = self._dense_ref(q, k, v, do, causal, scale, 1)
    rtol, atol = TOL_BWD[dtype]
    for got, ref in ((dq, ref_dq), (dk, ref_dk), (dv, ref_dv)):
      self.assertArraysAllClose(got, ref, rtol=rtol, atol=atol, check_dtypes=False)

  @parameterized.named_parameters(("s96", 96), ("s160", 160), ("s200", 200))
  def test_dense_backward_batch_gt1_seqlen_not_multiple_of_128(self, s):
    """批大小大于 1、序列长度未按 128 对齐时，输出步长仍须匹配对齐后的缓冲区。"""
    b, h, hk, d = 3, 4, 2, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(jnp.float16)
    k = (jax.random.normal(jax.random.key(1), (b, s, hk, d)) * .3).astype(jnp.float16)
    v = (jax.random.normal(jax.random.key(2), (b, s, hk, d)) * .3).astype(jnp.float16)
    do = (jax.random.normal(jax.random.key(3), (b, s, h, d)) * .5).astype(jnp.float16)

    def loss(qq, kk, vv):
      oo, _ = self.fa.fa_fwd_custom(qq, kk, vv, causal=False, softmax_scale=scale)
      return (oo.astype(jnp.float32) * jnp.asarray(do, jnp.float32)).sum()

    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    _, _, ref_dq, ref_dk, ref_dv = self._dense_ref(q, k, v, do, False, scale, 1)
    for got, ref in ((dq, ref_dq), (dk, ref_dk), (dv, ref_dv)):
      self.assertArraysAllClose(got, ref, rtol=3e-3, atol=3e-3, check_dtypes=False)

  def test_custom_entry_returns_consistent_eager_and_jit(self):
    """fa_fwd_custom 在 eager / jit / grad 下都返回 (o, lse) 的契约回归。"""
    q = (jax.random.normal(jax.random.key(0), (1, 64, 2, 64)) * .3).astype(jnp.float16)
    o, lse = self.fa.fa_fwd_custom(q, q, q)                 # eager
    self.assertEqual(o.shape, (1, 64, 2, 64))
    self.assertEqual(lse.shape, (1, 2, 64))
    oj, lj = jax.jit(lambda a: self.fa.fa_fwd_custom(a, a, a))(q)   # jit
    self.assertArraysEqual(o, oj)
    self.assertArraysEqual(lse, lj)
    gq = jax.grad(lambda a: self.fa.fa_fwd_custom(a, a, a)[0]
                  .astype(jnp.float32).sum())(q)             # grad 仍可用
    self.assertEqual(gq.shape, (1, 64, 2, 64))
    self.assertTrue(bool(jnp.isfinite(gq).all()))

  def test_varlen_forward_and_backward(self):
    lens = [64, 96]
    h, hk, d = 2, 1, 64
    tq, msq = sum(lens), max(lens)
    scale = d ** -0.5
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    rng = np.random.default_rng(0)
    q = rng.normal(0, .3, (tq, h, d)).astype(np.float16)
    k = rng.normal(0, .3, (tq, hk, d)).astype(np.float16)
    v = rng.normal(0, .3, (tq, hk, d)).astype(np.float16)
    do = rng.normal(0, .5, (tq, h, d)).astype(np.float16)
    qj, kj, vj, doj = (jnp.asarray(x) for x in (q, k, v, do))
    cuj = jnp.asarray(cu)
    o, lse = self.fa.fa_fwd_varlen(qj, kj, vj, cuj, cuj, msq, msq,
                                   causal=False, softmax_scale=scale)
    ref_o, ref_lse, ref_dq, ref_dk, ref_dv = self._varlen_ref(q, k, v, do, cu,
                                                              False, scale)
    self.assertArraysAllClose(o, ref_o, rtol=2e-3, atol=2e-3, check_dtypes=False)
    self.assertArraysAllClose(lse, ref_lse, rtol=1e-4, atol=1e-4)
    # 反向: 内核路径 (默认) 与 dense_loop 降级都验收, 并互相对照。
    dq, dk, dv = self.fa.fa_varlen_bwd(qj, kj, vj, o, doj, lse, cuj, cuj, msq,
                                       msq, causal=False, softmax_scale=scale)
    for got, ref in ((dq, ref_dq), (dk, ref_dk), (dv, ref_dv)):
      self.assertArraysAllClose(got, ref, rtol=3e-3, atol=3e-3, check_dtypes=False)
    dq2, dk2, dv2 = self.fa.fa_varlen_bwd(qj, kj, vj, o, doj, lse, cuj, cuj, msq,
                                          msq, causal=False, softmax_scale=scale,
                                          mode="dense_loop")
    for a, b_ in ((dq, dq2), (dk, dk2), (dv, dv2)):
      self.assertArraysAllClose(a, b_, rtol=3e-3, atol=3e-3, check_dtypes=False)

  @jtu.sample_product(dtype=(jnp.float16, jnp.bfloat16))
  def test_varlen_window_forward_and_backward(self, dtype):
    """变长 + 滑窗 (含 causal 与显式窗口取交): 前向与反向都对 fp32 参考验收。"""
    lens = [96, 160, 64]
    h, hk, d = 2, 2, 64
    tq, msq = sum(lens), max(lens)
    scale = d ** -0.5
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    rng = np.random.default_rng(3)
    q = rng.normal(0, .3, (tq, h, d)).astype(dtype)
    k = rng.normal(0, .3, (tq, hk, d)).astype(dtype)
    v = rng.normal(0, .3, (tq, hk, d)).astype(dtype)
    do = rng.normal(0, .5, (tq, h, d)).astype(dtype)
    qj, kj, vj, doj = (jnp.asarray(x) for x in (q, k, v, do))
    cuj = jnp.asarray(cu)
    tol = TOL[dtype]
    for causal, wl, wr in ((False, 64, 64), (True, 64, 0), (True, 64, 64)):
      ref_o, ref_lse, ref_dq, ref_dk, ref_dv = self._varlen_ref(
          q, k, v, do, cu, causal, scale, wl, wr)
      o, lse = self.fa.fa_fwd_varlen(qj, kj, vj, cuj, cuj, msq, msq,
                                     causal=causal, softmax_scale=scale,
                                     window_size=(wl, wr))
      self.assertArraysAllClose(o, ref_o, rtol=tol[0], atol=tol[1],
                                check_dtypes=False)
      for mode in ("dense_loop", "kernel"):
        dq, dk, dv = self.fa.fa_varlen_bwd(
            qj, kj, vj, o, doj, lse, cuj, cuj, msq, msq, causal=causal,
            softmax_scale=scale, window_size=(wl, wr), mode=mode,
            seq_lens_q=lens, seq_lens_k=lens)
        for got, ref in ((dq, ref_dq), (dk, ref_dk), (dv, ref_dv)):
          self.assertArraysAllClose(got, ref, rtol=TOL_BWD[dtype][0],
                                    atol=TOL_BWD[dtype][1], check_dtypes=False)

  def test_deterministic_backward(self):
    b, s, h, hk, d = 2, 256, 4, 2, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(jnp.float16)
    k = (jax.random.normal(jax.random.key(1), (b, s, hk, d)) * .3).astype(jnp.float16)
    v = (jax.random.normal(jax.random.key(2), (b, s, hk, d)) * .3).astype(jnp.float16)
    do = (jax.random.normal(jax.random.key(3), (b, s, h, d)) * .5).astype(jnp.float16)

    def grads(det):
      def loss(qq, kk, vv):
        oo, _ = self.fa.fa_fwd_custom(qq, kk, vv, causal=False, softmax_scale=scale,
                                      deterministic=det)
        return (oo.astype(jnp.float32) * jnp.asarray(do, jnp.float32)).sum()
      return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    r1, r2, r0 = grads(True), grads(True), grads(False)
    for a, b_ in zip(r1, r2):        # 同设置两次调用 → 逐位相同 (可复现性)
      self.assertArraysEqual(a, b_)
    for a, b_ in zip(r1, r0):        # det=1 与 det=0 是同一数学 → 逐位相同
      self.assertArraysEqual(a, b_)
    _, _, ref_dq, ref_dk, ref_dv = self._dense_ref(q, k, v, do, False, scale, 1)
    for got, ref in ((r1[0], ref_dq), (r1[1], ref_dk), (r1[2], ref_dv)):
      self.assertArraysAllClose(got, ref, rtol=3e-3, atol=3e-3, check_dtypes=False)

  def test_dropout(self):
    b, s, h, d, dp = 1, 128, 2, 64, 0.1
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(jnp.float16)
    k = (jax.random.normal(jax.random.key(1), (b, s, h, d)) * .3).astype(jnp.float16)
    v = (jax.random.normal(jax.random.key(2), (b, s, h, d)) * .3).astype(jnp.float16)
    base, _ = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale)
    o1, _ = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale,
                           dropout_p=dp, seed=7)
    o2, _ = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale,
                           dropout_p=dp, seed=7)
    o3, _ = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale,
                           dropout_p=dp, seed=1234)
    o0, _ = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale,
                           dropout_p=0.0, seed=7)
    self.assertArraysEqual(o1, o2)                     # 同 seed → 逐位一致
    self.assertFalse(np.array_equal(np.asarray(o1), np.asarray(o3)),
                     "不同 seed 的输出不应完全相同")
    self.assertArraysEqual(o0, base)                   # dropout=0 与原路径逐位一致
    ratio = float(np.abs(np.asarray(o1, np.float32)).mean()
                  / (np.abs(np.asarray(base, np.float32)).mean() + 1e-12))
    self.assertBetween(ratio, 0.8, 1.25)               # 保留概率 ~0.9 → 量级不变

  @jtu.sample_product(dtype=(jnp.float16, jnp.bfloat16))
  def test_sliding_window(self, dtype):
    """滑窗 (wl=64, wr=0): 与 fp32 窗内 mask 参考对比。"""
    b, s, h, d, wl = 1, 128, 2, 64, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(dtype)
    k = (jax.random.normal(jax.random.key(1), (b, s, h, d)) * .3).astype(dtype)
    v = (jax.random.normal(jax.random.key(2), (b, s, h, d)) * .3).astype(dtype)
    o, lse = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale,
                            window_size=(wl, 0))
    qf, kf, vf = (np.asarray(x, np.float32).transpose(0, 2, 1, 3) for x in (q, k, v))
    sc = np.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    i = np.arange(s)[:, None]
    j = np.arange(s)[None, :]
    # 非因果滑窗 (wl, 0) 的语义 = 因果 + 左窗: mask 掉 j > i (未来) 与 i - j > wl
    # (超出左窗); 剩余 j ∈ [i-wl, i]
    wmask = (j > i) | (i - j > wl)
    sc = np.where(wmask[None, None], -np.inf, sc)
    m = sc.max(-1, keepdims=True)
    p = np.exp(sc - m)
    p /= p.sum(-1, keepdims=True)
    ref_o = np.einsum("bhqk,bhkd->bhqd", p, vf).transpose(0, 2, 1, 3)
    ref_lse = (m + np.log(np.exp(sc - m).sum(-1, keepdims=True))).squeeze(-1)  # (b,h,s)
    rtol, atol = TOL[dtype]
    self.assertArraysAllClose(o, ref_o, rtol=rtol, atol=atol, check_dtypes=False)
    self.assertArraysAllClose(lse, ref_lse, rtol=1e-3, atol=1e-3)

  @jtu.sample_product(dtype=(jnp.float16, jnp.bfloat16))
  def test_sliding_window_backward(self, dtype):
    """滑窗反向回归: (causal, wl, wr) 三种组合的前向与梯度都对 fp32 参考验收。"""
    b, s, h, d, wl = 1, 128, 2, 64, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(dtype)
    k = (jax.random.normal(jax.random.key(1), (b, s, h, d)) * .3).astype(dtype)
    v = (jax.random.normal(jax.random.key(2), (b, s, h, d)) * .3).astype(dtype)
    do = (jax.random.normal(jax.random.key(3), (b, s, h, d)) * .5).astype(dtype)
    rtol, atol = TOL[dtype]
    brtol, batol = TOL_BWD[dtype]
    for causal, wr in ((False, 64), (False, 0), (True, 64)):
      def loss(qq, kk, vv):
        oo, _ = self.fa.fa_fwd_custom(qq, kk, vv, causal=causal,
                                      softmax_scale=scale, window_size=(wl, wr))
        return (oo.astype(jnp.float32) * jnp.asarray(do, jnp.float32)).sum()
      gq, gk, gv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
      ref_o, _, ref_dq, ref_dk, ref_dv = self._dense_ref(
          q, k, v, do, causal, scale, 1, wl, wr)
      o, _ = self.fa.fa_fwd(q, k, v, causal=causal, softmax_scale=scale,
                            window_size=(wl, wr))
      self.assertArraysAllClose(o, ref_o, rtol=rtol, atol=atol, check_dtypes=False)
      for got, ref in ((gq, ref_dq), (gk, ref_dk), (gv, ref_dv)):
        self.assertArraysAllClose(got, ref, rtol=brtol, atol=batol,
                                  check_dtypes=False)

  def test_attention_sinks(self):
    """sinks (s_aux): sink 项参与 softmax 分母, 与 fp32 参考对比。"""
    b, s, h, d = 1, 128, 2, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(jnp.float16)
    k = (jax.random.normal(jax.random.key(1), (b, s, h, d)) * .3).astype(jnp.float16)
    v = (jax.random.normal(jax.random.key(2), (b, s, h, d)) * .3).astype(jnp.float16)
    sinks = jnp.asarray(np.array([0.7, -0.4], np.float32))
    o, lse = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale, sinks=sinks)
    qf, kf, vf = (np.asarray(x, np.float32).transpose(0, 2, 1, 3) for x in (q, k, v))
    sc = np.einsum("bhqd,bhkd->bhqk", qf, kf) * scale   # 非因果
    # sink: [scores, sink] 拼接后整体 softmax; O 只取前 s 项 (sink 不产生输出)
    snk = np.asarray(sinks, np.float32)
    cat = np.concatenate([sc, np.broadcast_to(snk[None, :, None, None], (b, h, s, 1))],
                         axis=-1)
    mx = cat.max(-1, keepdims=True)
    p = np.exp(cat - mx)
    p = p / p.sum(-1, keepdims=True)
    ref_o = np.einsum("bhqk,bhkd->bhqd", p[..., :s], vf).transpose(0, 2, 1, 3)
    # 内核 lse 语义 = 含 sink 的 logsumexp (与训练侧 dP 的用法一致)
    ref_lse = (mx + np.log(np.exp(cat - mx).sum(-1, keepdims=True))).squeeze(-1)
    self.assertArraysAllClose(o, ref_o, rtol=2e-3, atol=2e-3, check_dtypes=False)
    self.assertArraysAllClose(lse, ref_lse, rtol=1e-3, atol=1e-3)

  def test_paged_attention_decode(self):
    b, sq, h, hk, d, dv, page = 2, 1, 4, 2, 64, 64, 128
    L = 512
    scale = d ** -0.5
    rng = np.random.default_rng(0)
    nblk = (L + page - 1) // page
    q = rng.normal(0, .3, (b, sq, h, d)).astype(np.float16)
    kc = rng.normal(0, .3, (nblk * b, page, hk, d)).astype(np.float16)
    vc = rng.normal(0, .3, (nblk * b, page, hk, dv)).astype(np.float16)
    bt = np.stack([np.arange(i * nblk, (i + 1) * nblk, dtype=np.int32)
                   for i in range(b)])
    seqlens = np.full((b,), L, np.int32)
    o = self.fa.fa_fwd_kvcache(jnp.asarray(q), jnp.asarray(kc), jnp.asarray(vc),
                               jnp.asarray(bt), jnp.asarray(seqlens), L,
                               causal=True, softmax_scale=scale)
    ref = self._pa_ref(q, kc, vc, bt, seqlens, scale, True)
    self.assertArraysAllClose(o, ref, rtol=5e-3, atol=5e-3, check_dtypes=False)

  def test_prefix_prefill(self):
    lens_q, lens_k, h, hk, d, page = [4, 3], [130, 200], 2, 1, 128, 64
    tq, msq, msk = sum(lens_q), max(lens_q), max(lens_k)
    scale = d ** -0.5
    rng = np.random.default_rng(0)
    q = rng.normal(0, .3, (tq, h, d)).astype(np.float16)
    kc = rng.normal(0, .3, (8, page, hk, d)).astype(np.float16)
    vc = rng.normal(0, .3, (8, page, hk, d)).astype(np.float16)
    bt = np.zeros((2, 4), np.int32)
    for i in range(2):
      need = (lens_k[i] + page - 1) // page
      bt[i, :need] = np.arange(i * need, (i + 1) * need)
    cu_q = np.concatenate([[0], np.cumsum(lens_q)]).astype(np.int32)
    o, lse = self.fa.fa_prefix_prefill(
        jnp.asarray(q), jnp.asarray(kc), jnp.asarray(vc), jnp.asarray(bt),
        jnp.asarray(np.asarray(lens_k, np.int32)), jnp.asarray(cu_q), msq, msk,
        causal=True, softmax_scale=scale)
    ref = self._prefix_ref(q, kc, vc, bt, lens_k, cu_q, scale, True)
    self.assertArraysAllClose(o, ref, rtol=5e-3, atol=5e-3, check_dtypes=False)

  def test_mla_decode_targets_are_not_registered(self):
    registrations = self.fa._PLUGIN_MOD.registrations()["ROCM"]
    names = {name for name, _capsule, _version in registrations}
    self.assertNotIn("fa_mla_v0", names)
    self.assertNotIn("fa_mla_ffi", names)
    self.assertIn("fa_fwd_kvcache_v0", names)
    self.assertIn("fa_pa_ffi", names)
    self.assertIn("fa_prefix_v0", names)
    self.assertIn("fa_prefix_ffi", names)

  def test_first_call_under_jit_does_not_leak_tracers(self):
    snippet = (
        "import jax, jax.numpy as jnp, numpy as np\n"
        "from jax._src.cudnn import fa_attention as f\n"
        "q=jnp.ones((1,64,2,64),jnp.float16)\n"
        "with jax.checking_leaks():\n"
        "  out=jax.jit(lambda x:f.fa_fwd(x,x,x))(q)\n"
        "  jax.block_until_ready(out)\n"
        "  eager=f.fa_fwd(q,q,q)\n"
        "  grad=jax.grad(lambda x:f.fa_fwd_custom(x,x,x)[0].astype(jnp.float32).sum())(q)\n"
        "np.testing.assert_allclose(out[0],eager[0],rtol=2e-3,atol=2e-3)\n"
        "assert np.isfinite(np.asarray(grad)).all()\n"
        "print('COLD_JIT_PASS')\n")
    result = subprocess.run([sys.executable, "-c", snippet],
                            env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
                            capture_output=True, text=True, timeout=120)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("COLD_JIT_PASS", result.stdout)

  def test_missing_vendor_library_ffi_error(self):
    snippet = (
        "import jax, jax.numpy as jnp\n"
        "from jax._src.cudnn import fa_attention as f\n"
        "q=jnp.zeros((1,4,1,128),jnp.float16)\n"
        "try:\n"
        "  jax.block_until_ready(f.fa_fwd_attn_length(q,q,q,3))\n"
        "except Exception as e:\n"
        "  assert '算子库未加载' in str(e), str(e)\n"
        "  print('EXPECTED_LIBRARY_ERROR')\n"
        "else:\n"
        "  raise AssertionError('missing library returned outputs')\n")
    env = {**os.environ, "FA_LIB_PATH": "/missing/fa/vendor_library.so",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    result = subprocess.run([sys.executable, "-c", snippet], env=env,
                            capture_output=True, text=True, timeout=120)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("EXPECTED_LIBRARY_ERROR", result.stdout)

  def test_unsupported_paths_raise(self):
    q = jnp.zeros((1, 64, 2, 64), jnp.float16)
    with self.assertRaisesRegex(ValueError, "softcap"):
      # Unsupported features must fail before reaching the kernel.
      self.fa.fa_fwd(q, q, q, softcap=50.0)
    with self.assertRaisesRegex(ValueError, "alibi"):
      self.fa.fa_fwd(q, q, q, alibi_slopes=jnp.zeros((2,), jnp.float32))
    with self.assertRaisesRegex(ValueError, "alibi"):
      self.fa.fa_fwd_custom(q, q, q, alibi_slopes=jnp.zeros((2,), jnp.float32))
    with self.assertRaisesRegex(NotImplementedError, "attn_mask"):
      self.fa.fa_fwd_attn_mask(q, q, q, jnp.zeros((64, 64), jnp.float16))
    with self.assertRaisesRegex(NotImplementedError, "padding_mask"):
      self.fa.fa_fwd_padding_mask(q, q, q, jnp.zeros((1, 64), jnp.bool_))
    # varlen 反向的内核路径已可用, 不再是守卫用例; 这里只保留未知 mode 的报错,
    # 两条路径的数值一致性由 test_varlen_forward_and_backward 验收。
    tq, hv, hkv, dv_ = 128, 2, 1, 64
    qv3 = jnp.zeros((tq, hv, dv_), jnp.float16)
    kv3 = jnp.zeros((tq, hkv, dv_), jnp.float16)
    lse3 = jnp.zeros((hv, tq), jnp.float32)
    cus = jnp.asarray(np.asarray([0, 64, 128], np.int32))
    with self.assertRaisesRegex(ValueError, "未知 mode"):
      self.fa.fa_varlen_bwd(qv3, kv3, kv3, qv3, qv3, lse3, cus, cus, 64, 64,
                            mode="whatever")
    with self.assertRaisesRegex(NotImplementedError, "INT8|int8"):
      q8 = jnp.zeros((1, 1, 2, 128), jnp.int8)
      kc8 = jnp.zeros((2, 128, 1, 128), jnp.int8)
      vc8 = jnp.zeros((2, 128, 1, 128), jnp.int8)
      bt = jnp.zeros((1, 2), jnp.int32)
      cs = jnp.asarray(np.asarray([128], np.int32))
      self.fa.fa_fwd_kvcache(q8, kc8, vc8, bt, cs, 128, _int8=True)
    with self.assertRaisesRegex(NotImplementedError, "mla_prefix"):
      qp = jnp.zeros((1, 2, 576), jnp.float16)
      qv = jnp.zeros((1, 2, 512), jnp.float16)
      kp = jnp.zeros((2, 128, 1, 576), jnp.float16)
      vp = jnp.zeros((2, 128, 1, 512), jnp.float16)
      cu = jnp.asarray(np.asarray([0, 1], np.int32))
      self.fa.fa_mla_prefix(qp, qv, kp, vp, bt, cs, cu, cu, 64)

  def _mesh2(self):
    devs = jax.devices()
    if len(devs) < 2:
      self.skipTest(f"需要 >=2 个设备验证 head 分片 (本机 {len(devs)} 个)")
    return Mesh(np.array(devs[:2]), axis_names=("x",))

  def test_forward_head_sharding_matches_single_device(self):
    mesh = self._mesh2()
    b, s, h, d = 1, 128, 4, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(jnp.float16)
    k = (jax.random.normal(jax.random.key(1), (b, s, h, d)) * .3).astype(jnp.float16)
    v = (jax.random.normal(jax.random.key(2), (b, s, h, d)) * .3).astype(jnp.float16)
    ref_o, ref_lse = self.fa.fa_fwd(q, k, v, causal=False, softmax_scale=scale)
    sh = NamedSharding(mesh, PartitionSpec(None, None, "x", None))
    lse_sh = NamedSharding(mesh, PartitionSpec(None, "x", None))
    o, lse = jax.jit(
        lambda a, bb, c: self.fa.fa_fwd(a, bb, c, causal=False, softmax_scale=scale),
        in_shardings=(sh, sh, sh), out_shardings=(sh, lse_sh))(q, k, v)
    self.assertArraysAllClose(o, ref_o, rtol=2e-3, atol=2e-3, check_dtypes=False)
    self.assertArraysAllClose(lse, ref_lse, rtol=1e-4, atol=1e-4)

  def test_backward_head_sharding_matches_single_device(self):
    mesh = self._mesh2()
    b, s, h, d = 1, 128, 4, 64
    scale = d ** -0.5
    q = (jax.random.normal(jax.random.key(0), (b, s, h, d)) * .3).astype(jnp.float16)
    k = (jax.random.normal(jax.random.key(1), (b, s, h, d)) * .3).astype(jnp.float16)
    v = (jax.random.normal(jax.random.key(2), (b, s, h, d)) * .3).astype(jnp.float16)
    do = (jax.random.normal(jax.random.key(3), (b, s, h, d)) * .5).astype(jnp.float16)

    def loss(qq, kk, vv):
      oo, _ = self.fa.fa_fwd_custom(qq, kk, vv, causal=False, softmax_scale=scale)
      return (oo.astype(jnp.float32) * jnp.asarray(do, jnp.float32)).sum()

    ref = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    sh = NamedSharding(mesh, PartitionSpec(None, None, "x", None))
    got = jax.jit(jax.grad(loss, argnums=(0, 1, 2)),
                  in_shardings=(sh, sh, sh),
                  out_shardings=(sh, sh, sh))(q, k, v)
    for a, b_ in zip(got, ref):
      self.assertArraysAllClose(a, b_, rtol=3e-3, atol=3e-3, check_dtypes=False)


if __name__ == "__main__":
  absltest.main()
