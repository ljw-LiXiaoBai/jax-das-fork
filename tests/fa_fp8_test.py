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

"""FP8 前向与分页注意力测试。

架构不支持时跳过数值用例；守卫通过或 FA_DEBUG=5 空跑通过均不代表数值验收。
"""
import os
import subprocess
import sys

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


def _quant(x, descale):
  """按 descale 量化到 e4m3: x / descale -> e4m3 (与参考的 div 语义对应)."""
  return (jnp.asarray(x, jnp.float32)
          / jnp.asarray(descale, jnp.float32)).astype(jnp.float8_e4m3fn)


def _dequant(x):
  """fp8 -> fp32 (供参考实现; 量化误差按 e4m3 的 2^-3 尾数量级核验)."""
  return np.asarray(x.astype(jnp.float32), np.float32)


class FaFp8Test(jtu.JaxTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.fa = _load_fa()

  def _arch_std_ok(self) -> bool:
    """标准 FP8 门槛: getArch() >= 938."""
    return bool(self.fa._fp8_arch_ok())

  def _arch_pa_ok(self) -> bool:
    """FP8 PagedAttention 门槛: gfx92a(930) 或 >= 938."""
    return bool(self.fa._fp8_pa_arch_ok())

  def _dense_fp8_ref(self, q, k, v, qd, kd, vd, scale, causal, mul):
    """fp32 参考: dequant = x*descale (mul) 或 x/descale (div); 返回 (o, lse)."""
    def de(x, d):
      d = np.asarray(d, np.float32)[:, None, :, None]
      return (x * d if mul else x / d).astype(np.float32)
    qf, kf, vf = de(q, qd), de(k, kd), de(v, vd)
    g = qf.shape[2] // kf.shape[2]
    kf2 = np.repeat(kf, g, 2)
    vf2 = np.repeat(vf, g, 2)
    sc = np.einsum("bshd,blhd->bhsl", qf, kf2) * scale
    if causal:
      m = np.arange(sc.shape[-1])
      sc = np.where(m[None, :] > m[:, None], -np.inf, sc)
    mx = sc.max(-1, keepdims=True)
    p = np.exp(sc - mx)
    p = p / p.sum(-1, keepdims=True)
    o = np.einsum("bhsl,blhd->bshd", p, vf2)
    lse = (mx + np.log(np.exp(sc - mx).sum(-1, keepdims=True))
           ).squeeze(-1).transpose(0, 2, 1)
    return o, lse

  def _pa_fp8_ref(self, q, kc, vc, bt, seqlens, qd, kd, vd, scale, causal, mul):
    """分页参考: gather 后 dequant (右对齐 causal)。"""
    def de(x, d, heads):
      d = np.asarray(d, np.float32)
      if heads == 1:
        return (x * d[:, None, None, None] if mul else x / d[:, None, None, None]
                ).astype(np.float32)
      return (x * d[:, None, :, None] if mul else x / d[:, None, :, None]
              ).astype(np.float32)
    page, hk = kc.shape[1], kc.shape[2]
    dv = vc.shape[3]
    g = q.shape[2] // hk
    outs = []
    for i in range(len(seqlens)):
      L = int(seqlens[i])
      idx = np.asarray(bt[i, : (L + page - 1) // page])
      k_i = np.asarray(kc[idx]).reshape(-1, hk, kc.shape[3])[:L]
      v_i = np.asarray(vc[idx]).reshape(-1, hk, dv)[:L]
      q_i = np.asarray(q[i], np.float32) * np.asarray(qd[i], np.float32)[None, :, None]
      if not mul:
        q_i = np.asarray(q[i], np.float32) / np.asarray(qd[i], np.float32)[None, :, None]
      k_i = de(k_i, kd[i], 1)
      v_i = de(v_i, vd[i], 1)
      k_e = np.repeat(k_i, g, 1)
      v_e = np.repeat(v_i, g, 1)
      sc = np.einsum("shd,lhd->hsl", q_i, np.asarray(k_e, np.float32)) * scale
      if causal:
        kv = np.arange(L)
        pos = np.arange(L - q.shape[1], L)
        sc = np.where(kv[None, :] <= pos[:, None], sc, -np.inf)
      mx = sc.max(-1, keepdims=True)
      p = np.exp(sc - mx)
      p = p / p.sum(-1, keepdims=True)
      outs.append(np.einsum("hsl,lhd->shd", p, np.asarray(v_e, np.float32)))
    return np.stack(outs)

  def _pipeline_selfcheck(self, snippet):
    """FA_DEBUG=5 (内核空跑) 下参数管线不崩。返回是否输出 PIPE_OK。

    子进程的接入层 import 必须与本测试一致: 树内安装态走 jax._src.cudnn;
    本地验证态 (插件 .so 手工放置) 走同目录直导。"""
    env = {**os.environ, "FA_DEBUG": "5"}
    probe = ("import numpy as np, jax, jax.numpy as jnp\n"
             "from jax._src.cudnn import fa_attention as F\n"
             "F._register()\n")
    r = subprocess.run([sys.executable, "-c", probe + snippet],
                       capture_output=True, text=True, env=env, timeout=120)
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    return "PIPE_OK" in r.stdout

  @parameterized.named_parameters(
      ("d128_gqa_bf16out", 2, 128, 4, 2, 128, False, jnp.bfloat16),
      ("d128_causal_fp16out", 2, 128, 4, 2, 128, True, jnp.float16),
      ("d128_mha", 2, 64, 2, 2, 128, False, jnp.bfloat16),
      ("d256", 1, 64, 2, 1, 256, False, jnp.bfloat16),
  )
  def test_fp8_forward_numeric(self, b, s, h, hk, d, causal, out_dtype):
    if not self._arch_std_ok():
      self.skipTest(f"本卡 arch={self.fa._arch_raw()} 不支持标准 FP8 (需 >=938)")
    rng = np.random.default_rng(0)
    scale = d ** -0.5
    qd = rng.uniform(0.5, 2.0, (b, h)).astype(np.float32)
    kd = rng.uniform(0.5, 2.0, (b, hk)).astype(np.float32)
    vd = rng.uniform(0.5, 2.0, (b, hk)).astype(np.float32)
    q = _quant(rng.normal(0, .3, (b, s, h, d)), qd[:, None, :, None])
    k = _quant(rng.normal(0, .3, (b, s, hk, d)), kd[:, None, :, None])
    v = _quant(rng.normal(0, .3, (b, s, hk, d)), vd[:, None, :, None])
    o, lse = self.fa.fa_fwd_fp8(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v),
                                jnp.asarray(qd), jnp.asarray(kd), jnp.asarray(vd),
                                causal=causal, softmax_scale=scale,
                                out_dtype=out_dtype)
    o_np = np.asarray(o, np.float32)
    qf, kf, vf = _dequant(q), _dequant(k), _dequant(v)
    # 分别按乘、除 descale 计算 fp32 参考并取较小误差；此处不锁定缩放约定。
    errs = {}
    for tag, mul in (("mul", True), ("div", False)):
      r_o, _ = self._dense_fp8_ref(qf, kf, vf, qd, kd, vd, scale, causal, mul)
      errs[tag] = float(np.abs(o_np - r_o).max())
    best = min(errs, key=errs.get)
    self.assertLess(errs[best], 2e-2,
                    f"descale 语义={best}: {errs}")
    self.assertEqual(o.dtype, out_dtype)

  def test_fp8_forward_guard(self):
    if self._arch_std_ok():
      self.skipTest("本卡支持标准 FP8, 守卫不触发")
    d = 128
    q = jnp.zeros((1, 8, 2, d), jnp.float32).astype(jnp.float8_e4m3fn)
    sc = jnp.ones((1, 2), jnp.float32)
    with self.assertRaises(ValueError):
      self.fa.fa_fwd_fp8(q, q, q, sc, sc, sc, softmax_scale=d ** -0.5)

  def test_fp8_forward_pipeline(self):
    if self._arch_std_ok():
      self.skipTest("本卡支持标准 FP8, 空跑自检无意义")
    snippet = (
        "F._fp8_arch_ok._v=True;"  # 绕过守卫, 只验管线 (import 由 probe 提供)
        "d=128; q=jnp.zeros((1,8,2,d),jnp.float32).astype(jnp.float8_e4m3fn);"
        "sc=jnp.ones((1,2),jnp.float32);"
        "o,lse=F.fa_fwd_fp8(q,q,q,sc,sc,sc,softmax_scale=d**-0.5);"
        "print('PIPE_OK', np.asarray(o).shape)")
    self.assertTrue(self._pipeline_selfcheck(snippet),
                    "FA_DEBUG=5 空跑下 FP8 参数管线应不崩")

  @parameterized.named_parameters(
      ("decode_gqa", 2, 4, 2, 1, 128, [512, 300], False, jnp.bfloat16),
      ("mtp_causal", 2, 4, 2, 4, 128, [512, 300], True, jnp.bfloat16),
      ("page64", 2, 4, 1, 1, 64, [300, 200], False, jnp.bfloat16),
      ("fp16out", 2, 4, 2, 1, 128, [256, 128], False, jnp.float16),
  )
  def test_fp8_pa_numeric(self, b, h, hk, sq, page, seqlens, causal, out_dtype):
    if not self._arch_pa_ok():
      self.skipTest(f"本卡 arch={self.fa._arch_raw()} 不支持 FP8 PA "
                    "(需 gfx92a(930) 或 >=938)")
    rng = np.random.default_rng(0)
    d = 128
    scale = d ** -0.5
    max_sk = max(seqlens)
    max_blk = (max_sk + page - 1) // page
    nb = b * max_blk + 4
    perm = rng.permutation(nb)
    bt = np.zeros((b, max_blk), np.int32)
    for i, L in enumerate(seqlens):
      need = (int(L) + page - 1) // page
      bt[i, :need] = perm[i * max_blk:i * max_blk + need] % nb
    qd = rng.uniform(0.5, 2.0, (b, h)).astype(np.float32)
    kd = rng.uniform(0.5, 2.0, (b, hk)).astype(np.float32)
    vd = rng.uniform(0.5, 2.0, (b, hk)).astype(np.float32)
    qf = rng.normal(0, .3, (b, sq, h, d)).astype(np.float32)
    kf = rng.normal(0, .3, (nb, page, hk, d)).astype(np.float32)
    vf = rng.normal(0, .3, (nb, page, hk, d)).astype(np.float32)
    q = _quant(qf, qd[:, None, :, None])
    kc = _quant(kf, kd[:, None, :, None])
    vc = _quant(vf, vd[:, None, :, None])
    o = self.fa.fa_fwd_kvcache(jnp.asarray(q), jnp.asarray(kc), jnp.asarray(vc),
                               jnp.asarray(bt), jnp.asarray(np.asarray(seqlens, np.int32)),
                               max_sk, causal=causal, softmax_scale=scale,
                               q_descale=jnp.asarray(qd), k_descale=jnp.asarray(kd),
                               v_descale=jnp.asarray(vd), out_dtype=out_dtype)
    o_np = np.asarray(o, np.float32)
    errs = {}
    for tag, mul in (("mul", True), ("div", False)):
      r = self._pa_fp8_ref(_dequant(q), _dequant(kc), _dequant(vc), bt, seqlens,
                           qd, kd, vd, scale, causal, mul)
      errs[tag] = float(np.abs(o_np - r).max())
    best = min(errs, key=errs.get)
    self.assertLess(errs[best], 2e-2, f"descale 语义={best}: {errs}")
    self.assertEqual(o.dtype, out_dtype)

  def test_fp8_pa_guard(self):
    if self._arch_pa_ok():
      self.skipTest("本卡支持 FP8 PA, 守卫不触发")
    d = 128
    q = jnp.zeros((1, 1, 2, d), jnp.float32).astype(jnp.float8_e4m3fn)
    kc = jnp.zeros((3, 128, 1, d), jnp.float32).astype(jnp.float8_e4m3fn)
    bt = jnp.asarray([[0, 1]], jnp.int32)
    sk = jnp.asarray([256], jnp.int32)
    sc = jnp.ones((1, 2), jnp.float32)
    with self.assertRaises(ValueError):
      self.fa.fa_fwd_kvcache(q, kc, kc, bt, sk, 256, q_descale=sc,
                             k_descale=sc, v_descale=sc)

  def test_fp8_pa_pipeline(self):
    if self._arch_pa_ok():
      self.skipTest("本卡支持 FP8 PA, 空跑自检无意义")
    snippet = (
        "F._fp8_pa_arch_ok._v=True;"  # 绕过守卫, 只验管线 (import 由 probe 提供)
        "b,h,hk,d,page,L=2,4,2,128,128,256;"
        "q=jnp.zeros((b,1,h,d),jnp.float32).astype(jnp.float8_e4m3fn);"
        "kc=jnp.zeros((3,page,hk,d),jnp.float32).astype(jnp.float8_e4m3fn);"
        "bt=jnp.asarray([[0,1],[0,1]],jnp.int32);"
        "sk=jnp.asarray([L,L],jnp.int32);"
        "sc=jnp.ones((b,h),jnp.float32); skd=jnp.ones((b,hk),jnp.float32);"
        "o=F.fa_fwd_kvcache(q,kc,kc,bt,sk,L,q_descale=sc,"
        "k_descale=skd,v_descale=skd);"
        "print('PIPE_OK', np.asarray(o).dtype, np.asarray(o).shape)")
    self.assertTrue(self._pipeline_selfcheck(snippet),
                    "FA_DEBUG=5 空跑下 FP8 PA 参数管线应不崩")


if __name__ == "__main__":
  absltest.main()
