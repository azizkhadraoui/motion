#!/usr/bin/env python
# Which du/dt is right? meanflow_train.py's startup check found torch.func.jvp and a float32 central
# difference (eps 1e-3) disagreeing by ~20%. Here both are computed in FLOAT64 on the warm-started
# MeanFlow net, with the finite difference swept over eps. In float64 the difference converges to the
# true derivative as eps shrinks (until ~1e-6), so whichever method it converges to is correct.
# The float32 difference at the training eps is also reported against that reference.

import os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, new_mfnet

M = load_main("jvp-diag"); DEVICE = M.DEVICE
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
ck = torch.load(os.path.join(M.CK, "latent_best.pt"), map_location=DEVICE, weights_only=False)
net = new_mfnet(M); net.load_state_dict(ck["state"], strict=False)
for m in net.modules():
    if isinstance(m, nn.Dropout): m.p = 0.0
    if isinstance(m, nn.MultiheadAttention): m.dropout = 0.0
net.train()
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    MATH = lambda: sdpa_kernel(SDPBackend.MATH)
except Exception:
    MATH = lambda: torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

B = 8; idx = list(range(B))
L = torch.tensor([int(M.train_lens[i]) for i in idx], device=DEVICE)
ts = torch.tensor(M.tr_seq[idx], device=DEVICE); tm = torch.tensor(M.tr_mask[idx], device=DEVICE)
tp = torch.tensor(M.tr_pool[idx], device=DEVICE)
g = torch.Generator(device="cpu").manual_seed(0)
z = torch.randn(B, M.T_LAT, M.RVQ_CODE_DIM, generator=g).to(DEVICE); tz = torch.randn(z.shape, generator=g).to(DEVICE)
t = torch.rand(B, generator=g).to(DEVICE) * 0.6 + 0.1; s = (t + 0.2).clamp(max=1.0)


def run(dtype):
    n = net.to(dtype); c = (ts.to(dtype), tm, tp.to(dtype), L)
    zz, tzz, tt, ss = z.to(dtype), tz.to(dtype), t.to(dtype), s.to(dtype)
    f = lambda a, b, d: n(a, b, d, *c)
    with torch.no_grad(), MATH():
        jf = torch.func.jvp(f, (zz, tt, ss), (tzz, torch.ones_like(tt), torch.zeros_like(ss)))[1]
        fd = {}
        for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            fd[eps] = (f(zz + eps * tzz, tt + eps, ss) - f(zz - eps * tzz, tt - eps, ss)) / (2 * eps)
        # split: z-direction only and t-direction only, to localise any disagreement
        jz = torch.func.jvp(f, (zz, tt, ss), (tzz, torch.zeros_like(tt), torch.zeros_like(ss)))[1]
        jt = torch.func.jvp(f, (zz, tt, ss), (torch.zeros_like(tzz), torch.ones_like(tt), torch.zeros_like(ss)))[1]
        e = 1e-4 if dtype == torch.float64 else 1e-3
        fz = (f(zz + e * tzz, tt, ss) - f(zz - e * tzz, tt, ss)) / (2 * e)
        ft = (f(zz, tt + e, ss) - f(zz, tt - e, ss)) / (2 * e)
    return jf, fd, (jz, fz), (jt, ft)


rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))
jf64, fd64, (jz64, fz64), (jt64, ft64) = run(torch.float64)
print("\nFLOAT64")
print(f"  |jvp| = {jf64.norm().item():.4f}")
for eps, d in fd64.items():
    print(f"  eps={eps:<7g} rel(fd, func-jvp) = {rel(d, jf64):.3e}   rel(fd, fd@1e-5) = {rel(d, fd64[1e-5]):.3e}")
print(f"  z-direction only: rel(fd, jvp) = {rel(fz64, jz64):.3e}   |jvp_z| = {jz64.norm().item():.4f}")
print(f"  t-direction only: rel(fd, jvp) = {rel(ft64, jt64):.3e}   |jvp_t| = {jt64.norm().item():.4f}")
jf32, fd32, (jz32, fz32), (jt32, ft32) = run(torch.float32)
print("\nFLOAT32 (training precision), against the float64 references")
print(f"  func-jvp fp32 vs func-jvp fp64: {rel(jf32, jf64):.3e}")
print(f"  fd@1e-3 fp32 vs func-jvp fp64:  {rel(fd32[1e-3], jf64):.3e}")
print(f"  fd@1e-3 fp32 vs fd@1e-5 fp64:   {rel(fd32[1e-3], fd64[1e-5]):.3e}")
print(f"  z-only fd vs jvp (fp32): {rel(fz32, jz32):.3e}    t-only fd vs jvp (fp32): {rel(ft32, jt32):.3e}")
conv = rel(fd64[1e-5], jf64)
print("\nVERDICT: " + ("func-jvp is CORRECT (float64 finite difference converges to it); prefer MF_JVP=func."
                       if conv < 1e-3 else
                       "func-jvp DISAGREES with the converged float64 finite difference; keep the fd fallback."))
