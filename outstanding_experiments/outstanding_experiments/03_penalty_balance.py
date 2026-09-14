#!/usr/bin/env python
"""
03_penalty_balance.py — does the bone term actually steer the guidance?

WHY THIS EXISTS
  diff_penalty (line 554) returns PEN_BONE*Lb + PEN_FOOT*Lf with PEN_BONE=0.5,
  PEN_FOOT=1.0. But Lb is a SQUARED bone deviation (metres squared) while Lf is an
  UNSQUARED mean contact-frame foot speed (metres). The two are added across different
  units, so the weights do not mean what they appear to.

  One correction to an earlier note of mine: the guidance branch normalises the gradient
  to unit norm before using it,

      g = g / ||g||;   v = v - gwt * g * ||v||

  so the MAGNITUDE of the penalty is irrelevant to the step size. What matters is the
  DIRECTION of g. The question is therefore not "which term is bigger" but "how much of
  the gradient direction does the bone term contribute". That is what this script
  measures, in the latent space where the guidance actually acts.

WHAT IT REPORTS, at several points along a real sampling trajectory:
  ||g_bone||, ||g_foot||   gradient norms w.r.t. the latent state, per term
  ratio                    ||g_bone|| / ||g_foot||
  cos(g, g_bone)           how far the applied direction aligns with the bone term
  cos(g_bone, g_foot)      whether the two terms even agree
  Lb, Lf                   the raw loss values, for the record

HOW TO READ IT
  If ||g_bone|| / ||g_foot|| is small and cos(g, g_bone) is near zero, the guidance is
  effectively foot-skate guidance and the bone term is a rounding error on the
  direction. That would explain the rising BLE under direct-space guidance that the
  report flags as odd, with no bug involved -- and it would mean the soft-guidance
  experiment is not a clean test of BONE guidance in either branch, which matters
  because that experiment now carries the in-trajectory argument.

  If cos(g, g_bone) is substantial, the anomaly is not explained by the weighting and
  something else is going on.

  Run it on BOTH pipelines: the latent branch differentiates through the decoder, the
  direct branch does not, so the balance can differ.

USAGE
  WORK_DIR=$WORK/runs python 03_penalty_balance.py
  env: PB_BASE=latent|direct   PB_N=32   PB_TS=0.1,0.3,0.5,0.7,0.9
"""
import os, sys, json, importlib.util
os.environ.setdefault("VARIANT", "eval")
os.environ["USE_WANDB"] = "0"
os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch

MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN)
M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[penbal] importing {MAIN} ...", flush=True)
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP":
        raise
    print("[penbal] models loaded.")

DEVICE = M.DEVICE; rvq = M.rvq
EI, EJ = M.EI, M.EJ
rest_len = M.rest_len
recover_from_ric = M.recover_from_ric
lengths_to_mask = M.lengths_to_mask
embed_text = M.embed_text
FOOT_JOINTS = M.FOOT_JOINTS
PROJ_FOOT_H = M.PROJ_FOOT_H
PEN_BONE, PEN_FOOT = M.PEN_BONE, M.PEN_FOOT
MAXLEN = M.MAX_MOTION_LEN
_cfg, _timesteps = M._cfg, M._timesteps

BASE = os.environ.get("PB_BASE", "latent")
N = int(os.environ.get("PB_N", "32"))
TS = [float(x) for x in os.environ.get("PB_TS", "0.1,0.3,0.5,0.7,0.9").split(",")]
WORK_DIR = os.environ.get("WORK_DIR", ".")
IS_LAT = (BASE == "latent")

net = M.load_net(BASE, IS_LAT)
Z_MEAN, Z_STD = M.z_mean_t, M.z_std_t


def pen_terms(mn, L):
    """diff_penalty split into its two weighted halves. Mirrors lines 550-557 exactly."""
    raw = mn * M.std_t + M.mean_t
    fm = lengths_to_mask(L, MAXLEN).float()
    J = recover_from_ric(raw)
    bone = (J[:, :, EI, :] - J[:, :, EJ, :]).norm(dim=-1)
    Lb = (((bone - rest_len) ** 2).mean(-1) * fm).sum() / (fm.sum() + 1e-6)
    fj = J[:, :, FOOT_JOINTS, :]
    vel = (fj[:, 1:, :, [0, 2]] - fj[:, :-1, :, [0, 2]]).norm(dim=-1)
    ht = fj[:, 1:, :, 1]
    cw = (ht < PROJ_FOOT_H).float() * fm[:, 1:].unsqueeze(-1)
    Lf = (vel * cw).sum() / (cw.sum() + 1e-6)
    return PEN_BONE * Lb, PEN_FOOT * Lf


def flat_norm(g):
    return g.flatten(1).norm(dim=1)


def cos_rows(a, b):
    a = a.flatten(1); b = b.flatten(1)
    return ((a * b).sum(1) / (a.norm(dim=1).clamp_min(1e-12) *
                              b.norm(dim=1).clamp_min(1e-12)))


# ---- build one real trajectory and probe it at several t --------------------
rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
tseq, tmask, tpool = embed_text(caps)
ts_, tm_, tp_ = (torch.tensor(tseq, device=DEVICE), torch.tensor(tmask, device=DEVICE),
                 torch.tensor(tpool, device=DEVICE))

torch.manual_seed(0)
B, cd, Tl = N, net.cd, net.Tlen
z = torch.randn(B, Tl, cd, device=DEVICE)
ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
nm = torch.ones(B, M.T5_MAXLEN, dtype=torch.bool, device=DEVICE)
npl = net.null_pool.unsqueeze(0).expand(B, -1)
grid = _timesteps(M.ODE_STEPS, "linear")
probe_at = sorted({int(round(t * M.ODE_STEPS)) for t in TS})

rows = []
print(f"\n{'='*100}")
print(f"PENALTY BALANCE — base={BASE}, {N} clips, gradients w.r.t. the latent state")
print(f"PEN_BONE={PEN_BONE}  PEN_FOOT={PEN_FOOT}   (Lb is squared, m^2; Lf unsquared, m)")
print("=" * 100, flush=True)
print(f" {'t':>6}{'Lb*w':>12}{'Lf*w':>12}{'|g_bone|':>12}{'|g_foot|':>12}"
      f"{'ratio':>9}{'cos(g,gb)':>11}{'cos(gb,gf)':>12}")
print("-" * 100)

for i in range(M.ODE_STEPS):
    tval = float(grid[i]); dt = float(grid[i + 1] - grid[i])
    t = torch.full((B,), tval, device=DEVICE)
    if i in probe_at:
        with torch.enable_grad():
            zc = z.detach().requires_grad_(True)
            mnc = (rvq.decoder(zc * Z_STD + Z_MEAN) if IS_LAT else zc)
            pb, pf = pen_terms(mnc, lens)
            gb = torch.autograd.grad(pb, zc, retain_graph=True)[0]
            gf = torch.autograd.grad(pf, zc)[0]
        g = gb + gf
        nb_, nf_ = flat_norm(gb), flat_norm(gf)
        r = dict(t=tval, Lb=float(pb), Lf=float(pf),
                 g_bone=float(nb_.mean()), g_foot=float(nf_.mean()),
                 ratio=float((nb_ / nf_.clamp_min(1e-12)).mean()),
                 cos_g_gb=float(cos_rows(g, gb).mean()),
                 cos_gb_gf=float(cos_rows(gb, gf).mean()))
        rows.append(r)
        print(f" {r['t']:>6.2f}{r['Lb']:>12.3e}{r['Lf']:>12.3e}{r['g_bone']:>12.3e}"
              f"{r['g_foot']:>12.3e}{r['ratio']:>9.4f}{r['cos_g_gb']:>11.4f}"
              f"{r['cos_gb_gf']:>12.4f}", flush=True)
    with torch.no_grad():
        v = _cfg(net, z, t, ts_, tm_, tp_, lens, ns, nm, npl, M.GUIDANCE, 0.0)
        z = z + dt * v

print("=" * 100)
mr = float(np.mean([r["ratio"] for r in rows]))
mc = float(np.mean([r["cos_g_gb"] for r in rows]))
print(f"\n  mean ||g_bone||/||g_foot|| = {mr:.4f}      mean cos(g, g_bone) = {mc:.4f}")
print("""
READING
  The guidance branch normalises g to unit norm before applying it, so only the
  DIRECTION of g matters. cos(g, g_bone) is therefore the quantity of interest: it is
  the fraction of the applied direction that the bone term is responsible for.

  cos(g, g_bone) near zero  -> the applied direction is essentially the foot gradient.
                               The sweep is foot-skate guidance with a vestigial bone
                               term, which explains BLE rising under guidance without
                               any bug, and means the experiment is not a clean test of
                               bone guidance. Say so in the paper and, if a clean test is
                               wanted, rerun with PEN_FOOT=0.
  cos(g, g_bone) substantial-> the weighting does not explain the anomaly; look further.

  cos(g_bone, g_foot) negative is worth noting separately: it would mean the two terms
  actively oppose each other, so satisfying feet costs bone validity along the
  trajectory, which is the differentiable echo of the O3 result.
""")

dst = os.path.join(WORK_DIR, f"penalty_balance_{BASE}.json")
json.dump(dict(base=BASE, n=N, pen_bone=PEN_BONE, pen_foot=PEN_FOOT,
               mean_ratio=mr, mean_cos_g_gb=mc, rows=rows),
          open(dst, "w"), indent=2)
print(f"raw results -> {dst}")
