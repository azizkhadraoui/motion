#!/usr/bin/env python
# 4.1 — PROJFLOW-CLASS LINEAR CONSTRAINT, LATENT vs DIRECT.
#
# THE QUESTION. Our negative result rests on one constraint, bone length, which is nonlinear and of our
# own choosing. ProjFlow (CVPR 2026) enforces LINEAR constraints y = A x by closed-form endpoint
# projection and succeeds. Is our collapse about PLACEMENT (the decoder in the loop) or about our
# CONSTRAINT CLASS? This script runs ProjFlow's operator, on a linear constraint, on both models.
#
# THE CONSTRAINT. Keyframe joint-position control with targets from the ground-truth clip of the same
# caption: at K keyframes spread over the clip, the root height and the root-relative positions (RIC
# dims) of a set of joints (default: both feet and both wrists) must equal the ground truth. In the
# normalised 263-d representation these coordinates are an affine function of the state, so y = A x
# holds exactly with A a coordinate selector.
#
# THE OPERATOR (ProjFlow, hard constraint Sigma -> 0):
#       dx1* = R^-1 A^T (A R^-1 A^T)^-1 (y - A x1_hat),      x1_hat = x_t + (1 - t) v(x_t, t)
# R = I gives the Euclidean projection (replace the keyframe coordinates). R = I + w L_T, with L_T the
# temporal path-graph Laplacian, spreads each correction smoothly over neighbouring frames of the same
# coordinate (ProjFlow's point that the correction should be coherently distributed). The corrected
# endpoint defines the corrected velocity v' = (x1_hat + dx1* - x_t) / (1 - t), which is integrated as
# usual; on the last step dt = 1 - t, so the state lands ON the corrected endpoint.
#
#   CDFM : A acts on the state directly. The final step lands exactly on the constraint set.
#   LFM  : A lives in joint space, so every step is decode -> project -> re-encode (re-standardised):
#            'roundtrip' : dz1 = E(D(z1_hat) + dx) - z1_hat              (the literal operation)
#            'delta-enc' : dz1 = E(D(z1_hat) + dx) - E(D(z1_hat))        (cancels the round-trip bias;
#                          the most favourable reading of the latent arm, included so a reviewer cannot
#                          say we handicapped it)
#
# PREDICTION (written before the numbers): works in CDFM (residual ~0 at FID near unconstrained),
# does not survive the round trip in LFM (residual well above 0 or FID collapse).
# Either outcome is reportable: if linear constraints DO survive the round trip where bone length does
# not, that is a finding about which constraint classes a decoder preserves.
#
#   sbatch run_linear_constraint.sh          (env: LC_N, LC_REPS, LC_K, LC_JOINTS, LC_W, LC_BASES)

import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, ClipSet, run_protocol, decode, encode, null_cond, fmt

M = load_main("linear")
DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; MAXLEN = M.MAX_MOTION_LEN

N = int(os.environ.get("LC_N", "512"))
REPS = int(os.environ.get("LC_REPS", "3"))
K = int(os.environ.get("LC_K", "5"))
JOINTS = [int(j) for j in os.environ.get("LC_JOINTS", "10,11,20,21").split(",")]
W_GRID = [float(w) for w in os.environ.get("LC_W", "0,20").split(",")]
BASES = os.environ.get("LC_BASES", "direct,latent").split(",")
WINDOWS = [float(w) for w in os.environ.get("LC_WINDOWS", "1.0,0.1").split(",")]
WORK_DIR = os.environ.get("WORK_DIR", ".")

DIMS = [3] + [4 + (j - 1) * 3 + c for j in JOINTS for c in range(3)]   # root height + RIC of joints
DIMS_T = torch.tensor(DIMS, device=DEVICE)
STD_D = M.std_t[DIMS_T]                                                # raw units per normalised unit
print(f"[linear] constraint: {K} keyframes x {len(DIMS)} coords (root height + RIC of joints {JOINTS}) "
      f"= {K*len(DIMS)} scalar equalities per clip; R weights w in {W_GRID}")


def temporal_G(w, T=MAXLEN):
    """(I + w L_T)^-1 for the path graph over T frames. w = 0 -> identity (Euclidean projection)."""
    if w == 0: return torch.eye(T, device=DEVICE, dtype=torch.float64)
    Lp = torch.zeros(T, T, dtype=torch.float64)
    i = torch.arange(T - 1)
    Lp[i, i] += 1; Lp[i + 1, i + 1] += 1; Lp[i, i + 1] -= 1; Lp[i + 1, i] -= 1
    return torch.linalg.inv(torch.eye(T, dtype=torch.float64) + w * Lp).to(DEVICE)


G_CACHE = {w: temporal_G(w) for w in W_GRID}


class LinCons:
    """y = A x on one batch: A selects (keyframe, coordinate) entries of the normalised state."""
    def __init__(self, clips, s, e, L, w):
        B = e - s
        self.y = clips.real_norm(s, e)                                   # targets = GT in normalised coords
        kf = np.stack([np.round(np.linspace(0, int(l) - 1, K)).astype(int) for l in L.tolist()])
        self.kf = torch.tensor(kf, device=DEVICE)                        # (B,K)
        G = G_CACHE[w]; P = []
        for b in range(B):
            k = self.kf[b]
            P.append((G[:, k] @ torch.linalg.inv(G[k][:, k])).float())  # R^-1 A^T (A R^-1 A^T)^-1, (T,K)
        self.P = torch.stack(P)                                          # (B,T,K)
        self.bidx = torch.arange(B, device=DEVICE)[:, None]

    def _at(self, x):   # (B,K,D) entries of x at the constrained positions
        return x[self.bidx, self.kf][:, :, DIMS_T]

    def project(self, x):
        r = self._at(self.y) - self._at(x)                               # (B,K,D)
        dx = torch.zeros_like(x)
        dx[:, :, DIMS_T] = torch.einsum("btk,bkd->btd", self.P, r)
        return x + dx

    def residual(self, x):
        """Per-clip mean |Ax - y| in RAW units (metres for every selected coordinate)."""
        return ((self._at(x) - self._at(self.y)).abs() * STD_D).mean((1, 2)).cpu().numpy()


@torch.no_grad()
def sample_lc(net, is_lat, ts, tm, tp, L, seed, cons, mode, window=1.0, enc="roundtrip", final=False,
              n=ODE, guidance=GUID):
    torch.manual_seed(seed)
    B = tp.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns, nm, npl = null_cond(net, B, M)
    start = int(round(n * (1.0 - window)))
    grid = M._timesteps(n, "linear")
    for i in range(n):
        t = float(grid[i]); dt = float(grid[i + 1] - grid[i])
        v = M._cfg(net, z, torch.full((B,), t, device=DEVICE), ts, tm, tp, L, ns, nm, npl, guidance, 0.0)
        if mode == "inode" and i >= start:
            x1 = z + (1.0 - t) * v                                       # Tweedie endpoint estimate
            if is_lat:
                m1 = decode(M, x1, True); z1p = encode(M, cons.project(m1), True)
                d = z1p - (x1 if enc == "roundtrip" else encode(M, m1, True))
            else:
                d = cons.project(x1) - x1
            v = v + d / (1.0 - t)                                        # corrected velocity
        z = z + dt * v
    mn = decode(M, z, is_lat)
    pre = cons.residual(mn)
    if mode == "terminal" or final:
        mn = cons.project(mn)
    return mn, pre


print("\n" + "=" * 118)
print(f"LINEAR (ProjFlow-class) CONSTRAINT PLACEMENT — {N} clips x {REPS} reps ({N}-clip protocol)")
print("=" * 118, flush=True)
clips = ClipSet(M, N)
results = {}
for tag in BASES:
    is_lat = tag.startswith("latent")
    if not M._have(tag): print(f"[linear] {tag} checkpoint missing, skipped"); continue
    net = M.load_net(tag, is_lat)
    print(f"\n--- base {tag} (latent={is_lat}) ---", flush=True)
    configs = [("unconstrained", dict(mode="none"), 0.0)]
    for w in W_GRID:
        rn = "R=I" if w == 0 else f"R=I+{w:g}L_T"
        configs.append((f"terminal post-decode [{rn}]", dict(mode="terminal"), w))
        encs = ["roundtrip", "delta-enc"] if is_lat else ["roundtrip"]
        for win in WINDOWS:
            for enc in encs:
                nm_ = f"in-ODE {int(win*100)}% {enc if is_lat else ''} [{rn}]".replace("  ", " ")
                configs.append((nm_, dict(mode="inode", window=win, enc=enc), w))
        configs.append((f"in-ODE 100% + terminal [{rn}]", dict(mode="inode", window=1.0, final=True), w))
    for label, kw, w in configs:
        def gen(s, e, ts, tm, tp, L, seed, kw=kw, w=w):
            cons = LinCons(clips, s, e, L, w)
            mn, pre = sample_lc(net, is_lat, ts, tm, tp, L, seed, cons, **kw)
            return mn, dict(RES=cons.residual(mn), RES_pre=pre)
        r = run_protocol(M, clips, gen, REPS, label=f"{tag} | {label}", extra_keys=("RES", "RES_pre"))
        print(f"      residual |Ax-y| = {fmt(*r['RES'], p=5)} m   (before any terminal projection "
              f"{fmt(*r['RES_pre'], p=5)} m)", flush=True)
        results[f"{tag} | {label}"] = dict(base=tag, label=label, w=w, **kw, **{k: v for k, v in r.items()})

# ------------------------------------------------------------------ summary + pre-committed verdicts
print("\n" + "=" * 118)
print(f" {'base | configuration':<52}{'FID':>16}{'R@3':>14}{'residual (m)':>18}{'BLE':>12}")
print("-" * 118)
for k, r in results.items():
    print(f" {k:<52}{fmt(*r['FID']):>16}{fmt(*r['R3'], p=3):>14}{fmt(*r['RES'], p=5):>18}{r['BLE'][0]:>12.5f}")
print("=" * 118)

print("\nREADING (thresholds fixed before the run)")
print("  'satisfies'  : final residual <= 5% of the unconstrained residual")
print("  'no collapse': FID <= unconstrained FID + max(0.05, 25% of it)")
verdict = {}
for tag in BASES:
    unc = results.get(f"{tag} | unconstrained")
    if not unc: continue
    f0, r0 = unc["FID"][0], unc["RES"][0]
    print(f"\n  [{tag}] unconstrained: FID {f0:.4f}, natural residual {r0:.5f} m")
    for k, r in results.items():
        if r["base"] != tag or r["mode"] != "inode" or r.get("final"): continue
        sat = r["RES"][0] <= 0.05 * r0; ok = r["FID"][0] <= f0 + max(0.05, 0.25 * f0)
        v = "WORKS" if (sat and ok) else ("constraint not met" if not sat else "") + \
            (" + " if (not sat and not ok) else "") + ("FID collapse" if not ok else "")
        verdict[k] = v
        print(f"    {r['label']:<44} residual {r['RES'][0]:.5f} m ({100*r['RES'][0]/max(r0,1e-12):5.1f}% of natural)  "
              f"FID {r['FID'][0]:.4f}  -> {v}")
cd = [v for k, v in verdict.items() if k.startswith("direct")]
ld = [v for k, v in verdict.items() if k.startswith("latent")]
print()
if cd and ld:
    if all(v == "WORKS" for v in cd) and not any(v == "WORKS" for v in ld):
        print("  >>> PREDICTION HOLDS. ProjFlow's operator works on the direct model and fails through the")
        print("      decoder, for a LINEAR constraint. The negative result is about placement, not about the")
        print("      bone-length constraint class; ProjFlow becomes a baseline we explain rather than a threat.")
    elif any(v == "WORKS" for v in ld):
        print("  >>> A LATENT ARM WORKS. Linear keyframe constraints survive the round trip where bone length")
        print("      does not. The claim must narrow to the constraint class a decoder preserves; report which")
        print("      variant (roundtrip / delta-enc, window) and treat it as a finding, not a failure.")
    else:
        print("  >>> MIXED. Check which direct rows fail before reading the latent arm: if ProjFlow's operator")
        print("      does not work on CDFM either, this constraint/step budget is not ProjFlow's regime.")

dst = os.path.join(WORK_DIR, "linear_constraint_placement.json")
json.dump(dict(n=N, reps=REPS, K=K, joints=JOINTS, dims=DIMS, w_grid=W_GRID, results=results, verdict=verdict),
          open(dst, "w"), indent=2)
print(f"\nraw results -> {dst}")
