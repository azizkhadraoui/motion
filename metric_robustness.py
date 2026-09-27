#!/usr/bin/env python
# 3.2 — METRIC ROBUSTNESS. Do the headline claims survive a change of distribution metric?
#
# WHY. Every headline claim rests on FID, a single Gaussian-assumption metric, and BLE/FID have
# disagreed three separate times (training penalty, best-of-N selection, velocity-space guidance).
# This script re-scores the same treatments under metrics that make different assumptions:
#   * sliced Wasserstein-2 (no Gaussian assumption, 1000 fixed random directions)
#   * MMD^2, RBF kernel, bandwidth fixed from the REAL features (unbiased estimator)
#   * precision / recall / density / coverage (Naeem et al. 2020, k = 5)
#   * diversity (mean pairwise feature distance, HumanML3D definition)
# all in the same evaluator feature space as FID, on the same clips and noise.
#
# PRECISION AND RECALL ARE THE DIAGNOSTIC PAIR. FID conflates fidelity and coverage. If projection
# raises precision while cutting recall, part of its FID behaviour is distribution-narrowing and the
# paper must say so.
#
# PROTOCOL. Default MR_N = 1024 clips, so these numbers are the 1024-CLIP PROTOCOL: FID here is not
# comparable to the 512-clip tables (FID is biased upward at smaller n). Only within-table
# comparisons are valid. Replications vary the sampling noise; CIs are across replications, and
# method-vs-unconstrained differences are PAIRED per replication (same clips, same seeds).
#
# Pre-committed readings, printed at the end:
#   (1) rank agreement: Kendall tau between the FID ranking of treatments and each other metric's
#       ranking. tau >= 0.6 for SW and MMD -> FID-based orderings are robust.
#   (2) narrowing flag: a treatment whose paired precision change is > 0 and recall change < 0, both
#       with CIs excluding zero, is flagged as distribution-narrowing.
#   (3) sign disagreements: for each treatment vs unconstrained, metrics whose paired difference has
#       a CI excluding zero and a sign opposite to FID's.
#
#   sbatch run_metrics.sh          (env: MR_N, MR_REPS, MR_BASES, MR_GW, MR_BON)

import os, sys, json, math, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, ClipSet, run_protocol, decode, encode, null_cond, ci95, fmt

M = load_main("metrics")
DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE

N = int(os.environ.get("MR_N", "1024"))
REPS = int(os.environ.get("MR_REPS", "3"))
BASES = os.environ.get("MR_BASES", "latent,direct").split(",")
GW = float(os.environ.get("MR_GW", "0.1"))            # soft-guidance weight (mildest with a clear effect)
BON = int(os.environ.get("MR_BON", "4"))              # best-of-N candidates
WIN, STRIDE = 0.10, 4                                  # the in-ODE configuration discussed in the draft
WORK_DIR = os.environ.get("WORK_DIR", ".")
METRICS = ["FID", "SW", "MMD", "precision", "recall", "density", "coverage", "R3", "DIV", "BLE", "FSR"]
LOWER_BETTER = {"FID": True, "SW": True, "MMD": True, "BLE": True, "FSR": True}


@torch.no_grad()
def sample_inode(net, is_lat, ts, tm, tp, L, seed):
    """In-ODE projection over the last WIN of the trajectory every STRIDE steps, RE-STANDARDISED
    re-encode, final projection on. Identical to 01_inproc_restd.py 'restd ON +final proj'."""
    torch.manual_seed(seed); B = tp.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE); ns, nm, npl = null_cond(net, B, M)
    start = int(round(ODE * (1.0 - WIN))); grid = M._timesteps(ODE, "linear")
    for i in range(ODE):
        t = float(grid[i]); dt = float(grid[i + 1] - grid[i])
        v = M._cfg(net, z, torch.full((B,), t, device=DEVICE), ts, tm, tp, L, ns, nm, npl, GUID, 0.0)
        z = z + dt * v
        if i >= start and (i - start) % STRIDE == 0:
            mn = decode(M, z, is_lat)
            z = encode(M, M._joints_to_norm(M.project_joints(M._gj(mn), L), mn), is_lat)
    mn = decode(M, z, is_lat)
    return M._joints_to_norm(M.project_joints(M._gj(mn), L), mn)


def zstats(): return (M.z_mean_t, M.z_std_t)


def treatments(tag, is_lat):
    net = M.load_net(tag, is_lat); zs_base = zstats()
    pen_tag = tag + "_pen"
    T = [("unconstrained", lambda s, e, ts, tm, tp, L, sd: (M.sample(net, is_lat, ts, tm, tp, L, seed=sd), {})),
         ("post-hoc projection (bone+foot)", lambda s, e, ts, tm, tp, L, sd: (M.sample(net, is_lat, ts, tm, tp, L, mode="posthoc", seed=sd), {})),
         ("post-hoc bone only", lambda s, e, ts, tm, tp, L, sd: (
             (lambda x: M._joints_to_norm(M.project_bonelength(M._gj(x)), x))(M.sample(net, is_lat, ts, tm, tp, L, seed=sd)), {})),
         (f"in-ODE proj ({int(WIN*100)}%, k={STRIDE}) + final", lambda s, e, ts, tm, tp, L, sd: (sample_inode(net, is_lat, ts, tm, tp, L, sd), {})),
         (f"soft guidance w={GW:g}", lambda s, e, ts, tm, tp, L, sd: (M.sample(net, is_lat, ts, tm, tp, L, mode="guided", guide_w=GW, seed=sd), {})),
         ]

    def bestofn(s, e, ts, tm, tp, L, sd):
        cands, bles = [], []
        for j in range(BON):
            x = M.sample(net, is_lat, ts, tm, tp, L, seed=sd + 7919 * j)
            cands.append(x); bles.append(torch.tensor(M.ble_pc_joints(M._gj(x), L), device=DEVICE))
        pick = torch.stack(bles).argmin(0)                                 # proxy uses no ground truth
        X = torch.stack(cands)                                             # (N,B,T,D)
        return X[pick, torch.arange(X.shape[1], device=DEVICE)], {}
    T.append((f"best-of-{BON} by BLE (no projection)", bestofn))
    T = [(l, g, zs_base) for l, g in T]
    if M._have(pen_tag):
        pnet = M.load_net(pen_tag, is_lat); zs_pen = zstats()             # the penalty model's own z-stats
        T.append(("penalty-trained model", lambda s, e, ts, tm, tp, L, sd: (M.sample(pnet, is_lat, ts, tm, tp, L, seed=sd), {}), zs_pen))
    return net, T


def kendall(a, b):
    n = len(a); c = d = 0
    for i, j in itertools.combinations(range(n), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0; d += s < 0
    return (c - d) / max(c + d, 1)


print("\n" + "=" * 120)
print(f"METRIC ROBUSTNESS — {N}-CLIP PROTOCOL (not comparable to 512-clip tables), {REPS} reps")
print("=" * 120, flush=True)
clips = ClipSet(M, N)
out = {}
for tag in BASES:
    is_lat = tag.startswith("latent")
    if not M._have(tag): print(f"[metrics] {tag} missing, skipped"); continue
    print(f"\n--- base {tag} ---", flush=True)
    net, T = treatments(tag, is_lat)
    res = {}
    for label, gen, zs in T:
        M.z_mean_t, M.z_std_t = zs             # the latent standardisation belonging to THIS model
        res[label] = run_protocol(M, clips, gen, REPS, label=label, full_metrics=True)
    out[tag] = res

    print(f"\n  {tag}: all metrics (mean ± 95% CI over {REPS} reps)")
    print("  " + f"{'treatment':<38}" + "".join(f"{m:>17}" for m in METRICS[:7]))
    for label, r in res.items():
        print("  " + f"{label[:38]:<38}" + "".join(f"{fmt(*r[m], p=4 if m != 'MMD' else 5):>17}" for m in METRICS[:7]))

# ------------------------------------------------------------------ readings
print("\n" + "=" * 120 + "\nREADING (thresholds fixed before the run)\n" + "=" * 120)
reading = {}
for tag, res in out.items():
    labels = list(res)
    base = "unconstrained"
    print(f"\n[{tag}]")
    fid_rank = [res[l]["FID"][0] for l in labels]
    taus = {}
    for m in ["SW", "MMD", "precision", "recall", "density", "coverage"]:
        vals = [res[l][m][0] for l in labels]
        if not LOWER_BETTER.get(m): vals = [-v for v in vals]       # orient so that lower = better
        taus[m] = kendall(fid_rank, vals)
    print("  (1) Kendall tau vs FID ranking: " + "  ".join(f"{m}={t:+.2f}" for m, t in taus.items()))
    robust = taus["SW"] >= 0.6 and taus["MMD"] >= 0.6
    print("      -> " + ("FID orderings ROBUST (SW and MMD agree)." if robust else
                        "FID orderings NOT robust: at least one of SW/MMD reorders the treatments. Report both."))
    flags = {}
    for l in labels:
        if l == base: continue
        d = {}
        for m in METRICS:
            diff = np.array(res[l]["_per_rep"][m]) - np.array(res[base]["_per_rep"][m])
            d[m] = ci95(diff)
        narrowing = (d["precision"][0] - d["precision"][1] > 0) and (d["recall"][0] + d["recall"][1] < 0)
        sig = lambda m: d[m][1] == d[m][1] and abs(d[m][0]) > d[m][1]
        fid_dir = np.sign(d["FID"][0])
        disagree = []
        for m in ["SW", "MMD", "precision", "recall", "density", "coverage"]:
            if not sig(m) or not sig("FID"): continue
            better_m = (d[m][0] < 0) if LOWER_BETTER.get(m) else (d[m][0] > 0)
            if better_m != (fid_dir < 0): disagree.append(m)
        flags[l] = dict(diff={m: d[m] for m in METRICS}, narrowing=bool(narrowing), disagree=disagree)
        print(f"  {l[:40]:<40} dFID={fmt(*d['FID'])}  dPrec={fmt(*d['precision'], p=3)}  "
              f"dRec={fmt(*d['recall'], p=3)}  dCov={fmt(*d['coverage'], p=3)}"
              + ("   [NARROWING]" if narrowing else "") + (f"   [disagrees with FID: {','.join(disagree)}]" if disagree else ""))
    reading[tag] = dict(kendall=taus, robust=bool(robust), flags=flags)
print("\n  (2) Any [NARROWING] on a projection row means part of its FID behaviour is distribution")
print("      narrowing, not fidelity: the paper must say so next to that number.")
print("  (3) Any [disagrees with FID] row is a claim that must be stated per-metric, not via FID alone.")

dst = os.path.join(WORK_DIR, f"metric_robustness_{N}clip.json")
json.dump(dict(protocol=f"{N}-clip", reps=REPS, gw=GW, bon=BON, results=out, reading=reading),
          open(dst, "w"), indent=2, default=float)
print(f"\nraw results -> {dst}")
