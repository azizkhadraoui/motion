#!/usr/bin/env python
# Tier 3 — DIRECT IN-PROCESS (window 10%, stride 4) SEED REPLICATION.
#
# A single seed gave CDFM in-process projection FID 0.1582 at BLE 0 -- better than CDFM unconstrained
# (0.2088) and post-hoc (0.1723). That is inside the seed band and cannot enter the draft unreplicated.
# This script replicates it over IR_REPS noise seeds on the fixed 512-clip set and reports PAIRED
# differences (same clips, same seeds per replication).
#
# Pre-committed reading: the in-process advantage over post-hoc is REAL only if the paired 95% CI of
# FID(in-process) - FID(post-hoc) lies entirely below zero. Otherwise report the two as tied.
#
#   sbatch run_inproc_direct.sh          (env: IR_REPS=5, IR_N=512, IR_BASES=direct)

import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, ClipSet, run_protocol, decode, encode, null_cond, ci95, fmt

M = load_main("inproc-rep")
DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE
REPS = int(os.environ.get("IR_REPS", "5")); N = int(os.environ.get("IR_N", "512"))
BASES = os.environ.get("IR_BASES", "direct").split(",")
WIN, STRIDE = 0.10, 4
WORK_DIR = os.environ.get("WORK_DIR", ".")


@torch.no_grad()
def sample_inode(net, is_lat, ts, tm, tp, L, seed):
    torch.manual_seed(seed); B = tp.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE); ns, nm, npl = null_cond(net, B, M)
    start = int(round(ODE * (1.0 - WIN))); grid = M._timesteps(ODE, "linear")
    for i in range(ODE):
        t = float(grid[i]); dt = float(grid[i + 1] - grid[i])
        z = z + dt * M._cfg(net, z, torch.full((B,), t, device=DEVICE), ts, tm, tp, L, ns, nm, npl, GUID, 0.0)
        if i >= start and (i - start) % STRIDE == 0:
            mn = decode(M, z, is_lat)
            z = encode(M, M._joints_to_norm(M.project_joints(M._gj(mn), L), mn), is_lat)
    mn = decode(M, z, is_lat)
    return M._joints_to_norm(M.project_joints(M._gj(mn), L), mn)


clips = ClipSet(M, N); out = {}
print(f"\n{'='*100}\nIN-PROCESS REPLICATION — {N}-clip protocol, {REPS} reps, window {WIN:.0%}, stride {STRIDE}\n{'='*100}")
for tag in BASES:
    is_lat = tag.startswith("latent"); net = M.load_net(tag, is_lat)
    r = {}
    r["unconstrained"] = run_protocol(M, clips, lambda s, e, ts, tm, tp, L, sd: (M.sample(net, is_lat, ts, tm, tp, L, seed=sd), {}), REPS, label=f"{tag} unconstrained")
    r["post-hoc"] = run_protocol(M, clips, lambda s, e, ts, tm, tp, L, sd: (M.sample(net, is_lat, ts, tm, tp, L, mode="posthoc", seed=sd), {}), REPS, label=f"{tag} post-hoc")
    r["in-process"] = run_protocol(M, clips, lambda s, e, ts, tm, tp, L, sd: (sample_inode(net, is_lat, ts, tm, tp, L, sd), {}), REPS, label=f"{tag} in-process (10%, k=4)")
    d_ph = ci95(np.array(r["in-process"]["_per_rep"]["FID"]) - np.array(r["post-hoc"]["_per_rep"]["FID"]))
    d_un = ci95(np.array(r["in-process"]["_per_rep"]["FID"]) - np.array(r["unconstrained"]["_per_rep"]["FID"]))
    real = d_ph[0] + d_ph[1] < 0
    print(f"\n  [{tag}] paired dFID in-process - post-hoc      = {fmt(*d_ph)}")
    print(f"  [{tag}] paired dFID in-process - unconstrained = {fmt(*d_un)}")
    print(f"  per-rep FID in-process: {[round(x, 4) for x in r['in-process']['_per_rep']['FID']]}")
    print("  -> " + ("in-process advantage over post-hoc is REAL (CI below zero); it may enter the draft."
                     if real else "NOT established: the two are tied within seed variance. Do not claim the single-seed 0.1582."))
    out[tag] = dict(rows=r, d_inproc_minus_posthoc=d_ph, d_inproc_minus_unc=d_un, advantage_real=bool(real))
json.dump(dict(n=N, reps=REPS, window=WIN, stride=STRIDE, results=out),
          open(os.path.join(WORK_DIR, "inproc_direct_replicated.json"), "w"), indent=2, default=float)
print(f"\nraw results -> {os.path.join(WORK_DIR, 'inproc_direct_replicated.json')}")
