#!/usr/bin/env python
# 5.2 (MF-2) — CONSTRAINT PLACEMENT AS THE STEP BUDGET COLLAPSES TO ONE.
#
# THE STRUCTURAL POINT. In-ODE enforcement acts on the intermediate states of the sampling trajectory.
# With n steps there are n - 1 of them; at one NFE there are none. In-ODE projection is then not
# merely worse, it is UNDEFINED -- terminal projection is the only placement that exists. The field is
# moving toward one-step generators (reflow, MeanFlow, iMF), so the placement question resolves in
# favour of terminal correction there. This script measures that as a trend over
# NFE in {1, 2, 4, 8, 50} rather than asserting it.
#
# MODELS (each sampled at the guidance it was trained for):
#   reflow   latent student, CFG baked into its pairs -> guidance 1.0 (2.5 double-applies guidance)
#   latent   the LFM teacher, guidance 2.5
#   direct   the CDFM teacher, guidance 2.5 (the direct-space column)
#   mf / imf MeanFlow / improved-MeanFlow heads from meanflow_train.py, if their checkpoints exist;
#            CFG is baked into their training target, so guidance 1.0, and each step uses the average
#            velocity over the step interval.
#
# TREATMENTS per (model, NFE):
#   unconstrained               FID, BLE
#   terminal projection         project_joints on the output (BLE = 0 by construction)
#   in-ODE (interior states)    project every interior state, re-standardised re-encode, NO final
#                               projection: what the in-trajectory placement achieves on its own
#   in-ODE + terminal           the same followed by the final projection
# plus, for latent models, RT_BLE: BLE of D(E(P(x))) -- does the projected output survive one
# autoencoder round trip? (the frozen-decoder non-preservation, measured on this sampler's outputs)
#
# PRE-COMMITTED READING: terminal projection 'retains its properties' at an NFE if BLE = 0 and its FID
# is within max(0.05, 10%) of that NFE's unconstrained FID. In-ODE 'degrades' at an NFE if its FID
# exceeds unconstrained by more than max(0.05, 25%) or its own BLE is above unconstrained.
#
#   sbatch run_onestep.sh          (env: OS_N, OS_REPS, OS_NFE, OS_MODELS)

import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, ClipSet, run_protocol, decode, encode, null_cond, fmt, load_mf

M = load_main("onestep")
DEVICE = M.DEVICE; GUID = M.GUIDANCE

N = int(os.environ.get("OS_N", "512"))
REPS = int(os.environ.get("OS_REPS", "3"))
NFES = [int(x) for x in os.environ.get("OS_NFE", "1,2,4,8,50").split(",")]
MODELS = os.environ.get("OS_MODELS", "reflow,latent,direct,mf,imf").split(",")
WORK_DIR = os.environ.get("WORK_DIR", ".")


def get_model(name):
    """-> (kind, net, is_lat, guidance, zstats) or None."""
    if name in ("mf", "imf"):
        r = load_mf(M, name)
        if r is None: return None
        net, zm, zs = r; return ("mf", net, True, 1.0, (zm, zs))
    if not M._have(name): return None
    is_lat = name != "direct" and not name.startswith("direct")
    net = M.load_net(name, is_lat)
    return ("fm", net, is_lat, 1.0 if name == "reflow" else GUID, (M.z_mean_t, M.z_std_t))


@torch.no_grad()
def sample_os(kind, net, is_lat, g, ts, tm, tp, L, seed, n, inode):
    torch.manual_seed(seed); B = tp.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE); ns, nm, npl = null_cond(net, B, M)
    grid = M._timesteps(n, "linear")
    for i in range(n):
        t = float(grid[i]); s = float(grid[i + 1]); tt = torch.full((B,), t, device=DEVICE)
        if kind == "mf":
            v = net(z, tt, torch.full((B,), s, device=DEVICE), ts, tm, tp, L)
        else:
            v = M._cfg(net, z, tt, ts, tm, tp, L, ns, nm, npl, g, 0.0)
        z = z + (s - t) * v
        if inode and i < n - 1:                        # interior states only; none exist at n = 1
            mn = decode(M, z, is_lat)
            z = encode(M, M._joints_to_norm(M.project_joints(M._gj(mn), L), mn), is_lat)
    return decode(M, z, is_lat)


def proj(mn, L): return M._joints_to_norm(M.project_joints(M._gj(mn), L), mn)


print("\n" + "=" * 118)
print(f"ONE-STEP PLACEMENT — {N}-clip protocol, {REPS} reps, NFE {NFES}")
print("=" * 118, flush=True)
clips = ClipSet(M, N)
res = {}
for name in MODELS:
    got = get_model(name)
    if got is None: print(f"[onestep] {name}: no checkpoint, skipped"); continue
    kind, net, is_lat, g, zs = got
    print(f"\n--- {name} ({kind}, latent={is_lat}, guidance {g}) ---", flush=True)
    for n in NFES:
        for tr in ["unconstrained", "terminal", "in-ODE", "in-ODE + terminal"]:
            if tr.startswith("in-ODE") and n == 1:
                res[f"{name}|{n}|{tr}"] = dict(model=name, nfe=n, treatment=tr, undefined=True)
                print(f"  n={n:<3} {tr:<20} STRUCTURALLY UNDEFINED (no interior state at 1 NFE)")
                continue
            def gen(s, e, ts, tm, tp, L, seed, n=n, tr=tr):
                M.z_mean_t, M.z_std_t = zs
                mn = sample_os(kind, net, is_lat, g, ts, tm, tp, L, seed, n, inode=tr.startswith("in-ODE"))
                if tr in ("terminal", "in-ODE + terminal"): mn = proj(mn, L)
                ex = {}
                if is_lat:
                    rt = decode(M, encode(M, proj(mn, L), True), True)
                    ex["RT_BLE"] = M.ble_pc_joints(M._gj(rt), L)
                else:
                    ex["RT_BLE"] = np.zeros(e - s)
                return mn, ex
            r = run_protocol(M, clips, gen, REPS, label=f"{name} n={n} {tr}", extra_keys=("RT_BLE",))
            res[f"{name}|{n}|{tr}"] = dict(model=name, nfe=n, treatment=tr, undefined=False,
                                           **{k: v for k, v in r.items()})

# ------------------------------------------------------------------ table + reading
print("\n" + "=" * 118)
print(f" {'model':<8}{'NFE':>4}  {'treatment':<20}{'FID':>18}{'R@3':>14}{'BLE':>14}{'RT_BLE':>12}")
print("-" * 118)
for k, r in res.items():
    if r["undefined"]:
        print(f" {r['model']:<8}{r['nfe']:>4}  {r['treatment']:<20}{'— undefined —':>18}"); continue
    print(f" {r['model']:<8}{r['nfe']:>4}  {r['treatment']:<20}{fmt(*r['FID']):>18}{fmt(*r['R3'], p=3):>14}"
          f"{r['BLE'][0]:>14.5f}{r['RT_BLE'][0]:>12.5f}")
print("=" * 118)

print("\nREADING (thresholds fixed before the run)")
reading = {}
for name in MODELS:
    rows = {(r["nfe"], r["treatment"]): r for r in res.values() if r["model"] == name}
    if not rows: continue
    term_ok, inode_bad = [], []
    for n in NFES:
        u = rows.get((n, "unconstrained")); tm_ = rows.get((n, "terminal")); io = rows.get((n, "in-ODE"))
        if not u: continue
        f0 = u["FID"][0]
        if tm_:
            term_ok.append((n, tm_["BLE"][0] < 1e-4 and tm_["FID"][0] <= f0 + max(0.05, 0.10 * f0)))
        if io and not io["undefined"]:
            inode_bad.append((n, io["FID"][0] > f0 + max(0.05, 0.25 * f0) or io["BLE"][0] > u["BLE"][0]))
    reading[name] = dict(terminal_retains=term_ok, inode_degrades=inode_bad)
    print(f"  {name:<8} terminal retains BLE=0 at ~no FID cost: "
          + " ".join(f"n={n}:{'yes' if ok else 'NO'}" for n, ok in term_ok))
    print(f"  {'':<8} in-ODE degrades:                         n=1:undefined "
          + " ".join(f"n={n}:{'yes' if b else 'no'}" for n, b in inode_bad))
print("\n  The claim holds for a model if terminal retains at every NFE (including 1) while in-ODE degrades")
print("  at small NFE and is undefined at 1. If terminal fails at n=1 for some model, the one-step argument")
print("  must be restricted to the models where it holds; report which.")

dst = os.path.join(WORK_DIR, "onestep_placement.json")
json.dump(dict(n=N, reps=REPS, nfes=NFES, results=res, reading=reading), open(dst, "w"), indent=2, default=float)
print(f"\nraw results -> {dst}")
