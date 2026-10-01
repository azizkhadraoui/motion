#!/usr/bin/env python
# FID-optimisation experiments — the evaluation half. Every experiment: screen a grid on the 512-clip
# SCREENING set (disjoint from the report set), pick the best configuration of each technique by screening
# FID, then report those against the baseline (CFG 2.5, 50 Euler steps, base weights) on the standard
# 512-clip REPORT set with 5 paired seeds and the full metric panel.
#
#   --exp guidance  E1: CFG-scale sweep, CFG-Zero* (scale x zero-init), APG (scale x eta x momentum)
#   --exp neon      E2: theta_Neon = (1+w) theta_r - w theta_s, jointly with the CFG scale; the base model at
#                       the same CFG scales is screened too, so a gain from re-tuning CFG alone is not
#                       credited to Neon
#   --exp auto      E3: autoguidance with the small guide (two training budgets) x weight x optional CFG term
#   --exp dfm       E4: contrastive-FM fine-tunes vs the matched plain fine-tunes (lambda = 0) x CFG scale
#
# Pre-committed reading: a technique HELPS only if its paired dFID against the baseline has a 95% CI
# entirely below zero on the report set AND SW or MMD moves the same way (a FID-only gain is reported as
# such, not as an improvement). Precision/recall/coverage changes are reported alongside.
#
#   sbatch run_fidopt_eval.sh <exp> <base>

import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, ClipSet
import fidopt as FO

ap = argparse.ArgumentParser()
ap.add_argument("--exp", choices=["guidance", "neon", "auto", "dfm"], required=True)
ap.add_argument("--base", choices=["latent", "direct"], required=True)
ap.add_argument("--reps", type=int, default=int(os.environ.get("FO_REPS", "5")))
ap.add_argument("--n", type=int, default=int(os.environ.get("FO_N", "512")))
A = ap.parse_args()
M = load_main(f"fidopt-{A.exp}-{A.base}")
DEVICE = M.DEVICE; IS_LAT = A.base == "latent"
cd = M.RVQ_CODE_DIM if IS_LAT else M.NFEATS; Tl = M.T_LAT if IS_LAT else M.MAX_MOTION_LEN

base = M.load_net(A.base, IS_LAT); ZS = (M.z_mean_t, M.z_std_t)
SCREEN = ClipSet(M, A.n, start=A.n); REPORT = ClipSet(M, A.n, start=0)
BASE_NAME = "baseline (CFG 2.5)"
gen = lambda net, cfg, guide=None: FO.make_gen(M, net, IS_LAT, cfg, guide=guide, zstats=ZS)
TAG = os.environ.get("FO_TAG", "")
ck = lambda task, step: os.path.join(M.CK, f"fidopt_{task}_{A.base}{TAG}_s{step}.pt")
EV = [int(x) for x in os.environ["FO_EVAL_STEPS"].split(",")] if os.environ.get("FO_EVAL_STEPS") else None


def load_state(path):
    return torch.load(path, map_location=DEVICE, weights_only=False)


def net_from_state(state, arch=None):
    if arch is None:
        n = M.FMNet(cd, Tl, M.LHID if IS_LAT else M.DHID, M.LLAYERS if IS_LAT else M.DLAYERS, M.LHEADS if IS_LAT else M.DHEADS)
    else:
        n = M.FMNet(cd, Tl, arch["hid"], arch["layers"], arch["heads"])
    n = n.to(DEVICE); n.load_state_dict(state); n.eval(); return n


print(f"\n{'='*110}\nFIDOPT {A.exp} — base {A.base}; screening on clips [{A.n}, {2*A.n}), reporting on [0, {A.n}) x {A.reps} seeds\n{'='*110}", flush=True)
selected = []                                   # (name, gen) to report
screen_res = {}

if A.exp == "guidance":
    fams = {"CFG scale": [dict(mode="cfg", g=g) for g in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]],
            "CFG-Zero*": [dict(mode="cfgzero", g=g, zero_init=K) for g in [2.0, 2.5, 3.0, 4.0] for K in [0, 1, 2]],
            "APG": [dict(mode="apg", g=g, eta=eta, beta=beta) for g in [2.5, 4.0, 6.0, 8.0] for eta in [0.0, 0.5] for beta in [0.0, -0.5]]}
    for fam, cfgs in fams.items():
        r = FO.screen(M, SCREEN, [(FO.label(c), gen(base, c)) for c in cfgs], fam)
        screen_res[fam] = r
        best = min(cfgs, key=lambda c: r[FO.label(c)])
        if FO.label(best) != FO.label(FO.BASELINE):
            selected.append((f"{fam} best: {FO.label(best)}", gen(base, best)))

elif A.exp == "neon":
    theta_r = {k: v.detach().clone() for k, v in base.state_dict().items()}
    GS = [2.0, 2.5, 3.0]
    items = [(f"base, CFG {g}", gen(base, dict(mode="cfg", g=g))) for g in GS]
    r_base = FO.screen(M, SCREEN, items, "base CFG")
    best_g = min(GS, key=lambda g: r_base[f"base, CFG {g}"])
    if best_g != 2.5: selected.append((f"base, CFG {best_g} (CFG re-tune only)", gen(base, dict(mode="cfg", g=best_g))))
    screen_res["base"] = r_base; cand = []
    for step in (EV or [1000, 4000]):
        p = ck("neon", step)
        if not os.path.exists(p): print(f"  missing {p}, skipped"); continue
        theta_s = load_state(p)["state"]
        for w in [0.5, 1.0, 2.0, 4.0]:
            merged = {k: ((1 + w) * theta_r[k] - w * theta_s[k].to(DEVICE)) if theta_r[k].is_floating_point() else theta_r[k]
                      for k in theta_r}
            net = net_from_state(merged)
            items = [(f"Neon B={step} w={w}, CFG {g}", gen(net, dict(mode="cfg", g=g))) for g in GS]
            r = FO.screen(M, SCREEN, items, "Neon"); screen_res.update(r)
            cand += [(r[n_], step, w, g) for (n_, _), g in zip(items, GS)]
            del net; torch.cuda.empty_cache()
    if cand:
        f_, step, w, g = min(cand)
        theta_s = load_state(ck("neon", step))["state"]
        best_net = net_from_state({k: ((1 + w) * theta_r[k] - w * theta_s[k].to(DEVICE)) if theta_r[k].is_floating_point() else theta_r[k]
                                   for k in theta_r})
        selected.append((f"Neon B={step} w={w}, CFG {g}", gen(best_net, dict(mode="cfg", g=g))))
        # the self-trained model itself (w = -1 direction is theta_s): shows the degradation Neon reverses
        selected.append((f"self-trained theta_s B={step}, CFG {g} (reference)", gen(net_from_state(theta_s), dict(mode="cfg", g=g))))

elif A.exp == "auto":
    cand = []
    for step in (EV or [5000, 20000]):
        p = ck("auto", step)
        if not os.path.exists(p): print(f"  missing {p}, skipped"); continue
        d = load_state(p); guide_net = net_from_state(d["state"], d["arch"])
        guide = lambda z, t, ts, tm, tp, L, g_=guide_net: g_(z, t, ts, tm, tp, L)
        cfgs = [dict(mode="auto", w=w, g=gc) for w in [1.5, 2.0, 3.0] for gc in [1.0, 1.5, 2.0]]
        items = [(f"guide@{step} {FO.label(c)}", gen(base, c, guide)) for c in cfgs]
        r = FO.screen(M, SCREEN, items, "autoguidance"); screen_res.update(r)
        cand += [(r[n_], step, c) for (n_, _), c in zip(items, cfgs)]
    if cand:
        f_, step, c = min(cand, key=lambda x: x[0])
        d = load_state(ck("auto", step)); guide_net = net_from_state(d["state"], d["arch"])
        selected.append((f"guide@{step} {FO.label(c)}", gen(base, c, lambda z, t, ts, tm, tp, L: guide_net(z, t, ts, tm, tp, L))))

elif A.exp == "dfm":
    GS = [2.0, 2.5, 3.0]; best = {}
    for task in ["dfm", "ftctl"]:
        cand = []
        for step in (EV or [2500, 5000, 10000]):
            p = ck(task, step)
            if not os.path.exists(p): print(f"  missing {p}, skipped"); continue
            net = net_from_state(load_state(p)["state"])
            items = [(f"{task}@{step}, CFG {g}", gen(net, dict(mode="cfg", g=g))) for g in GS]
            r = FO.screen(M, SCREEN, items, task); screen_res.update(r)
            cand += [(r[n_], step, g) for (n_, _), g in zip(items, GS)]
            del net; torch.cuda.empty_cache()
        if cand:
            f_, step, g = min(cand); best[task] = (step, g)
            selected.append((f"{'contrastive FM' if task == 'dfm' else 'plain fine-tune (control)'} @{step}, CFG {g}",
                             gen(net_from_state(load_state(ck(task, step))["state"]), dict(mode="cfg", g=g))))

items = [(BASE_NAME, gen(base, FO.BASELINE))] + selected
rows = FO.report(M, REPORT, items, A.reps, f"{A.exp}/{A.base}", BASE_NAME)

print("\n  READING (pre-committed: helps = paired dFID CI below zero AND SW or MMD improves too)")
verdict = {}
for name, r in rows.items():
    if name == BASE_NAME: continue
    d = r["paired"]; fid_better = r["better_than_baseline"]
    agree = (d["SW"][0] + d["SW"][1] < 0) or (d["MMD"][0] + d["MMD"][1] < 0)
    v = "HELPS" if (fid_better and agree) else ("FID-only gain (not corroborated by SW/MMD)" if fid_better else "no improvement")
    verdict[name] = v
    print(f"   {name:<60} -> {v}")
FO.save(dict(exp=A.exp, base=A.base, n=A.n, reps=A.reps, screen=screen_res, report=rows, verdict=verdict),
        f"fidopt_{A.exp}_{A.base}{TAG}.json")
print("=== FIDOPT EVAL DONE ===")
