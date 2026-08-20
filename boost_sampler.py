#!/usr/bin/env python
# A1 + A3 — TWO SAMPLER AXES YOUR FID-TRICKS ABLATION DID NOT TEST.
#
# The earlier ablation swept CFG *magnitude* (rescale), the timestep *grid*, and the *solver*, and
# found nothing beat plain Euler at 0.1473. Both axes here are different in kind:
#
#   A1 GUIDANCE INTERVAL (Kynkaanniemi et al., NeurIPS 2024, arXiv:2404.07724). Keep guidance at 2.5
#      but apply it only on a sub-interval of t, running unguided (scale 1.0) at both ends. Their
#      finding is that guidance is actively harmful early in the trajectory and mainly useful in the
#      middle; on ImageNet-512 the interval trick moved FID 1.81 -> 1.40. This changes WHERE guidance
#      acts, not how strong it is, so it is not covered by the rescale sweep.
#
#   A3 STOCHASTIC INJECTION (arXiv:2410.02217, arXiv:2510.06634). Convert the deterministic ODE into
#      the marginal-preserving SDE. For the linear/OT path z_t = (1-t)z_0 + t z_1 the score follows
#      from the predicted velocity in closed form, with no extra network:
#           E[z_0 | z_t] = z_t - t*v      =>      grad log p_t(z_t) = -(z_t - t*v) / (1-t)
#      so the SDE step is
#           dz = [v + (sigma^2/2) * grad log p] dt + sigma dW,   sigma(t) = churn * (1-t)
#      The (1-t) factor makes sigma vanish at t=1 and keeps the score term bounded, which matters
#      because the naive constant-sigma form blows up as t->1. churn=0 recovers your exact baseline.
#
# Both are pure inference-time on the frozen checkpoints. Every config is also run with post-decode
# projection, so you learn whether any FID gain survives the projection (it should: projection acts
# after decoding) and whether the pairing is additive.
#
#   sbatch run_boost_sampler.sh

import os, sys, json, time, itertools
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[boost-sampler] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[boost-sampler] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
_cfg = M._cfg; _timesteps = M._timesteps; rvq = M.rvq
eval_variant = M.eval_variant; load_net = M.load_net; _agg = M._agg
project_joints = M.project_joints; _gj = M._gj; _joints_to_norm = M._joints_to_norm

BASE = os.environ.get("BOOST_BASE", "latent")
IS_LAT = (BASE in ("latent", "latent_pen"))
GI_GRID = os.environ.get("BOOST_GI", "0.0-1.0,0.1-1.0,0.2-1.0,0.1-0.9,0.2-0.8,0.3-0.9,0.0-0.8")
CHURN_GRID = [float(x) for x in os.environ.get("BOOST_CHURN", "0,0.25,0.5,1.0,2.0").split(",")]

# module-level knobs read by the patched sampler
GI_LO, GI_HI, CHURN = 0.0, 1.0, 0.0

def make_sampler(orig):
    @torch.no_grad()
    def s(net, is_latent, tseq, tmask, tpool, length, n=ODE, guidance=GUID, mode="none", seed=None,
          guide_w=None, cfg_rescale=0.0, schedule="linear", solver="euler", gw_schedule="const"):
        if seed is not None: torch.manual_seed(seed)
        B = tpool.shape[0]
        z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
        ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
        nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
        npl = net.null_pool.unsqueeze(0).expand(B, -1)
        ts = _timesteps(n, schedule)
        for i in range(n):
            t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
            tt = torch.full((B,), t, device=DEVICE)
            g = guidance if (GI_LO <= t <= GI_HI) else 1.0          # A1: guidance interval
            v = _cfg(net, z, tt, tseq, tmask, tpool, length, ns, nm, npl, g, cfg_rescale)
            if CHURN > 0.0 and t < 1.0:                              # A3: stochastic injection
                sig = CHURN * (1.0 - t)
                score = -(z - t * v) / max(1.0 - t, 1e-4)
                z = z + dt * (v + 0.5 * sig * sig * score) + sig * math_sqrt(dt) * torch.randn_like(z)
            else:
                z = z + dt * v
        mn = rvq.decoder(z * M.z_std_t + M.z_mean_t) if is_latent else z
        if mode == "posthoc":
            mn = _joints_to_norm(project_joints(_gj(mn), length), mn)
        return mn
    return s

def math_sqrt(x):
    return float(np.sqrt(x))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", f"boost_sampler_{BASE}"), job_type="ablation")

net = load_net(BASE, IS_LAT)
orig = M.sample
M.sample = make_sampler(orig)
rows = []

def run(label, gi_lo, gi_hi, churn, mode):
    global GI_LO, GI_HI, CHURN
    GI_LO, GI_HI, CHURN = gi_lo, gi_hi, churn
    t0 = time.time()
    r = eval_variant(net, IS_LAT, mode)
    fm, _, _ = _agg(r["fsr"]); bm, _, _ = _agg(r["ble"])
    row = dict(label=label, gi_lo=gi_lo, gi_hi=gi_hi, churn=churn, mode=mode,
               FID=round(float(r["fid"]), 4), R3=round(float(r["R3"]), 4),
               BLE=round(bm, 5), FSR=round(fm, 4), sec=round(time.time() - t0))
    rows.append(row)
    print(f"  {label:<34} {mode:<9} FID={row['FID']:<9} R@3={row['R3']:<7} BLE={row['BLE']:<9} ({row['sec']}s)")
    wandb.log({f"boost/{label}/{mode}/FID": row["FID"], f"boost/{label}/{mode}/R3": row["R3"]})
    return row

try:
    print(f"\n{'='*96}\nBASELINE (reproduces the existing sampler exactly)\n{'='*96}")
    base_none = run("baseline", 0.0, 1.0, 0.0, "none")
    base_post = run("baseline", 0.0, 1.0, 0.0, "posthoc")

    print(f"\n{'='*96}\nA1 — GUIDANCE INTERVAL\n{'='*96}")
    for spec_s in GI_GRID.split(","):
        lo, hi = [float(x) for x in spec_s.split("-")]
        if (lo, hi) == (0.0, 1.0): continue                       # that is the baseline
        run(f"interval [{lo},{hi}]", lo, hi, 0.0, "none")

    print(f"\n{'='*96}\nA3 — STOCHASTIC INJECTION\n{'='*96}")
    for c in CHURN_GRID:
        if c == 0.0: continue
        run(f"churn {c}", 0.0, 1.0, c, "none")

    # best of each axis, combined, and re-checked under projection
    unc = [r for r in rows if r["mode"] == "none"]
    b_int = min([r for r in unc if r["churn"] == 0.0 and r["label"] != "baseline"], key=lambda r: r["FID"], default=None)
    b_chu = min([r for r in unc if r["churn"] > 0.0], key=lambda r: r["FID"], default=None)
    if b_int and b_chu:
        print(f"\n{'='*96}\nCOMBINED best interval + best churn, and both under post-decode projection\n{'='*96}")
        run(f"combined {b_int['label']} + {b_chu['label']}", b_int["gi_lo"], b_int["gi_hi"], b_chu["churn"], "none")
        run(f"{b_int['label']} + proj", b_int["gi_lo"], b_int["gi_hi"], 0.0, "posthoc")
        run(f"{b_chu['label']} + proj", 0.0, 1.0, b_chu["churn"], "posthoc")
finally:
    M.sample = orig

print(f"\n{'='*96}")
print(f" {'config':<36}{'mode':>10}{'FID':>10}{'R@3':>8}{'dFID':>10}")
print("-" * 96)
for r in rows:
    ref = base_post["FID"] if r["mode"] == "posthoc" else base_none["FID"]
    print(f" {r['label']:<36}{r['mode']:>10}{r['FID']:>10}{r['R3']:>8}{r['FID']-ref:>+10.4f}")
print("=" * 96)

NOISE = 0.006   # the 20-seed 95% CI half-width on the LFM baseline
best = min([r for r in rows if r["mode"] == "none"], key=lambda r: r["FID"])
print(f"\nBest unconstrained config: {best['label']} at FID {best['FID']} "
      f"(baseline {base_none['FID']}, delta {best['FID']-base_none['FID']:+.4f}).")
if base_none["FID"] - best["FID"] > NOISE:
    print(f">>> The gain exceeds the +/-{NOISE} single-seed noise band. Worth confirming across seeds")
    print("    before adopting: re-run the winner through the replicated-CI harness.")
else:
    print(f">>> The gain is inside the +/-{NOISE} noise band, i.e. NOT distinguishable from seed variance.")
    print("    Report as a second null result alongside the FID-tricks ablation rather than adopting it.")
print("\nNote: single seed per config here. This sweep is for SCREENING only; anything you intend to")
print("claim needs the 20-seed treatment, exactly as the 0.147 -> 0.169 correction taught you.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), f"boost_sampler_{BASE}.json")
json.dump(rows, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.log({"boost_sampler_table": wandb.Table(
    columns=["config", "mode", "gi_lo", "gi_hi", "churn", "FID", "R@3", "BLE", "FSR"],
    data=[[r["label"], r["mode"], r["gi_lo"], r["gi_hi"], r["churn"], r["FID"], r["R3"], r["BLE"], r["FSR"]] for r in rows])})
wandb.finish()
print("A1/A3 sweep done.")
