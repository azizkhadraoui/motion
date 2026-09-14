#!/usr/bin/env python
"""
04_guidance_replicated.py — replicate the soft-guidance sweep.

WHY THIS EXISTS
  Every row of the published guidance sweep is a single draw. With the in-ODE
  hard-projection ablation suspended, that sweep is now the sole clean evidence for the
  in-trajectory claim, so the load-bearing result is held to a weaker standard than the
  result it replaced. The project's own rule is that replicated supersedes single-seed.
  This reruns the sweep with several replications and reports t-intervals.

  It reuses eval_variant directly, which already takes seed_offset and varies the
  sampling noise while holding the clip set fixed, so replications measure the same
  clips under different noise. Nothing is reimplemented.

OPTIONAL L1 ARM  (GR_L1=1)
  The guidance term descends a SQUARED bone penalty, whose gradient vanishes at the
  constraint boundary, so it cannot reach exact satisfaction in finite steps. Part of
  the observed plateau is therefore a property of the objective rather than of the
  representation. With GR_L1=1 the bone term is swapped for its unsquared form, whose
  subgradient has constant magnitude, and the sweep is repeated. If the plateau moves,
  the objective was contributing; if it does not, the reachability reading is
  strengthened.

  The swap is done by monkey-patching diff_penalty on the imported module, so the file
  on disk is untouched and no other experiment is affected.

USAGE
  WORK_DIR=$WORK/runs python 04_guidance_replicated.py
  env: GR_BASE=latent|direct  GR_REPS=3  GR_N=512  GR_L1=0
       GR_WEIGHTS=0.0,0.05,0.1,0.25,0.5,1.0,2.0
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("VARIANT", "eval")
os.environ["USE_WANDB"] = "0"
os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch

MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN)
M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[guid] importing {MAIN} ...", flush=True)
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP":
        raise
    print("[guid] models loaded.")

BASE = os.environ.get("GR_BASE", "latent")
REPS = int(os.environ.get("GR_REPS", "3"))
N = int(os.environ.get("GR_N", "512"))
USE_L1 = os.environ.get("GR_L1", "0") == "1"
WEIGHTS = [float(x) for x in os.environ.get(
    "GR_WEIGHTS", "0.0,0.05,0.1,0.25,0.5,1.0,2.0").split(",")]
WORK_DIR = os.environ.get("WORK_DIR", ".")
IS_LAT = (BASE == "latent")

net = M.load_net(BASE, IS_LAT)

# ---- optional L1 bone term -------------------------------------------------
if USE_L1:
    _EI, _EJ = M.EI, M.EJ
    def _diff_penalty_l1(mn, L):
        raw = mn * M.std_t + M.mean_t
        fm = M.lengths_to_mask(L, M.MAX_MOTION_LEN).float()
        J = M.recover_from_ric(raw)
        bone = (J[:, :, _EI, :] - J[:, :, _EJ, :]).norm(dim=-1)
        Lb = (((bone - M.rest_len).abs()).mean(-1) * fm).sum() / (fm.sum() + 1e-6)
        fj = J[:, :, M.FOOT_JOINTS, :]
        vel = (fj[:, 1:, :, [0, 2]] - fj[:, :-1, :, [0, 2]]).norm(dim=-1)
        ht = fj[:, 1:, :, 1]
        cw = (ht < M.PROJ_FOOT_H).float() * fm[:, 1:].unsqueeze(-1)
        Lf = (vel * cw).sum() / (cw.sum() + 1e-6)
        return M.PEN_BONE * Lb + M.PEN_FOOT * Lf
    M.diff_penalty = _diff_penalty_l1
    print("[guid] bone term swapped to the UNSQUARED (L1) form for this run only.")

TCRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 19: 2.093}


def ci95(v):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    t = TCRIT.get(len(v) - 1, 2.0)
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(len(v)))


tag = f"{BASE}{'_L1' if USE_L1 else ''}"
print(f"\n{'='*104}")
print(f"SOFT-GUIDANCE SWEEP, REPLICATED — {BASE}, {REPS} replications, {N} clips"
      f"{'  [L1 bone term]' if USE_L1 else ''}")
print("=" * 104, flush=True)
print(f" {'guide_w':>9}{'FID':>20}{'R@3':>20}{'BLE mean':>20}{'FSR':>18}")
print("-" * 104)

rows = []
for gw in WEIGHTS:
    per = {"fid": [], "r3": [], "ble": [], "fsr": []}
    t0 = time.time()
    for r in range(REPS):
        out = M.eval_variant(net, IS_LAT, "guided" if gw > 0 else "none",
                             N=N, guide_w=(gw if gw > 0 else None), seed_offset=r)
        per["fid"].append(out["fid"]); per["r3"].append(out["R3"])
        per["ble"].append(float(np.mean(out["ble"])))
        per["fsr"].append(float(np.mean(out["fsr"])))
    rec = dict(guide_w=gw, secs=round(time.time() - t0),
               **{k: dict(zip(("mean", "ci"), ci95(v))) for k, v in per.items()},
               per_rep=per)
    rows.append(rec)
    f, rr, b, s = rec["fid"], rec["r3"], rec["ble"], rec["fsr"]
    print(f" {gw:>9.2f}{f['mean']:>13.4f}+/-{f['ci']:<6.4f}"
          f"{rr['mean']:>13.4f}+/-{rr['ci']:<6.4f}"
          f"{b['mean']:>13.5f}+/-{b['ci']:<6.5f}"
          f"{s['mean']:>11.5f}+/-{s['ci']:<6.5f}", flush=True)

print("=" * 104)
base = rows[0]
worst = max(rows, key=lambda r: r["fid"]["mean"])
bb = min(rows, key=lambda r: r["ble"]["mean"])
print(f"\n  FID at guide_w=0: {base['fid']['mean']:.4f} +/- {base['fid']['ci']:.4f}")
print(f"  worst FID: {worst['fid']['mean']:.4f} at guide_w={worst['guide_w']} "
      f"({worst['fid']['mean']/max(base['fid']['mean'],1e-9):.1f}x the unguided value)")
print(f"  best BLE:  {bb['ble']['mean']:.5f} at guide_w={bb['guide_w']} "
      f"(unguided {base['ble']['mean']:.5f}, "
      f"{100*(1-bb['ble']['mean']/max(base['ble']['mean'],1e-9)):.1f}% improvement)")
print("""
  Quote the within-sweep baseline, not the replicated headline mean, for every ratio
  above: all rows here share one configuration, so the comparison is internally valid
  and mixing in the 20-seed number would repeat the baseline slip flagged in the audit.
""")

dst = os.path.join(WORK_DIR, f"guidance_replicated_{tag}.json")
json.dump(dict(base=BASE, l1=USE_L1, reps=REPS, n=N, rows=rows), open(dst, "w"), indent=2)
print(f"raw results -> {dst}")
