#!/usr/bin/env python
# A7 — POST-DECODE PROJECTION BEYOND BONE LENGTH, plus the kinematic plausibility panel.
#
# This is the idea that most directly strengthens the paper's own thesis: if post-decode correction is
# the right place to enforce geometry, it should extend beyond bone length to other cheap kinematic
# constraints. Everything here is post-processing on decoded motion -- no simulator, no training, and
# GPU cost is negligible next to sampling.
#
# OPERATORS COMPARED (each applied to the same decoded samples):
#   O0  none                       the unconstrained reference
#   O1  bone only                  project_bonelength: exact BLE=0, no foot handling
#   O2  foot then bone             your current project_joints (the paper's operator)
#   O3  foot, bone, foot again     re-apply the foot fix AFTER the bone rescale. Your own docstring
#                                  notes the final rescale slightly perturbs the foot, so this tests
#                                  whether one extra pass buys FSR back. BLE is no longer exactly 0
#                                  afterwards -- reported, not hidden.
#   O4  smooth, foot, bone         a 5-tap temporal box filter on joint positions before correcting,
#                                  targeting jitter (DisCoRD-style jerk metrics); bone rescale last
#                                  keeps BLE exact.
#   O5  ground clamp, foot, bone   lift any below-floor joint to y>=0 before correcting.
#
# METRICS: FID, R@3, BLE, FSR, plus two plausibility measures the paper currently never reports --
# JERK (mean third-derivative magnitude of joint positions, lower is smoother) and GROUND PENETRATION
# (mean depth below the floor plane). Real motion is measured on all of them as the reference row,
# so the table shows what the target actually is rather than only comparing variants to each other.
#
# CAUTION built in: an operator can improve one metric and wreck another (smoothing lowers jerk but
# can flatten real dynamics; foot fixes can distort global trajectory). The table shows all metrics
# side by side so no single number can be cherry-picked.
#
#   sbatch run_boost_postdecode.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[postdecode] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[postdecode] models loaded.")

DEVICE = M.DEVICE; MAXLEN = M.MAX_MOTION_LEN; EVAL_N = M.EVAL_N
sample = M.sample; load_net = M.load_net; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; ble_pc_joints = M.ble_pc_joints; fsr_pc = M.fsr_pc
lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
project_foot = M.project_foot; project_bonelength = M.project_bonelength
_joints_to_norm = M._joints_to_norm

BASE = os.environ.get("BOOST_BASE", "latent"); IS_LAT = (BASE in ("latent", "latent_pen"))
N_EVAL = int(os.environ.get("EVAL_N", str(EVAL_N)))
SMOOTH_K = int(os.environ.get("SMOOTH_K", "5"))

def smooth_joints(J, k=SMOOTH_K):
    """Centered temporal box filter on joint positions, replicate-padded at the ends."""
    B, T, Jn, C = J.shape
    x = J.permute(0, 2, 3, 1).reshape(B, Jn * C, T)
    pad = k // 2
    x = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    x = torch.nn.functional.avg_pool1d(x, kernel_size=k, stride=1)
    return x.reshape(B, Jn, C, T).permute(0, 3, 1, 2)

def ground_clamp(J):
    out = J.clone(); out[..., 1] = out[..., 1].clamp_min(0.0); return out

OPS = {
    "O0 none":                    lambda J, L: J,
    "O1 bone only":               lambda J, L: project_bonelength(J),
    "O2 foot->bone (current)":    lambda J, L: project_bonelength(project_foot(J, L)),
    "O3 foot->bone->foot":        lambda J, L: project_foot(project_bonelength(project_foot(J, L)), L),
    "O4 smooth->foot->bone":      lambda J, L: project_bonelength(project_foot(smooth_joints(J), L)),
    "O5 clamp->foot->bone":       lambda J, L: project_bonelength(project_foot(ground_clamp(J), L)),
}

def jerk(J, L):
    """Mean |third difference| of joint positions over valid frames (lower = smoother)."""
    d3 = J[:, 3:] - 3 * J[:, 2:-1] + 3 * J[:, 1:-2] - J[:, :-3]
    fm = lengths_to_mask(L, MAXLEN).float()[:, 3:].unsqueeze(-1)
    return float((d3.norm(dim=-1) * fm).sum() / (fm.sum() * J.shape[2] + 1e-8))

def penetration(J, L):
    """Mean depth below the floor plane over valid frames (lower = better)."""
    fm = lengths_to_mask(L, MAXLEN).float()                      # (B,T)
    depth = (-J[..., 1]).clamp_min(0.0)                          # (B,T,J)
    return float((depth * fm.unsqueeze(-1)).sum() / (fm.sum() * J.shape[2] + 1e-8))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", f"boost_postdecode_{BASE}"), job_type="ablation")

net = load_net(BASE, IS_LAT)
rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

acc = {k: dict(fsr=[], ble=[], jerk=[], pen=[], mf=[]) for k in OPS}
real = dict(fsr=[], ble=[], jerk=[], pen=[], mf=[])

print(f"\n{'='*112}\nA7 — POST-DECODE OPERATORS (base={BASE}, {N_EVAL} clips, smooth k={SMOOTH_K})\n{'='*112}")
t0 = time.time()
with torch.no_grad():
    for s in range(0, N_EVAL, 32):
        e = min(s + 32, N_EVAL)
        ts = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
        tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
        gm = lengths_to_mask(L, MAXLEN)
        x = sample(net, IS_LAT, ts, tm, tp, L, seed=s)          # sample ONCE, correct many ways
        J0 = _gj(x)
        for name, op in OPS.items():
            J = op(J0, L)
            xn = x if name == "O0 none" else _joints_to_norm(J, x)
            Jm = _gj(xn)
            a = acc[name]
            a["fsr"].append(fsr_pc(Jm, L)); a["ble"].append(ble_pc_joints(Jm, L))
            a["jerk"].append(jerk(Jm, L)); a["pen"].append(penetration(Jm, L))
            a["mf"].append(memb(xn * gm[..., None], L))
        rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
        Jr = _gj(rm)
        real["fsr"].append(fsr_pc(Jr, L)); real["ble"].append(ble_pc_joints(Jr, L))
        real["jerk"].append(jerk(Jr, L)); real["pen"].append(penetration(Jr, L))
        real["mf"].append(memb(rm * gm[..., None], L))
print(f"  sampling + correction done in {(time.time()-t0)/60:.1f}m")

R = np.concatenate(real["mf"], 0)
rows = []
for name in OPS:
    a = acc[name]; G = np.concatenate(a["mf"], 0)
    rows.append(dict(op=name, FID=round(float(fid_calc(G, R)), 4), R3=round(float(rprec(G, R)[3]), 4),
                     BLE=round(float(np.concatenate(a["ble"]).mean()), 5),
                     FSR=round(float(np.concatenate(a["fsr"]).mean()), 4),
                     JERK=round(float(np.mean(a["jerk"])), 5),
                     PEN=round(float(np.mean(a["pen"])), 5)))
real_row = dict(op="real motion (target)", FID=0.0, R3=float("nan"),
                BLE=round(float(np.concatenate(real["ble"]).mean()), 5),
                FSR=round(float(np.concatenate(real["fsr"]).mean()), 4),
                JERK=round(float(np.mean(real["jerk"])), 5),
                PEN=round(float(np.mean(real["pen"])), 5))

print(f"\n{'='*112}")
print(f" {'operator':<28}{'FID':>10}{'R@3':>8}{'BLE':>11}{'FSR':>10}{'JERK':>11}{'PENETR':>11}")
print("-" * 112)
for r in rows:
    print(f" {r['op']:<28}{r['FID']:>10}{r['R3']:>8}{r['BLE']:>11}{r['FSR']:>10}{r['JERK']:>11}{r['PEN']:>11}")
print("-" * 112)
print(f" {real_row['op']:<28}{'--':>10}{'--':>8}{real_row['BLE']:>11}{real_row['FSR']:>10}"
      f"{real_row['JERK']:>11}{real_row['PEN']:>11}")
print("=" * 112)

cur = [r for r in rows if r["op"].startswith("O2")][0]
print("\nREADING THE RESULT")
print(f"  Reference row is O2, the operator the paper already uses: FID {cur['FID']}, FSR {cur['FSR']}.")
for r in rows:
    if r["op"].startswith(("O0", "O2")): continue
    d = []
    if r["FID"] < cur["FID"] - 0.006: d.append(f"FID {r['FID']} better")
    if r["FID"] > cur["FID"] + 0.006: d.append(f"FID {r['FID']} WORSE")
    if r["FSR"] < cur["FSR"] * 0.9: d.append(f"FSR {r['FSR']} better")
    if r["JERK"] < cur["JERK"] * 0.9: d.append(f"jerk {r['JERK']} better")
    if r["BLE"] > 1e-9: d.append(f"BLE no longer exact ({r['BLE']})")
    print(f"  {r['op']:<28} {'; '.join(d) if d else 'no material change'}")
print("\n  An operator is only worth adopting if it improves a plausibility metric WITHOUT losing exact")
print("  BLE and without moving FID beyond the +/-0.006 seed band. O3 in particular trades exactness")
print("  for foot quality -- if you report it, report it as a distinct operator, not as the same one.")
print("  Single seed: screening only, as with the other boost sweeps.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), f"boost_postdecode_{BASE}.json")
json.dump(dict(rows=rows, real=real_row), open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.log({"postdecode_table": wandb.Table(columns=["operator", "FID", "R@3", "BLE", "FSR", "JERK", "PENETR"],
                                           data=[[r["op"], r["FID"], r["R3"], r["BLE"], r["FSR"], r["JERK"], r["PEN"]] for r in rows])})
wandb.finish()
print("A7 sweep done.")
