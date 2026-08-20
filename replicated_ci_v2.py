#!/usr/bin/env python
# REPLICATED EVALUATION v2 — closes TODO 2, TODO 3 and TODO 8 in one job, and provides TODO 9.
#
# Differences from replicated_ci.py:
#   + the two TRAINING-TIME SOFT-PENALTY bases are included (TODO 3 asks for their numbers under the
#     same protocol; the paper must not call the penalty baseline dominated from its definition alone)
#   + FSR (foot skate) is collected alongside FID/BLE/R@3 (TODO 8 needs foot-contact numbers, and the
#     harness already computes them -- the previous script simply discarded them)
#   + paired statistics are computed IN-SCRIPT with a two-sided p-value, not just a CI (TODO 2)
#   + every per-rep value is written to JSON, so the statistics can be recomputed or re-tabulated
#     later without exporting anything from W&B
#   + EVAL_N is honoured, so the same script serves TODO 9 by running on the full available clip set
#
# Usage:
#   sbatch run_ci_v2.sh                       # 20 reps, EVAL_N=1024  -> TODO 2, 3, 8
#   CI_REPS=5 EVAL_N=2644 sbatch run_ci_v2.sh # full available test set -> TODO 9

import os, sys, time, math, json
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[ci2] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[ci2] models loaded.")

eval_variant = M.eval_variant; load_net = M.load_net; _agg = M._agg; _have = M._have
N_REPS = int(os.environ.get("CI_REPS", "20"))

try:
    from scipy import stats as _st
    def tsf(t, df): return 2.0 * float(_st.t.sf(abs(t), df))
except Exception:
    def tsf(t, df):                        # two-sided p via the incomplete beta, no scipy needed
        x = df / (df + t * t)
        def betacf(a, b, x, it=200):
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c, d = 1.0, 1.0 - qab * x / qap
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30); h = d
            for m in range(1, it):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
                c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30); h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
                c = 1.0 + aa / (c if abs(c) > 1e-30 else 1e-30)
                de = d * c; h *= de
                if abs(de - 1.0) < 1e-12: break
            return h
        a, b = df / 2.0, 0.5
        lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
        ib = math.exp(lb) * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else \
             1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                            + b * math.log(1 - x) + a * math.log(x)) * betacf(b, a, 1 - x) / b
        return float(min(1.0, max(0.0, ib)))

TCRIT = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262,
         12:2.201,15:2.145,20:2.093,25:2.064,30:2.045}
def tcrit(n): return TCRIT.get(n, 1.96 if n > 30 else 2.093)

def ci95(a):
    a = np.asarray(a, dtype=float); n = len(a); m = a.mean()
    if n < 2: return m, float("nan")
    return m, tcrit(n) * a.std(ddof=1) / math.sqrt(n)

# label, checkpoint tag, is_latent, mode, needed-for
VARIANTS = [
    ("LFM (unconstrained)",   "latent",     True,  "none",    "main"),
    ("LFM + penalty",         "latent_pen", True,  "none",    "todo3"),
    ("CLFM + post-hoc proj",  "latent",     True,  "posthoc", "main"),
    ("CDFM (unconstrained)",  "direct",     False, "none",    "main"),
    ("CDFM + penalty",        "direct_pen", False, "none",    "todo3"),
    ("CDFM + post-hoc proj",  "direct",     False, "posthoc", "main"),
]
VARIANTS = [v for v in VARIANTS if _have(v[1])]
missing = [v[1] for v in VARIANTS if not _have(v[1])]
print(f"[ci2] running {len(VARIANTS)} variants, {N_REPS} reps, EVAL_N={M.EVAL_N}")

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "replicated_ci_v2"), job_type="evaluation")

nets = {}
def get_net(tag, is_lat):
    if tag not in nets: nets[tag] = load_net(tag, is_lat)
    return nets[tag]

res = {lab: {"fid": [], "ble": [], "R3": [], "fsr": [], "fsr_p95": []} for lab, *_ in VARIANTS}
print(f"\n{'='*98}\nREPLICATED EVALUATION v2 — {N_REPS} reps, varied sampling seed, fixed clip set\n{'='*98}")
t0 = time.time()
for rep in range(N_REPS):
    for lab, tag, is_lat, mode, _ in VARIANTS:
        r = eval_variant(get_net(tag, is_lat), is_lat, mode, seed_offset=rep)
        fm, fp, _ = _agg(r["fsr"]); bm, _, _ = _agg(r["ble"])
        res[lab]["fid"].append(float(r["fid"])); res[lab]["ble"].append(bm)
        res[lab]["R3"].append(float(r["R3"])); res[lab]["fsr"].append(fm); res[lab]["fsr_p95"].append(fp)
    el = time.time() - t0
    print(f"  rep {rep+1}/{N_REPS}  (elapsed {el/60:.1f}m, eta {el/(rep+1)*(N_REPS-rep-1)/60:.1f}m)")
    json.dump(res, open(os.path.join(os.environ.get("WORK_DIR", "."), "replicated_ci_v2_raw.json"), "w"))

print(f"\n{'='*98}")
print(f" {'variant':<24}{'FID (mean +/- 95CI)':>24}{'BLE mean':>12}{'FSR mean':>11}{'FSR p95':>10}{'R@3':>9}")
print("-" * 98)
summary = {}
for lab, *_ in VARIANTS:
    fm, fh = ci95(res[lab]["fid"]); bm, bh = ci95(res[lab]["ble"])
    sm, sh = ci95(res[lab]["fsr"]); pm, _ = ci95(res[lab]["fsr_p95"]); rm, rh = ci95(res[lab]["R3"])
    summary[lab] = dict(fid=fm, fid_ci=fh, ble=bm, ble_ci=bh, fsr=sm, fsr_ci=sh, fsr_p95=pm, r3=rm, r3_ci=rh)
    print(f" {lab:<24}{fm:>13.4f} +/- {fh:<7.4f}{bm:>12.5f}{sm:>11.4f}{pm:>10.4f}{rm:>9.4f}")
print("=" * 98)

# ------------------------------------------------------------------ TODO 2: paired statistics
def paired(unc, proj, name):
    if unc not in res or proj not in res: return None
    u = np.array(res[unc]["fid"]); p = np.array(res[proj]["fid"]); d = u - p
    n = len(d); md = d.mean()
    if n < 2 or d.std(ddof=1) == 0:
        print(f"  {name}: delta={md:+.4f} (no variance)"); return None
    se = d.std(ddof=1) / math.sqrt(n); t = md / se
    lo, hi = md - tcrit(n) * se, md + tcrit(n) * se
    p_val = tsf(t, n - 1)
    print(f"  {name}")
    print(f"    mean paired FID difference (unconstrained - post-decode) = {md:+.4f}")
    print(f"    95% CI [{lo:+.4f}, {hi:+.4f}]   paired t({n-1}) = {t:+.3f}   p = {p_val:.3g}")
    print(f"    -> {'CI excludes 0: improvement is statistically significant' if lo>0 or hi<0 else 'CI includes 0: NOT significant; report as such'}")
    return dict(delta=float(md), ci_lo=float(lo), ci_hi=float(hi), t=float(t), p=float(p_val), n=n)

print("\nTODO 2 — paired FID comparison (matched sampling seeds):")
stats = {"LFM": paired("LFM (unconstrained)", "CLFM + post-hoc proj", "LFM: unconstrained vs post-decode"),
         "CDFM": paired("CDFM (unconstrained)", "CDFM + post-hoc proj", "CDFM: unconstrained vs post-decode")}

# ------------------------------------------------------------------ TODO 3: penalty verdict
print("\nTODO 3 — training-time soft-penalty baseline, same protocol:")
for base, pen, post in [("LFM (unconstrained)", "LFM + penalty", "CLFM + post-hoc proj"),
                        ("CDFM (unconstrained)", "CDFM + penalty", "CDFM + post-hoc proj")]:
    if pen not in summary: 
        print(f"  {pen}: checkpoint not found, skipped"); continue
    b, q, o = summary[base], summary[pen], summary[post]
    print(f"  {pen}: FID {q['fid']:.4f}+/-{q['fid_ci']:.4f} (vs {b['fid']:.4f} unconstrained), "
          f"BLE {q['ble']:.5f} (vs {b['ble']:.5f}), R@3 {q['r3']:.4f}")
    print(f"    exact satisfaction: {'NO' if q['ble']>0 else 'yes'};  post-decode reaches BLE 0 at FID {o['fid']:.4f}")

out = dict(n_reps=N_REPS, eval_n=int(M.EVAL_N), summary=summary, paired=stats, raw=res, missing=missing)
dst = os.path.join(os.environ.get("WORK_DIR", "."), "replicated_ci_v2.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nfull results -> {dst}")

tbl = wandb.Table(columns=["variant", "FID", "FID_CI95", "BLE", "FSR", "FSR_p95", "R@3", "n_reps"],
                  data=[[lab, round(summary[lab]["fid"], 4), round(summary[lab]["fid_ci"], 4),
                         round(summary[lab]["ble"], 5), round(summary[lab]["fsr"], 4),
                         round(summary[lab]["fsr_p95"], 4), round(summary[lab]["r3"], 4), N_REPS]
                        for lab, *_ in VARIANTS])
wandb.log({"replicated_ci_v2_table": tbl})
wandb.finish()
print("Replicated evaluation v2 done.")
