#!/usr/bin/env python
# OPERATOR DECOMPOSITION UNDER THE REPLICATED PROTOCOL.
#
# WHY THIS IS NEEDED. Section 5.3 reports that post-decode correction lowers FID, and describes the
# foot-contact step separately as a heuristic without an exact guarantee. Single-seed screening
# (A7) suggests the FID credit is assigned to the wrong operator:
#
#     O0 no correction        FID 0.1473
#     O1 bone only            FID 0.1587     <- the EXACT analytic operator, on its own, is WORSE
#     O2 foot then bone       FID 0.1417     <- the reported operator
#
# If that holds under replication, the Pareto win comes from the heuristic foot correction, and the
# exact bone-length rescaling costs FID rather than providing it. The paper would still be correct
# that O2 lowers FID -- that is what the 20-seed table measures -- but the attribution in Section 5.3
# and the abstract would need to change, because a reader currently infers that exactness and the FID
# gain come from the same operation.
#
# DESIGN. Same protocol as the main replicated table: fixed clip set, 20 matched sampling seeds,
# every operator applied to the SAME decoded samples within a seed, so the comparisons are exactly
# paired and differ only in the correction. Reports FID/BLE/FSR/R@3 with t-based CIs, plus three
# paired tests that decompose the effect:
#
#     O2 - O0   total effect of the reported operator      (should reproduce the paper's 0.0148)
#     O1 - O0   effect of the exact bone operator alone
#     O2 - O1   effect of adding the foot correction
#
# O3 (foot, bone, foot) is included because it reached FSR exactly 0 in screening at the cost of
# BLE 0.00232. If that replicates it is a genuine finding about the two constraints: alternating the
# two projections does not converge into their intersection, which is evidence the bone-valid and
# contact-valid sets do not comfortably intersect in this representation.
#
#   sbatch run_decomp_ci.sh

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[decomp] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[decomp] models loaded.")

DEVICE = M.DEVICE; MAXLEN = M.MAX_MOTION_LEN; EVAL_N = M.EVAL_N
sample = M.sample; load_net = M.load_net; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; ble_pc_joints = M.ble_pc_joints; fsr_pc = M.fsr_pc
lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
project_foot = M.project_foot; project_bonelength = M.project_bonelength
_joints_to_norm = M._joints_to_norm

BASES = os.environ.get("DECOMP_BASES", "latent,direct").split(",")
N_REPS = int(os.environ.get("CI_REPS", "20"))
N_EVAL = int(os.environ.get("EVAL_N", str(EVAL_N)))

OPS = {
    "O0 none":               lambda J, L: J,
    "O1 bone only":          lambda J, L: project_bonelength(J),
    "O2 foot->bone":         lambda J, L: project_bonelength(project_foot(J, L)),
    "O3 foot->bone->foot":   lambda J, L: project_foot(project_bonelength(project_foot(J, L)), L),
}

TCRIT = {10:2.262,15:2.145,19:2.093,20:2.093,25:2.064,30:2.045}
def tcrit(n): return TCRIT.get(n - 1, 2.093 if n <= 31 else 1.96)
def ci95(a):
    a = np.asarray(a, float); n = len(a)
    return (a.mean(), tcrit(n) * a.std(ddof=1) / math.sqrt(n)) if n > 1 else (a.mean(), float("nan"))

def tsf(t, df):
    try:
        from scipy import stats; return 2.0 * float(stats.t.sf(abs(t), df))
    except Exception:
        return float("nan")

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "operator_decomposition"), job_type="evaluation")

out = {}
for tag in BASES:
    is_lat = (tag in ("latent", "latent_pen"))
    if not M._have(tag):
        print(f"[decomp] {tag} missing, skipped"); continue
    net = load_net(tag, is_lat)
    rng = np.random.default_rng(0)
    sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
    caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
    lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
    TSEQ, TMASK, TPOOL = embed_text(caps)

    res = {k: {"fid": [], "ble": [], "fsr": [], "R3": []} for k in OPS}
    print(f"\n{'='*104}\nOPERATOR DECOMPOSITION — {tag}, {N_REPS} seeds, {N_EVAL} clips\n{'='*104}")
    t0 = time.time()
    for rep in range(N_REPS):
        acc = {k: dict(fsr=[], ble=[], mf=[]) for k in OPS}; real_mf = []
        with torch.no_grad():
            for s in range(0, N_EVAL, 32):
                e = min(s + 32, N_EVAL)
                ts = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
                tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
                gm = lengths_to_mask(L, MAXLEN)
                x = sample(net, is_lat, ts, tm, tp, L, seed=s + rep * 100000)   # matched across operators
                J0 = _gj(x)
                for name, op in OPS.items():
                    J = op(J0, L)
                    xn = x if name == "O0 none" else _joints_to_norm(J, x)
                    Jm = _gj(xn)
                    acc[name]["fsr"].append(fsr_pc(Jm, L)); acc[name]["ble"].append(ble_pc_joints(Jm, L))
                    acc[name]["mf"].append(memb(xn * gm[..., None], L))
                rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
                real_mf.append(memb(rm * gm[..., None], L))
        R = np.concatenate(real_mf, 0)
        for name in OPS:
            G = np.concatenate(acc[name]["mf"], 0)
            res[name]["fid"].append(float(fid_calc(G, R)))
            res[name]["R3"].append(float(rprec(G, R)[3]))
            res[name]["ble"].append(float(np.concatenate(acc[name]["ble"]).mean()))
            res[name]["fsr"].append(float(np.concatenate(acc[name]["fsr"]).mean()))
        el = time.time() - t0
        print(f"  rep {rep+1}/{N_REPS}  " + "  ".join(f"{k.split()[0]}={res[k]['fid'][-1]:.4f}" for k in OPS)
              + f"   (elapsed {el/60:.1f}m, eta {el/(rep+1)*(N_REPS-rep-1)/60:.1f}m)")
        json.dump(res, open(os.path.join(os.environ.get("WORK_DIR", "."), f"decomp_raw_{tag}.json"), "w"))

    print(f"\n{'='*104}")
    print(f" {'operator':<22}{'FID (mean +/- 95CI)':>26}{'BLE':>12}{'FSR':>10}{'R@3':>10}")
    print("-" * 104)
    summ = {}
    for name in OPS:
        fm, fh = ci95(res[name]["fid"]); bm, _ = ci95(res[name]["ble"])
        sm, _ = ci95(res[name]["fsr"]); rm, _ = ci95(res[name]["R3"])
        summ[name] = dict(fid=fm, fid_ci=fh, ble=bm, fsr=sm, r3=rm)
        print(f" {name:<22}{fm:>15.4f} +/- {fh:<8.4f}{bm:>12.5f}{sm:>10.4f}{rm:>10.4f}")
    print("=" * 104)

    def paired(a, b, label):
        d = np.array(res[a]["fid"]) - np.array(res[b]["fid"])
        n = len(d); md = d.mean(); se = d.std(ddof=1) / math.sqrt(n); t = md / se if se else float("inf")
        lo, hi = md - tcrit(n) * se, md + tcrit(n) * se
        p = tsf(t, n - 1)
        sig = lo > 0 or hi < 0
        print(f"  {label}: mean paired dFID = {md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"t({n-1}) = {t:+.2f}  p = {p:.3g}  -> {'significant' if sig else 'not significant'}")
        return dict(delta=float(md), lo=float(lo), hi=float(hi), t=float(t), p=float(p))

    print("\nDecomposition (positive = the first operator has HIGHER FID, i.e. is worse):")
    st = {}
    st["bone_vs_none"] = paired("O1 bone only", "O0 none", "O1 - O0  exact bone operator alone")
    st["full_vs_none"] = paired("O2 foot->bone", "O0 none", "O2 - O0  reported operator (should match -0.0148)")
    st["full_vs_bone"] = paired("O2 foot->bone", "O1 bone only", "O2 - O1  effect of adding foot correction")

    print("\nREADING")
    if st["bone_vs_none"]["lo"] > 0:
        print("  The exact bone-length operator INCREASES FID on its own, significantly. Section 5.3 and")
        print("  the abstract must not imply that exact bone correction produces the FID improvement.")
        print("  Suggested framing: the terminal correction package lowers FID; decomposing it shows the")
        print("  gain comes from the foot-contact step, while the exact bone rescaling costs a small")
        print("  amount of FID and buys exactness. That is a more honest and more interesting claim --")
        print("  it separates 'exact' from 'better', which is the paper's own theme.")
    elif st["bone_vs_none"]["hi"] < 0:
        print("  The exact bone operator lowers FID on its own; the current attribution is fine.")
    else:
        print("  The bone operator's effect on FID is not distinguishable from zero. Then the honest")
        print("  statement is that exactness is obtained at no FID cost, and the improvement is the")
        print("  foot correction's -- still an attribution change from what is currently written.")
    o3 = summ.get("O3 foot->bone->foot")
    if o3:
        print(f"\n  O3 (foot, bone, foot): FSR {o3['fsr']:.4f}, BLE {o3['ble']:.5f}, FID {o3['fid']:.4f}.")
        print("  If FSR reaches ~0 while BLE becomes nonzero, the two corrections do not commute and")
        print("  alternating them does not converge into the intersection of the two constraint sets.")
        print("  Worth one sentence: exact bone validity and zero foot skate are not simultaneously")
        print("  attainable by alternating projection in this representation.")
    out[tag] = dict(summary=summ, paired=st, raw=res)

dst = os.path.join(os.environ.get("WORK_DIR", "."), "operator_decomposition.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nfull results -> {dst}")
wandb.finish()
print("Operator decomposition done.")
