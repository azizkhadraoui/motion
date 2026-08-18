#!/usr/bin/env python
# REPLICATED EVALUATION WITH 95% CONFIDENCE INTERVALS  (Baggag next-step #2)
#
# Every headline number so far is a single run. This re-runs each key variant across N replications,
# varying the SAMPLING seed (the real source of run-to-run variance) while keeping the clip set fixed,
# and reports mean +/- 95% CI on FID / BLE / R@3.
#
# It also tests the critical F2 claim ("projection improves FID") for significance, since the LFM
# margin (0.147 vs 0.142) is only ~0.005 and needs error bars to be defensible.
#
# Inference-only on existing checkpoints. Reuses the working eval harness (eval_variant), just looped
# over seeds with a per-replication seed offset.

import os, sys, time, math
os.environ.setdefault("VARIANT","eval"); os.environ["USE_WANDB"]="0"; os.environ["ABLATION_IMPORT"]="1"
import numpy as np, torch, importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN); M=importlib.util.module_from_spec(spec); sys.modules["expmod"]=M
print(f"[ci] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__!="M_ABLATION_STOP": raise
    print("[ci] models loaded.")

eval_variant=M.eval_variant; load_net=M.load_net; _agg=M._agg

N_REPS=int(os.environ.get("CI_REPS","20"))     # number of replications (seeds)

# the variants to replicate. keep it focused on the claim-bearing rows.
# each entry: (label, base_tag, is_latent, mode)
VARIANTS=[
    ("LFM (unconstrained)",       "latent", True,  "none"),
    ("CLFM + post-hoc proj",      "latent", True,  "posthoc"),
    ("CDFM (unconstrained)",      "direct", False, "none"),
    ("CDFM + post-hoc proj",      "direct", False, "posthoc"),
]

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN","replicated_ci"), job_type="evaluation")

def ci95(a):
    """mean and half-width of the 95% CI (t-based for small N)."""
    a=np.asarray(a,dtype=float); n=len(a); m=a.mean()
    if n<2: return m,float("nan")
    sd=a.std(ddof=1); se=sd/math.sqrt(n)
    # t critical for 95% two-sided; approx for common small n, fallback 1.96
    tcrit={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262,
           15:2.145,20:2.093,25:2.064,30:2.045}.get(n, 1.96 if n>30 else 2.093)
    return m, tcrit*se

# nets cached
nets={}
def get_net(tag,is_lat):
    if tag not in nets: nets[tag]=load_net(tag,is_lat)
    return nets[tag]

# collect per-rep metrics
results={lab:{"fid":[],"ble":[],"R3":[]} for lab,_,_,_ in VARIANTS}
print(f"\n{'='*88}\nREPLICATED EVALUATION — {N_REPS} reps, varied sampling seed, fixed clip set\n{'='*88}")
t_all=time.time()
for rep in range(N_REPS):
    for lab,tag,is_lat,mode in VARIANTS:
        net=get_net(tag,is_lat)
        r=eval_variant(net,is_lat,mode,seed_offset=rep)
        bm,_,_=_agg(r["ble"])
        results[lab]["fid"].append(float(r["fid"]))
        results[lab]["ble"].append(bm)
        results[lab]["R3"].append(float(r["R3"]))
    done=(rep+1)
    el=time.time()-t_all; eta=el/done*(N_REPS-done)
    print(f"  rep {done}/{N_REPS} done  (elapsed {el/60:.1f}m, eta {eta/60:.1f}m)")
    # log running means so progress is visible in W&B
    for lab in results:
        fm,fh=ci95(results[lab]["fid"]); wandb.log({f"ci/{lab}/FID_mean":fm}, step=done)

# ---- summary with CIs ----
print(f"\n{'='*88}")
print(f" {'variant':<26}{'FID (mean +/- 95%CI)':>26}{'BLE mean':>16}{'R@3':>14}")
print("-"*88)
summary={}
for lab,_,_,_ in VARIANTS:
    fm,fh=ci95(results[lab]["fid"]); bm,bh=ci95(results[lab]["ble"]); rm,rh=ci95(results[lab]["R3"])
    summary[lab]=dict(fid_m=fm,fid_ci=fh,ble_m=bm,ble_ci=bh,r3_m=rm,r3_ci=rh)
    print(f" {lab:<26}{fm:>10.4f} +/- {fh:<8.4f}{bm:>12.5f}    {rm:>8.4f}")
print("="*88)

# ---- F2 significance test: does post-hoc projection improve FID? (paired, per-rep) ----
def paired_test(unc_lab, proj_lab, name):
    u=np.array(results[unc_lab]["fid"]); p=np.array(results[proj_lab]["fid"])
    d=u-p   # positive => projection improves (lower FID)
    md=d.mean(); n=len(d)
    if n<2 or d.std(ddof=1)==0:
        print(f"  {name}: delta={md:+.4f} (insufficient variance for a test)"); return
    se=d.std(ddof=1)/math.sqrt(n); tstat=md/se
    tcrit={10:2.262,15:2.145,20:2.093,25:2.064,30:2.045}.get(n,2.093 if n<=30 else 1.96)
    ci_lo,ci_hi=md-tcrit*se, md+tcrit*se
    sig = (ci_lo>0) or (ci_hi<0)
    print(f"  {name}:")
    print(f"    mean FID improvement (unconstrained - projected) = {md:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    paired t = {tstat:+.2f} over {n} reps -> {'SIGNIFICANT (CI excludes 0)' if sig else 'NOT significant (CI includes 0)'}")

print("\nF2 significance — does post-hoc projection improve FID?")
paired_test("LFM (unconstrained)","CLFM + post-hoc proj","LFM: unconstrained vs post-hoc")
paired_test("CDFM (unconstrained)","CDFM + post-hoc proj","CDFM: unconstrained vs post-hoc")

# ---- log table ----
tbl=wandb.Table(columns=["variant","FID_mean","FID_CI95","BLE_mean","BLE_CI95","R3_mean","R3_CI95","n_reps"],
    data=[[lab,round(summary[lab]["fid_m"],4),round(summary[lab]["fid_ci"],4),
           round(summary[lab]["ble_m"],5),round(summary[lab]["ble_ci"],5),
           round(summary[lab]["r3_m"],4),round(summary[lab]["r3_ci"],4),N_REPS] for lab,_,_,_ in VARIANTS])
wandb.log({"replicated_ci_table":tbl})
# also dump raw per-rep values for reproducibility
for lab in results:
    wandb.log({f"raw/{lab}/fid":wandb.Table(data=[[i,v] for i,v in enumerate(results[lab]["fid"])],columns=["rep","FID"])})
wandb.finish()
print(f"\nReplicated evaluation done ({N_REPS} reps). Table + per-rep values logged to W&B.")
print("Interpretation: if an F2 CI excludes 0, projection's FID improvement is statistically real;")
print("if it includes 0, report the improvement as not significant (honest, and important to know).")
