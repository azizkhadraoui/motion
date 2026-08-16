#!/usr/bin/env python
# IDEA 1 — "Where in-trajectory guidance fails: a geometric limit for latent flow models."
#
# The field's mainstream is inference-time GUIDANCE: add a constraint-gradient term to the
# pretrained velocity during ODE integration (Feng et al. 2025; GuideFlow; OmniGuide).
# Our hard-projection ablation already showed in-latent HARD projection collapses. This shows
# SOFT guidance ALSO fails in latent space -> the limit is the latent geometry, not our operator.
#
# For each base (latent LFM, direct CDFM) we sweep guidance weight GUIDE_W and report FID + BLE.
# Expected: LATENT degrades toward collapse / can't reach low BLE without wrecking FID;
#           DIRECT can be guided down to low BLE at modest FID cost (guidance in the native space).
# This is inference-only on existing checkpoints. Additive to the ICLR ceiling section.

import os, sys, time
os.environ.setdefault("VARIANT","eval")
os.environ["USE_WANDB"]="0"
os.environ["ABLATION_IMPORT"]="1"
import numpy as np, torch

import importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN); M=importlib.util.module_from_spec(spec); sys.modules["expmod"]=M
print(f"[idea1] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
    print("[idea1] WARNING: no ablation-stop sentinel.")
except Exception as e:
    if type(e).__name__!="M_ABLATION_STOP": raise
    print("[idea1] models loaded; taking over.")

eval_variant=M.eval_variant; load_net=M.load_net; CK=M.CK; _agg=M._agg

import wandb
run=wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
               entity=(os.environ.get("WANDB_ENTITY") or None),
               name=os.environ.get("WANDB_RUN","idea1_soft_guidance"), job_type="ablation")

GUIDE_WEIGHTS=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]   # 0.0 == unconstrained reference

def run_base(tag, is_latent):
    print(f"\n{'='*84}\nSOFT-GUIDANCE SWEEP — {tag} ({'latent' if is_latent else 'direct'})\n{'='*84}")
    net=load_net(tag, is_latent)
    rows=[]
    for gw in GUIDE_WEIGHTS:
        M.GUIDE_W=float(gw)
        mode = "none" if gw==0.0 else "guided"
        t0=time.time(); r=eval_variant(net, is_latent, mode); dt=time.time()-t0
        fm,fp,_=_agg(r["fsr"]); bm,bp,bx=_agg(r["ble"])
        row=dict(base=tag, guide_w=gw, FID=round(float(r["fid"]),4), R3=round(float(r["R3"]),4),
                 FSR_mean=round(fm,4), BLE_mean=round(bm,5), BLE_max=round(bx,5), sec=round(dt))
        tag_lbl=f"{tag} gw={gw}"
        print(f"  {tag_lbl:<22} FID={row['FID']:<9} R@3={row['R3']:<7} BLE_mean={row['BLE_mean']:<9} BLE_max={row['BLE_max']:<9} ({row['sec']}s)")
        wandb.log({f"guided/{tag}/FID":row["FID"], f"guided/{tag}/BLE_mean":row["BLE_mean"],
                   f"guided/{tag}/R3":row["R3"], f"guide_w":gw})
        rows.append(row)
    return rows

allrows=[]
allrows+=run_base("latent", True)    # LFM
allrows+=run_base("direct", False)   # CDFM

# summary
print("\n"+"="*84)
print(f" {'base':<8}{'guide_w':>9}{'FID':>10}{'R@3':>8}{'BLE_mean':>11}{'BLE_max':>10}")
print("-"*84)
for r in allrows:
    print(f" {r['base']:<8}{r['guide_w']:>9}{r['FID']:>10}{r['R3']:>8}{r['BLE_mean']:>11}{r['BLE_max']:>10}")
print("="*84)

# verdict
def base_rows(b): return [r for r in allrows if r["base"]==b]
def unc(b): return [r for r in base_rows(b) if r["guide_w"]==0.0][0]
for b in ["latent","direct"]:
    br=base_rows(b); u=unc(b)
    # best BLE achieved while keeping FID within 2x of unconstrained
    ok=[r for r in br if r["guide_w"]>0 and r["FID"]<=2*u["FID"]]
    best_ble = min([r["BLE_mean"] for r in ok], default=None)
    worst_fid = max(r["FID"] for r in br)
    print(f"\n[{b}] unconstrained FID={u['FID']} BLE={u['BLE_mean']}")
    if ok:
        print(f"   best BLE with FID<=2x baseline: {best_ble}  (guidance helps here)")
    else:
        print(f"   NO guidance weight reaches BLE improvement within 2x FID; FID rises to {worst_fid} -> guidance fails.")

lat=base_rows("latent"); dr=base_rows("direct")
lat_fails = all(r["FID"]>3*unc("latent")["FID"] for r in lat if r["guide_w"]>=1.0)
dir_ok    = any(r["BLE_mean"]<unc("direct")["BLE_mean"] and r["FID"]<=2*unc("direct")["FID"] for r in dr if r["guide_w"]>0)
print("\n"+"-"*84)
if lat_fails and dir_ok:
    print(">>> RESULT: soft guidance succeeds in DIRECT space but FAILS in LATENT space.")
    print("    Confirms the limit is the latent geometry (frozen decoder), not the projection")
    print("    operator: NO form of in-trajectory enforcement — hard or soft — works in latent.")
elif lat_fails:
    print(">>> RESULT: soft guidance fails in latent space (FID collapses under strong guidance).")
    print("    Consistent with the structural ceiling; direct-space result should be checked.")
else:
    print(">>> RESULT: latent soft guidance did NOT clearly collapse — nuance the ceiling claim.")
    print("    Inspect the FID/BLE tradeoff curve; the limit may be softer than hard projection suggested.")

tbl=wandb.Table(columns=["base","guide_w","FID","R@3","FSR_mean","BLE_mean","BLE_max"],
                data=[[r["base"],r["guide_w"],r["FID"],r["R3"],r["FSR_mean"],r["BLE_mean"],r["BLE_max"]] for r in allrows])
wandb.log({"soft_guidance_table":tbl})
for b in ["latent","direct"]:
    line=[[r["guide_w"],r["FID"]] for r in base_rows(b)]
    wandb.log({f"FID_vs_guidew_{b}":wandb.plot.line(wandb.Table(data=line,columns=["guide_w","FID"]),"guide_w","FID",title=f"{b}: FID vs guidance weight")})
    bl=[[r["guide_w"],r["BLE_mean"]] for r in base_rows(b)]
    wandb.log({f"BLE_vs_guidew_{b}":wandb.plot.line(wandb.Table(data=bl,columns=["guide_w","BLE_mean"]),"guide_w","BLE_mean",title=f"{b}: BLE vs guidance weight")})
wandb.finish()
print("\nIdea-1 experiment done. Table + curves logged to W&B.")
