#!/usr/bin/env python
# INFERENCE-TIME FID ABLATION — no retraining, existing checkpoints.
# Assesses cheap sampling-time tricks against the unconstrained baseline:
#   (A) CFG std-rescaling         (Lin et al. — prevents CFG over-saturation)
#   (B) non-uniform timestep grid (cosine / power — more steps near t=1)
#   (C) Heun 2nd-order solver     (vs plain Euler)
#   (D) combined best
# Reported on the LATENT (LFM) base by default (its FID is the headline). Metric: FID (+ R@3, BLE).
# NOTE: these optimize the *baseline*; they are table-stakes, not the projection contribution.

import os, sys, time
os.environ.setdefault("VARIANT","eval"); os.environ["USE_WANDB"]="0"; os.environ["ABLATION_IMPORT"]="1"
import numpy as np, torch, importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN); M=importlib.util.module_from_spec(spec); sys.modules["expmod"]=M
print(f"[fid-ablation] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__!="M_ABLATION_STOP": raise
    print("[fid-ablation] models loaded.")

eval_variant=M.eval_variant; load_net=M.load_net; _agg=M._agg; ODE=M.ODE_STEPS
BASE=os.environ.get("ABL_BASE","latent"); IS_LAT = (BASE=="latent")
net=load_net(BASE, IS_LAT)

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN",f"fid_tricks_{BASE}"), job_type="ablation")

CONFIGS=[
    dict(label="baseline (euler, linear, no rescale)",           cfg_rescale=0.0, schedule="linear", solver="euler"),
    dict(label="CFG rescale 0.7",                                cfg_rescale=0.7, schedule="linear", solver="euler"),
    dict(label="CFG rescale 1.0",                                cfg_rescale=1.0, schedule="linear", solver="euler"),
    dict(label="cosine timesteps",                               cfg_rescale=0.0, schedule="cosine", solver="euler"),
    dict(label="power timesteps (dense near t=1)",               cfg_rescale=0.0, schedule="power",  solver="euler"),
    dict(label="Heun solver",                                    cfg_rescale=0.0, schedule="linear", solver="heun"),
    dict(label="Heun + cosine",                                  cfg_rescale=0.0, schedule="cosine", solver="heun"),
    dict(label="combined (Heun+cosine+rescale0.7)",              cfg_rescale=0.7, schedule="cosine", solver="heun"),
]

print(f"\n{'='*92}\nINFERENCE-TIME FID ABLATION — base={BASE}\n{'='*92}")
rows=[]
for c in CONFIGS:
    t0=time.time()
    r=eval_variant(net, IS_LAT, "none", cfg_rescale=c["cfg_rescale"], schedule=c["schedule"], solver=c["solver"])
    dt=time.time()-t0
    fm,fp,_=_agg(r["fsr"]); bm,bp,bx=_agg(r["ble"])
    row=dict(label=c["label"], FID=round(float(r["fid"]),4), R3=round(float(r["R3"]),4),
             BLE_mean=round(bm,5), sec=round(dt))
    print(f"  {c['label']:<42} FID={row['FID']:<9} R@3={row['R3']:<7} BLE={row['BLE_mean']:<8} ({row['sec']}s)")
    wandb.log({f"fidtrick/{c['label']}/FID":row["FID"], f"fidtrick/{c['label']}/R3":row["R3"]})
    rows.append(row)

print(f"\n{'='*92}")
base_fid=rows[0]["FID"]; best=min(rows,key=lambda r:r["FID"])
print(f" {'config':<42}{'FID':>9}{'R@3':>8}{'dFID':>9}")
print("-"*92)
for r in rows:
    d=r["FID"]-base_fid
    print(f" {r['label']:<42}{r['FID']:>9}{r['R3']:>8}{d:>+9.4f}")
print("="*92)
print(f"\nBaseline FID={base_fid} | Best FID={best['FID']} ({best['label']}) | improvement={base_fid-best['FID']:+.4f}")
print("NOTE: these are baseline-quality tricks (table stakes), not the projection contribution.")

tbl=wandb.Table(columns=["config","FID","R@3","BLE_mean"], data=[[r["label"],r["FID"],r["R3"],r["BLE_mean"]] for r in rows])
wandb.log({"fid_tricks_table":tbl})
wandb.finish()
print("\nInference-time FID ablation done. Table logged to W&B.")
