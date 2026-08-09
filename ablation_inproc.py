#!/usr/bin/env python
# CLFM in-process schedule ablation.
# Determines whether the CLFM in-process collapse (FID 55.9) is STRUCTURAL (ceiling evidence)
# or a TUNING ARTIFACT, by sweeping the projection schedule and watching FID.
#
# Sweep: projection window (final 10% / 20% / 50% of ODE steps) x stride (every step / every 2nd / every 4th),
# plus the reference points (unconstrained, post-hoc).
#
# It imports the main experiment module in eval mode (VARIANT=eval), which loads the frozen RVQ-VAE,
# the latent checkpoint, the evaluator and the test set, then re-runs sampling under each schedule.
#
# Run via run_ablation.sh (sbatch). Logs one W&B row per config + a summary table.

import os, sys, time, itertools
os.environ.setdefault("VARIANT","eval")     # load everything, train nothing
os.environ["USE_WANDB"]="0"                 # main module must NOT open its own W&B run
os.environ["ABLATION_IMPORT"]="1"           # tell the module to stop after models are loaded
import numpy as np, torch

# --- import the main module; it raises M_ABLATION_STOP once models/data/evaluator are ready ---
import importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN)
M=importlib.util.module_from_spec(spec)
sys.modules["expmod"]=M
print(f"[ablation] importing {MAIN} (loads RVQ-VAE, latent ckpt, evaluator, test set)...")
try:
    spec.loader.exec_module(M)
    print("[ablation] WARNING: module ran to completion without the ablation-stop sentinel.")
except Exception as e:
    if type(e).__name__=="M_ABLATION_STOP":
        print("[ablation] models loaded; taking over for the sweep.")
    else:
        raise

# --- pull what we need from the (partially-executed) module ---
sample=M.sample; eval_variant=M.eval_variant; load_net=M.load_net
CK=M.CK; _agg=M._agg; DEVICE=M.DEVICE

# --- our own W&B run for the ablation ---
import wandb
run=wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
               entity=(os.environ.get("WANDB_ENTITY") or None),
               name=os.environ.get("WANDB_RUN","clfm_inproc_ablation"),
               job_type="ablation")

# --- load the trained latent base once ---
assert M._have("latent"), "latent_best.pt not found under "+CK
net=load_net("latent",True)   # also restores z-stats into M.z_mean_t / M.z_std_t

def run_cfg(mode, window, stride, label):
    # set the schedule globals *in the module* so M.sample sees them
    M.PROJ_WINDOW=float(window); M.PROJ_STRIDE=int(stride)
    t0=time.time()
    r=eval_variant(net, True, mode)          # returns dict with fid, R3, fsr[], ble[]
    fm,fp,_=_agg(r["fsr"]); bm,bp,bx=_agg(r["ble"])
    dt=time.time()-t0
    row=dict(config=label, mode=mode, window=window, stride=stride,
             FID=round(float(r["fid"]),4), R3=round(float(r["R3"]),4),
             FSR_mean=round(fm,4), FSR_p95=round(fp,4),
             BLE_mean=round(bm,5), BLE_p95=round(bp,5), BLE_max=round(bx,5), sec=round(dt))
    print(f"  {label:<26} FID={row['FID']:<9} R@3={row['R3']:<7} FSR={row['FSR_mean']:<8} "
          f"BLE={row['BLE_mean']:<8} ({row['sec']}s)")
    wandb.log({f"ablation/{label}/FID":row["FID"], f"ablation/{label}/R3":row["R3"],
               f"ablation/{label}/FSR_mean":row["FSR_mean"], f"ablation/{label}/BLE_mean":row["BLE_mean"],
               f"inproc_FID":row["FID"], f"window":window, f"stride":stride})
    return row

print("\n"+"="*90)
print("CLFM IN-PROCESS SCHEDULE ABLATION  (latent base)")
print("="*90)
rows=[]
# reference points
rows.append(run_cfg("none",   0.0, 1, "unconstrained"))
rows.append(run_cfg("posthoc",0.0, 1, "post-hoc (final only)"))
# the sweep: window x stride
WINDOWS=[0.10,0.20,0.50]; STRIDES=[1,2,4]
for w,s in itertools.product(WINDOWS,STRIDES):
    lab=f"inproc w={int(w*100)}% k={s}"
    rows.append(run_cfg("inproc", w, s, lab))

# --- summary table ---
print("\n"+"="*90)
hdr=["config","FID","R@3","FSR mean","BLE mean","BLE max"]
print(f" {hdr[0]:<26}{hdr[1]:>9}{hdr[2]:>8}{hdr[3]:>10}{hdr[4]:>10}{hdr[5]:>10}")
print("-"*90)
for r in rows:
    print(f" {r['config']:<26}{r['FID']:>9}{r['R3']:>8}{r['FSR_mean']:>10}{r['BLE_mean']:>10}{r['BLE_max']:>10}")
print("="*90)

# verdict heuristic
inproc=[r for r in rows if r["mode"]=="inproc"]; best=min(inproc,key=lambda r:r["FID"])
unc=[r for r in rows if r["mode"]=="none"][0]
print(f"\nBest in-process config: {best['config']}  FID={best['FID']}")
print(f"Unconstrained FID={unc['FID']}, post-hoc FID={[r for r in rows if r['mode']=='posthoc'][0]['FID']}")
if best["FID"] < 3*unc["FID"]:
    print(">>> VERDICT: a gentler schedule RECOVERS in-process to a sane FID -> the 55.9 collapse is a")
    print("    TUNING ARTIFACT. In-process latent projection is viable with the right schedule.")
else:
    print(">>> VERDICT: in-process collapses under ALL schedules tested -> the effect is STRUCTURAL.")
    print("    Supports the ceiling argument: latent cannot enforce constraints from within the ODE.")

# log the table
tbl=wandb.Table(columns=["config","mode","window","stride","FID","R@3","FSR_mean","FSR_p95","BLE_mean","BLE_p95","BLE_max"],
                data=[[r["config"],r["mode"],r["window"],r["stride"],r["FID"],r["R3"],r["FSR_mean"],r["FSR_p95"],r["BLE_mean"],r["BLE_p95"],r["BLE_max"]] for r in rows])
wandb.log({"inproc_ablation_table":tbl})
# FID-vs-window line (at stride=1) for the dashboard
line=[[r["window"],r["FID"]] for r in rows if r["mode"]=="inproc" and r["stride"]==1]
if line:
    wandb.log({"FID_vs_window_k1":wandb.plot.line(wandb.Table(data=line,columns=["window","FID"]),"window","FID",title="in-process FID vs projection window (k=1)")})
wandb.finish()
print("\nAblation done. Table + curves logged to W&B.")
