#!/usr/bin/env python
# ITERATED ROUND-TRIP DIAGNOSTIC  (Baggag's requested new experiment)
#
# Tests directly whether the frozen autoencoder REPEATEDLY reintroduces constraint error, connecting
# the single-round-trip autoencoder diagnostic to the in-ODE dose-response (where damage scaled with
# projection frequency).
#
# Loop, starting from real motions:
#   project to BLE=0  ->  [encode -> decode -> project]  repeated K times, measuring BLE after each DECODE
#   (i.e. BLE is measured on the decoded motion, BEFORE the re-projection of that round)
# This isolates whether each autoencoder pass destroys the correction that the preceding projection made.
# Two curves are reported:
#   (a) BLE after decode, WITH re-projection each round (mimics in-process projection: correct, then AE
#       damages it, repeatedly) -> should hover around the AE's ~0.006 floor (error reintroduced each time)
#   (b) BLE after decode, WITHOUT re-projection (plain iterated autoencoder) -> may drift/compound
# The (a) curve is the direct analogue of the in-ODE dose-response.
#
# Inference-only on existing checkpoints.

import os, sys, time
os.environ.setdefault("VARIANT","eval"); os.environ["USE_WANDB"]="0"; os.environ["ABLATION_IMPORT"]="1"
import numpy as np, torch, importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN); M=importlib.util.module_from_spec(spec); sys.modules["expmod"]=M
print(f"[roundtrip] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__!="M_ABLATION_STOP": raise
    print("[roundtrip] models loaded.")

DEVICE=M.DEVICE; rvq=M.rvq; MAXLEN=M.MAX_MOTION_LEN
EDGES=M.EDGES; rest_len=M.rest_len; lengths_to_mask=M.lengths_to_mask
_gj=M._gj; project_joints=M.project_joints; _joints_to_norm=M._joints_to_norm; pad_norm=M.pad_norm

def ble_of_norm(mn, L):
    J=_gj(mn)
    bone=(J[:,:,[e[0] for e in EDGES],:]-J[:,:,[e[1] for e in EDGES],:]).norm(dim=-1)
    fm=lengths_to_mask(L,MAXLEN).float().unsqueeze(-1)
    err=(bone-rest_len).abs()*fm
    return (err.sum(dim=(1,2))/(fm.sum(dim=(1,2))*len(EDGES)+1e-6))   # per-clip BLE

def project_norm(mn, L):
    J=project_joints(_gj(mn), L); return _joints_to_norm(J, mn)

K=int(os.environ.get("RT_ROUNDS","8"))      # number of round trips
N=int(os.environ.get("RT_N","512"))
import random as _r; _r.seed(0)
idx=_r.sample(range(len(M.test_entries)), min(N,len(M.test_entries)))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN","iterated_roundtrip"), job_type="diagnostic")

# with-reprojection (a) and without-reprojection (b), both starting from projected BLE=0
ble_reproj=[[] for _ in range(K+1)]     # index 0 = after initial projection (should be 0)
ble_plain =[[] for _ in range(K+1)]

with torch.no_grad():
    for s in range(0,len(idx),64):
        b=idx[s:s+64]
        mn0=torch.stack([torch.tensor(pad_norm(M.test_entries[i]["motion"])[0],device=DEVICE) for i in b])
        L=torch.tensor([min(len(M.test_entries[i]["motion"]),MAXLEN) for i in b],device=DEVICE)

        # ---- curve (a): project, then repeat [encode->decode->project], measuring BLE after each decode ----
        mn=project_norm(mn0,L)                       # start at BLE=0
        ble_reproj[0].append(ble_of_norm(mn,L).cpu())
        cur=mn
        for k in range(1,K+1):
            dec=rvq.decoder(rvq.encoder(cur))        # one AE round trip
            ble_reproj[k].append(ble_of_norm(dec,L).cpu())   # BLE measured on decoded (before re-projection)
            cur=project_norm(dec,L)                  # re-project for the next round (the "correct each step" part)

        # ---- curve (b): project once, then plain iterated AE (no re-projection) ----
        mn=project_norm(mn0,L)
        ble_plain[0].append(ble_of_norm(mn,L).cpu())
        cur=mn
        for k in range(1,K+1):
            cur=rvq.decoder(rvq.encoder(cur))
            ble_plain[k].append(ble_of_norm(cur,L).cpu())

def agg(rows): 
    a=torch.cat(rows).numpy(); return float(a.mean()), float(np.percentile(a,95))

print(f"\n{'='*76}\nITERATED ROUND-TRIP  (start BLE=0, N={len(idx)} clips, {K} rounds)\n{'='*76}")
print(f" {'round':>6}{'BLE mean (reproject each round)':>34}{'BLE mean (plain AE, no reproj)':>32}")
print("-"*76)
rt_means=[]; pl_means=[]
for k in range(K+1):
    rm,rp=agg(ble_reproj[k]); pm,pp=agg(ble_plain[k])
    rt_means.append(rm); pl_means.append(pm)
    tag = "(initial projection)" if k==0 else ""
    print(f" {k:>6}{rm:>28.5f}    {pm:>28.5f}   {tag}")
    wandb.log({"roundtrip/reproject_BLE_mean":rm, "roundtrip/plain_BLE_mean":pm, "roundtrip/round":k})
print("="*76)

print(f"\nInterpretation:")
print(f"  (a) with re-projection each round: BLE returns to ~{np.mean(rt_means[1:]):.4f} after every decode,")
print(f"      i.e. each autoencoder pass reintroduces the constraint error the projection just removed.")
print(f"      This is the direct analogue of the in-ODE dose-response: correcting inside the loop cannot")
print(f"      stick, because the very next AE pass destroys it. Round-0 (initial projection) BLE = {rt_means[0]:.5f}.")
print(f"  (b) plain iterated AE (no re-projection): BLE = {pl_means[1]:.4f} after 1 pass, "
      f"{'compounding to '+format(pl_means[-1],'.4f')+' by round '+str(K) if pl_means[-1]>pl_means[1]*1.2 else 'roughly stable ~'+format(pl_means[-1],'.4f')}.")

tbl=wandb.Table(columns=["round","BLE_reproject_mean","BLE_plain_mean"],
                data=[[k,round(rt_means[k],5),round(pl_means[k],5)] for k in range(K+1)])
wandb.log({"iterated_roundtrip_table":tbl})
# curves
wandb.log({"BLE_vs_rounds_reproject":wandb.plot.line(wandb.Table(data=[[k,rt_means[k]] for k in range(K+1)],columns=["round","BLE"]),"round","BLE",title="BLE vs round trips (re-project each round)")})
wandb.log({"BLE_vs_rounds_plain":wandb.plot.line(wandb.Table(data=[[k,pl_means[k]] for k in range(K+1)],columns=["round","BLE"]),"round","BLE",title="BLE vs round trips (plain AE)")})
wandb.finish()
print(f"\nIterated round-trip diagnostic done. Table + curves logged to W&B.")
