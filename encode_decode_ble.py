#!/usr/bin/env python
# ENCODE -> DECODE -> BLE DIAGNOSTIC
# The decisive test for the ceiling MECHANISM. Distinguishes two hypotheses:
#
#   (A) REPRESENTATIONAL: the frozen decoder's image does not contain BLE=0 motion at all.
#       Test: take REAL motions (BLE~0 by construction), encode -> decode, measure BLE.
#       If reconstructed BLE > 0, the autoencoder CANNOT output perfectly-rigid motion -> the
#       constraint set barely intersects the decoder's image (representational ceiling).
#
#   (B) REACHABILITY: constraint-satisfying motions ARE in the decoder's image, but the ODE
#       trajectory cannot reach the latents that produce them (shown separately by the guidance
#       sweep: soft guidance collapses FID while never reaching BLE=0).
#
# Plus: does PROJECTING then RE-ENCODING preserve validity? i.e. take a projected (BLE=0) motion,
# encode it, decode it — does BLE stay 0, or does the round trip reintroduce error? This directly
# measures whether the constraint-satisfying set survives one autoencoder round trip (the exact
# operation in-process projection depends on).

import os, sys, time
os.environ.setdefault("VARIANT","eval"); os.environ["USE_WANDB"]="0"; os.environ["ABLATION_IMPORT"]="1"
import numpy as np, torch, importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN); M=importlib.util.module_from_spec(spec); sys.modules["expmod"]=M
print(f"[enc-dec-ble] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__!="M_ABLATION_STOP": raise
    print("[enc-dec-ble] models loaded.")

DEVICE=M.DEVICE; rvq=M.rvq; MAXLEN=M.MAX_MOTION_LEN
mean_t=M.mean_t; std_t=M.std_t; z_mean_t=M.z_mean_t; z_std_t=M.z_std_t
EDGES=M.EDGES; rest_len=M.rest_len
pad_norm=M.pad_norm; _gj=M._gj; project_joints=M.project_joints; _joints_to_norm=M._joints_to_norm
lengths_to_mask=M.lengths_to_mask

def ble_of_joints(J, L):
    """mean bone-length error over valid frames, per clip (J: B,T,22,3)."""
    bone=(J[:,:,[e[0] for e in EDGES],:]-J[:,:,[e[1] for e in EDGES],:]).norm(dim=-1)  # B,T,E
    fm=lengths_to_mask(L,MAXLEN).float().unsqueeze(-1)                                  # B,T,1
    err=(bone-rest_len).abs()*fm
    return (err.sum(dim=(1,2))/ (fm.sum(dim=(1,2))*len(EDGES)+1e-8))                     # B

N=int(os.environ.get("ENCDEC_N","512"))
import random as _r; _r.seed(0)
idx=_r.sample(range(len(M.test_entries)), min(N,len(M.test_entries)))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN","encode_decode_ble"), job_type="diagnostic")

real_ble=[]; recon_ble=[]; proj_ble=[]; proj_roundtrip_ble=[]
print(f"\n{'='*84}\nENCODE -> DECODE -> BLE  (N={len(idx)} real test motions)\n{'='*84}")
with torch.no_grad():
    for s in range(0,len(idx),64):
        b=idx[s:s+64]
        mn=torch.stack([torch.tensor(pad_norm(M.test_entries[i]["motion"])[0],device=DEVICE) for i in b])  # B,196,263 normalized
        L=torch.tensor([min(len(M.test_entries[i]["motion"]),MAXLEN) for i in b],device=DEVICE)

        # (1) REAL motion BLE (should be ~0)
        Jr=_gj(mn); real_ble.append(ble_of_joints(Jr,L).cpu())

        # (2) encode -> decode the REAL motion, measure BLE (tests decoder's image)
        z=rvq.encoder(mn); rec=rvq.decoder(z)   # note: encoder/decoder operate on normalized 263
        Jrec=_gj(rec); recon_ble.append(ble_of_joints(Jrec,L).cpu())

        # (3) PROJECT the real motion to BLE=0, confirm it's ~0
        Jp=project_joints(_gj(mn),L); mnp=_joints_to_norm(Jp,mn)
        proj_ble.append(ble_of_joints(_gj(mnp),L).cpu())

        # (4) encode -> decode the PROJECTED (BLE=0) motion: does the round trip preserve BLE=0?
        zp=rvq.encoder(mnp); recp=rvq.decoder(zp)
        proj_roundtrip_ble.append(ble_of_joints(_gj(recp),L).cpu())

def cat(x): return torch.cat(x).numpy()
real=cat(real_ble); recon=cat(recon_ble); proj=cat(proj_ble); rt=cat(proj_roundtrip_ble)

def stats(a): return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()))
rows=[("real motion (input)",real),
      ("real -> encode -> decode",recon),
      ("real -> project (BLE=0 target)",proj),
      ("projected -> encode -> decode",rt)]
print(f"\n {'stage':<34}{'BLE mean':>10}{'BLE p95':>10}{'BLE max':>10}")
print("-"*84)
for name,a in rows:
    st=stats(a); print(f" {name:<34}{st['mean']:>10.5f}{st['p95']:>10.5f}{st['max']:>10.5f}")
print("="*84)

# VERDICT
rmean=recon.mean(); rtmean=rt.mean(); realmean=real.mean()
print(f"\nReal BLE={realmean:.5f} (baseline, ~0 expected)")
print(f"Autoencoder round-trip on REAL motion: BLE {realmean:.5f} -> {rmean:.5f}  (x{rmean/max(realmean,1e-6):.1f})")
print(f"Autoencoder round-trip on PROJECTED (BLE=0) motion: 0.0 -> {rtmean:.5f}")
print()
if rmean > 5*max(realmean,1e-5) or rmean > 0.003:
    print(">>> The frozen decoder INTRODUCES bone-length error on reconstruction.")
    print("    Even a perfectly-rigid input decodes to a NON-rigid output -> the decoder's image")
    print("    contains little/no BLE=0 motion. This is REPRESENTATIONAL: the constraint set barely")
    print("    intersects the decoder's image, so in-latent enforcement cannot reach BLE=0 from within.")
    print("    (Post-hoc still works because it corrects AFTER decoding, outside the autoencoder.)")
else:
    print(">>> The decoder roughly PRESERVES bone lengths on reconstruction.")
    print("    Constraint-satisfying motion IS in the decoder's image -> the ceiling is REACHABILITY:")
    print("    the good latents exist but the ODE trajectory / guidance cannot reach them without")
    print("    leaving the training distribution (per the guidance-sweep FID collapse).")
print()
print(f"Key number — projected motion after one AE round trip: BLE {rtmean:.5f}")
print("This is exactly the operation in-process projection repeats each step; the residual here")
print("compounds over the ODE, explaining the in-process collapse.")

tbl=wandb.Table(columns=["stage","BLE_mean","BLE_p95","BLE_max"],
                data=[[n,float(a.mean()),float(np.percentile(a,95)),float(a.max())] for n,a in rows])
wandb.log({"encode_decode_ble_table":tbl,
           "real_ble":realmean,"recon_ble":rmean,"proj_roundtrip_ble":rtmean})
wandb.finish()
print("\nEncode-decode-BLE diagnostic done. Table logged to W&B.")
