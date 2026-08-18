#!/usr/bin/env python
# COMPOSITION ATTEMPT 2 (BOUNDED) — soft-mask blend in JOINT-POSITION space.
# The velocity-space binary-mask version failed (incoherent, tearing at the waist). This tries the
# single most likely fix, then STOPS regardless of outcome:
#   - generate each prompt's motion SEPARATELY (full clean trajectories, no velocity fighting)
#   - blend in absolute joint-position space with a FEATHERED (smooth) upper/lower mask
#   - re-encode blended positions to a valid 263-d motion via the projection's inverse-RIC path
# This removes the two worst problems: no discontinuity injected into the ODE, and a smooth spatial
# transition instead of a hard cut. It is NOT expected to fully solve composition (that needs trained
# modules); it is a bounded check whether a smarter blend is less broken.

import os, sys, time
os.environ.setdefault("VARIANT","eval"); os.environ["USE_WANDB"]="0"; os.environ["ABLATION_IMPORT"]="1"
import numpy as np, torch, importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN); M=importlib.util.module_from_spec(spec); sys.modules["expmod"]=M
print(f"[compose2] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__!="M_ABLATION_STOP": raise
    print("[compose2] models loaded.")

DEVICE=M.DEVICE; MAXLEN=M.MAX_MOTION_LEN; UNIT=M.UNIT_LEN; _CHAINS=M._CHAINS; N_JOINTS=M.N_JOINTS
_gj=M._gj; _joints_to_norm=M._joints_to_norm; EDGES=M.EDGES; rest_len=M.rest_len; project_joints=M.project_joints

BASE=os.environ.get("COMPOSE_BASE","direct"); IS_LAT=(BASE=="latent")
net=M.load_net(BASE, IS_LAT)

# self-contained GIF renderer (M.render_gif is post-sentinel)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, io as _io
from PIL import Image as _PImage
def _render_gif(J,path,color="#8e44ad",max_frames=40,fps=15,title=""):
    T=J.shape[0]; idx=np.linspace(0,T-1,min(max_frames,T)).astype(int); pad=0.4
    xmn,xmx=J[...,0].min()-pad,J[...,0].max()+pad; zmn,zmx=J[...,2].min()-pad,J[...,2].max()+pad
    ymn=min(J[...,1].min()-0.05,0.0); ymx=J[...,1].max()+0.3; span=max(xmx-xmn,zmx-zmn,1e-3); fr=[]
    for t in idx:
        fig=plt.figure(figsize=(2.6,2.6),dpi=70); ax=fig.add_subplot(111,projection="3d")
        for ch in _CHAINS: ax.plot([J[t,k,0] for k in ch],[J[t,k,2] for k in ch],[J[t,k,1] for k in ch],color=color,lw=2,marker="o",ms=2.5)
        ax.set_xlim(xmn,xmx); ax.set_ylim(zmn,zmx); ax.set_zlim(ymn,ymx); ax.set_box_aspect([1,1,max(0.4,(ymx-ymn)/span)])
        ax.view_init(elev=12,azim=60); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        if title: ax.set_title(title,fontsize=7)
        fig.tight_layout(pad=0.1); buf=_io.BytesIO(); fig.savefig(buf,format="png",dpi=70); buf.seek(0)
        fr.append(_PImage.open(buf).convert("RGB").copy()); buf.close(); plt.close(fig)
    fr[0].save(path,format="GIF",save_all=True,append_images=fr[1:],duration=int(1000/fps),loop=0)

# feathered upper-body weight per joint (1=upper, 0=lower, smooth around spine)
# spine chain [0,3,6,9,12,15]; arms hang off 9. Assign soft weights by "height" in the skeleton.
UPPER_CORE=[9,12,15,13,16,18,20,14,17,19,21]   # upper spine, head, arms -> weight ~1
LOWER_CORE=[0,1,2,4,7,10,5,8,11]               # pelvis, legs -> weight ~0
SPINE_MID=[3,6]                                 # lower/mid spine -> soft transition ~0.5
def joint_upper_weight():
    w=np.zeros(N_JOINTS,dtype=np.float32)
    for j in UPPER_CORE: w[j]=1.0
    for j in SPINE_MID:  w[j]=0.5      # feather
    return torch.tensor(w,device=DEVICE).view(1,1,N_JOINTS,1)   # (1,1,J,1)
WUP=joint_upper_weight()

@torch.no_grad()
def gen(prompt,length,seed=7):
    tq=M.embed_text([prompt]); tq=[torch.tensor(a,device=DEVICE) for a in tq]
    return M.sample(net,IS_LAT,tq[0],tq[1],tq[2],torch.tensor([length],device=DEVICE),seed=seed)

@torch.no_grad()
def compose_positions(pu,pl,length,seed=7):
    xU=gen(pu,length,seed); xL=gen(pl,length,seed)     # same seed -> aligned noise/global phase
    JU=_gj(xU); JL=_gj(xL)                              # (1,T,22,3) absolute joints
    # align global root (joint 0) of the upper motion onto the lower's, so upper rides the lower's locomotion
    JU_shift = JU - JU[:,:,0:1,:] + JL[:,:,0:1,:]
    Jc = WUP*JU_shift + (1-WUP)*JL                      # feathered per-joint blend
    Jc = project_joints(Jc, torch.tensor([length],device=DEVICE))   # enforce bone lengths on the blend
    # write blended joints back into a 263-d motion (use lower motion as the base for non-position channels)
    mnc = _joints_to_norm(Jc, xL)
    return mnc

PAIRS=[("a person waves both hands","a person walks forward"),
       ("a person raises arms overhead","a person walks in a circle"),
       ("a person crosses arms","a person jogs forward")]
FIG=M.FIG; _cdir=os.path.join(FIG,"compose2"); os.makedirs(_cdir,exist_ok=True)
import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN","compose2_softmask"), job_type="demo")

print(f"\n{'='*80}\nCOMPOSITION ATTEMPT 2 — soft-mask position blend (base={BASE})\n{'='*80}")
for pi,(pu,pl) in enumerate(PAIRS):
    L=(min(MAXLEN,140)//UNIT)*UNIT
    mnc=compose_positions(pu,pl,L,seed=7)
    J=_gj(mnc)[0,:L].cpu().numpy()
    bone=np.linalg.norm(J[:,[e[0] for e in EDGES],:]-J[:,[e[1] for e in EDGES],:],axis=-1)
    ble=float(np.abs(bone-rest_len.cpu().numpy()).mean())
    # crude coherence proxies: vertical bounce of pelvis (locomotion present?) and arm motion range (upper active?)
    pelvis_y=J[:,0,1]; arm_range=float(np.linalg.norm(J[:,20,:]-J[:,21,:],axis=-1).std()+np.abs(J[:,20,1]-J[:,20,1].mean()).mean())
    gp=os.path.join(_cdir,f"c2_{pi}.gif"); _render_gif(J,gp,title=f"U:{pu[:18]}/L:{pl[:18]}")
    print(f"  [{pi}] U='{pu}' L='{pl}'  BLE={ble:.4f}  pelvis_y_range={pelvis_y.ptp():.3f}  arm_activity={arm_range:.3f}")
    try: wandb.log({f"compose2/pair{pi}": wandb.Video(gp,caption=f"U:{pu}|L:{pl}",format="gif")})
    except Exception:
        try: wandb.log({f"compose2/pair{pi}": wandb.Image(gp)})
        except Exception as e: print("   wandb log failed:",e)
wandb.finish()
print(f"\nAttempt 2 done. GIFs under {_cdir}, logged to W&B.")
print("BOUNDED: inspect the GIFs. If still incoherent, composition is closed — move to replicated CIs.")
