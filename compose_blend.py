#!/usr/bin/env python
# COMPOSITIONAL VELOCITY BLENDING — zero-training, inference-time.
# Combine two prompts' velocity fields with a body-part mask at each ODE step:
#   v_composite = M * v(x, c_upper) + (1-M) * v(x, c_lower)
# e.g. upper body follows "waving hands", lower body follows "walking forward".
# Demonstrates compositionality of the flow-matching formulation (a workshop-flavored capability).
#
# HONEST CAVEAT built into the method: the 263-d HumanML3D vector interleaves root motion,
# joint positions (RIC dims 4:67), rotations (~67:193) and velocities. A clean spatial mask
# exists over the POSITION dims (per-joint), so we blend there; root/global channels are taken
# from the LOWER-body prompt (locomotion drives global trajectory). This is a demonstration of
# compositional control, not a claim of perfect part-disentanglement.

import os, sys, time
os.environ.setdefault("VARIANT","eval"); os.environ["USE_WANDB"]="0"; os.environ["ABLATION_IMPORT"]="1"
import numpy as np, torch, importlib.util
MAIN=os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"lfm_clfm_cdfm_experiment.py"))
spec=importlib.util.spec_from_file_location("expmod",MAIN); M=importlib.util.module_from_spec(spec); sys.modules["expmod"]=M
print(f"[compose] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__!="M_ABLATION_STOP": raise
    print("[compose] models loaded.")

DEVICE=M.DEVICE; NFEATS=M.NFEATS; N_JOINTS=M.N_JOINTS; T5_MAXLEN=M.T5_MAXLEN
ODE=M.ODE_STEPS; GUID=M.GUIDANCE; MAXLEN=M.MAX_MOTION_LEN; UNIT=M.UNIT_LEN
_CHAINS=M._CHAINS

# self-contained GIF renderer (M.render_gif is defined after the ablation sentinel, so unavailable here)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, io as _io
from PIL import Image as _PImage
def _render_gif(J,path,title="",color="#8e44ad",max_frames=40,fps=15):
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

BASE=os.environ.get("COMPOSE_BASE","direct"); IS_LAT=(BASE=="direct" and False) or (BASE=="latent")
net=M.load_net(BASE, IS_LAT)

# --- body-part joint sets from the kinematic chains ---
UPPER_JOINTS=sorted(set([9,12,15,13,16,18,20,14,17,19,21]))   # spine-top, head, both arms
LOWER_JOINTS=sorted(set([0,1,2,4,5,7,8,10,11,3,6]))            # pelvis, both legs, lower spine
# Build a 263-d mask that is 1 on UPPER position dims, 0 elsewhere. RIC positions live at 4:67
# as 21 joints x 3 (joint 0 = root, not in RIC positions block per recover_from_ric).
def part_mask_263(upper=True):
    m=torch.zeros(NFEATS,device=DEVICE)
    # RIC position block: dims 4:67 -> joints 1..21 (each 3). map joint j (1..21) -> [4+(j-1)*3 : +3]
    js = UPPER_JOINTS if upper else LOWER_JOINTS
    for j in js:
        if 1<=j<=21:
            a=4+(j-1)*3; m[a:a+3]=1.0
    return m  # 1 on this part's position dims

UP=part_mask_263(upper=True)      # upper-body position dims
# lower prompt drives everything not owned by upper (incl. root/global/rotation/velocity)
LO=1.0-UP

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN",f"compose_{BASE}"), job_type="demo")

@torch.no_grad()
def sample_composed(prompt_upper, prompt_lower, length, seed=7, n=ODE):
    tqU=M.embed_text([prompt_upper]); tqL=M.embed_text([prompt_lower])
    tqU=[torch.tensor(a,device=DEVICE) for a in tqU]; tqL=[torch.tensor(a,device=DEVICE) for a in tqL]
    ns=net.null_seq.unsqueeze(0); nm=torch.ones(1,T5_MAXLEN,dtype=torch.bool,device=DEVICE); npl=net.null_pool.unsqueeze(0)
    torch.manual_seed(seed); z=torch.randn(1,net.Tlen,net.cd,device=DEVICE); dt=1.0/n
    L=torch.tensor([length],device=DEVICE)
    for i in range(n):
        t=torch.full((1,),i*dt,device=DEVICE)
        def vel(tq):
            v=net(z,t,tq[0],tq[1],tq[2],L)
            if GUID!=1.0: vu=net(z,t,ns,nm,npl,L); v=vu+GUID*(v-vu)
            return v
        vU=vel(tqU); vL=vel(tqL)
        if IS_LAT:
            # in latent space we can't mask by body part (codes aren't per-joint); decode->blend->encode
            mU=M.rvq.decoder(z*M.z_std_t+M.z_mean_t)  # (not used directly; latent blend falls back to convex)
            v = 0.5*(vU+vL)   # latent: convex blend only (documented limitation)
        else:
            v = UP.view(1,1,-1)*vU + LO.view(1,1,-1)*vL   # direct: true body-part mask on 263-d
        z=z+dt*v
    mn=(M.rvq.decoder(z*M.z_std_t+M.z_mean_t) if IS_LAT else z)
    return mn

# demo prompt pairs (upper action + lower locomotion)
PAIRS=[
    ("a person waves both hands",          "a person walks forward"),
    ("a person raises arms overhead",      "a person walks in a circle"),
    ("a person crosses arms",              "a person jogs forward"),
]
FIG=M.FIG; import os as _os; _cdir=_os.path.join(FIG,"compose"); _os.makedirs(_cdir,exist_ok=True)
print(f"\n{'='*80}\nCOMPOSITIONAL BLENDING — base={BASE} ({'latent: convex only' if IS_LAT else 'direct: body-part mask'})\n{'='*80}")
for pi,(pu,pl) in enumerate(PAIRS):
    L=(min(MAXLEN,140)//UNIT)*UNIT
    x=sample_composed(pu,pl,L,seed=7)
    J=M._gj(x)[0,:L].cpu().numpy()
    # physical sanity of the composed motion
    bone=np.linalg.norm(J[:,[e[0] for e in M.EDGES],:]-J[:,[e[1] for e in M.EDGES],:],axis=-1)
    ble=float(np.abs(bone-M.rest_len.cpu().numpy()).mean())
    gp=_os.path.join(_cdir,f"compose_{pi}.gif")
    M_render = _render_gif
    M_render(J, gp, title=f"U:{pu[:20]} / L:{pl[:20]}", color="#8e44ad")
    print(f"  [{pi}] upper='{pu}'  lower='{pl}'  -> BLE={ble:.4f}  gif={gp}")
    try: wandb.log({f"compose/pair{pi}": wandb.Video(gp, caption=f"U:{pu} | L:{pl}", format="gif")})
    except Exception:
        try: wandb.log({f"compose/pair{pi}": wandb.Image(gp)})
        except Exception as e: print("   wandb log failed:",e)
wandb.finish()
print(f"\nCompositional blending done. GIFs under {_cdir}, logged to W&B.")
print("Note: direct (CDFM) supports a true per-joint velocity mask; latent (LFM) only convex blend,")
print("since RVQ codes are not per-joint — a clean point about where the direct representation helps.")
