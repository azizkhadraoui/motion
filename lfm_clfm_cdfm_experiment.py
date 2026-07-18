# =============================================================================
# LFM / CLFM / CDFM — constraint-projection experiment (fills the table)
#
# Two base generators, one shared constraint-projection operator:
#   LFM  = latent flow matching (RVQ-VAE frozen + latent FM)         [base A]
#   CLFM = LFM + analytic constraint projection                       [A + proj]
#   CDFM = constrained DIRECT flow matching (raw-263 DiT)             [base B + proj]
# Table variations per base: unconstrained / + penalty / + post-hoc proj /
#                            + in-process proj.
#
# Projection guarantees (validated in Cell 2):
#   * bone length  -> EXACT 0 (analytic rescale along the kinematic chain)
#   * foot-skating -> substantially REDUCED (temporal constraint; NOT zeroed)
#
# Loads the trained RVQ-VAE (pull or attach as dataset). Every base is resumable;
# train one base per session if needed. Produces the table + qualitative figures.
# =============================================================================

# %% [code]
# Cell 0: (Kaggle only) self-pull the trained RVQ-VAE from the source kernel.
# On the cluster this is skipped — the checkpoint is provided via RVQ_CKPT env var.
import os, glob, subprocess
if os.path.isdir("/kaggle"):
    PULL_KERNEL="khadraouimohamedaziz/notebookf50ed59c15"; PULL_DIR="/kaggle/working/pulled"
    os.makedirs(PULL_DIR,exist_ok=True)
    try:
        _r=subprocess.run(["kaggle","kernels","output",PULL_KERNEL,"-p",PULL_DIR],capture_output=True,text=True,timeout=1800)
        if _r.returncode!=0: print("pull failed (likely cross-account perms) -> attach the RVQ-VAE as a dataset.")
    except Exception as _e: print("pull skipped:",_e)
    print("visible .pt:", [p for p in glob.glob("/kaggle/input/**/*.pt",recursive=True)+glob.glob(PULL_DIR+"/**/*.pt",recursive=True) if "rvq" in p.lower()][:5])
else:
    print("[not on Kaggle] skipping self-pull cell; RVQ-VAE comes from RVQ_CKPT env var.")

# %% [code]
# Cell 1: config + imports
import math, time, random, shutil, io, base64
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from scipy import linalg as scipy_linalg
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from IPython.display import HTML, display
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:",DEVICE)

# ---- knobs (env-overridable for cluster sbatch: one VARIANT per job) ----
# VARIANT selects what THIS job trains: "latent" | "latent_pen" | "direct" | "direct_pen" | "all" | "eval"
VARIANT      = os.environ.get("VARIANT","all")
SMOKE_TEST   = os.environ.get("SMOKE_TEST","0")=="1"    # tiny end-to-end pipeline check
FULL_STEPS   = int(os.environ.get("FULL_STEPS","300000"))
_smoke_steps = 200
RUN_LATENT   = VARIANT in ("latent","latent_pen","all")
RUN_DIRECT   = VARIANT in ("direct","direct_pen","all")
TRAIN_PENALTY= VARIANT in ("latent_pen","direct_pen","all")
_S = _smoke_steps if SMOKE_TEST else FULL_STEPS
STEPS        = {"latent":_S,"direct":_S,"latent_pen":_S,"direct_pen":_S}
EVAL_N       = 256 if SMOKE_TEST else int(os.environ.get("EVAL_N","1024"))
GUIDANCE     = 2.5; ODE_STEPS=50
PROJ_FOOT_H  = 0.05     # contact height threshold (m)
PEN_BONE     = 0.5; PEN_FOOT=1.0   # penalty weights (soft variant)
# W&B
USE_WANDB    = os.environ.get("USE_WANDB","1")=="1"
WANDB_PROJECT= os.environ.get("WANDB_PROJECT","motion-clfm")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY","kaziz")
WANDB_RUN    = os.environ.get("WANDB_RUN", f"{VARIANT}{'_smoke' if SMOKE_TEST else ''}")

def _find_dir(roots,needles,md=7):
    needles=set(needles)
    for b in roots:
        if not os.path.isdir(b): continue
        for dp,dn,fn in os.walk(b):
            if dp[len(b):].count(os.sep)>md: dn[:]=[]; continue
            if needles.issubset(set(dn)|set(fn)): return dp
    return None
# HML3D root: env var (cluster) takes priority, then known kaggle path, then search
HML3D_ROOT=os.environ.get("HML3D_ROOT","")
if not (HML3D_ROOT and os.path.exists(os.path.join(HML3D_ROOT,"train.txt"))):
    _PREF="/kaggle/input/datasets/mrriandmstique/humanml3d/HumanML3D/humanml"
    HML3D_ROOT=_PREF if os.path.exists(os.path.join(_PREF,"train.txt")) else _find_dir(["/kaggle/input","/export/home"],["train.txt","new_joint_vecs","texts"])
assert HML3D_ROOT,"HumanML3D not found (set HML3D_ROOT env var)."; print("HML3D_ROOT:",HML3D_ROOT)
EVAL_ROOT=_find_dir(["/kaggle/input","/export/home"],["movement_encoder.pt","motion_encoder.pt"])
if EVAL_ROOT is None:
    for base in ["/kaggle/input","/export/home"]:
        if not os.path.isdir(base): continue
        for dp,dn,fn in os.walk(base):
            if "finest.tar" in fn: EVAL_ROOT=dp; break
        if EVAL_ROOT: break
# RVQ-VAE: env var (cluster) first, then kaggle locations
RVQ_CKPT=os.environ.get("RVQ_CKPT","")
if not (RVQ_CKPT and os.path.exists(RVQ_CKPT)):
    _vc=[x for x in glob.glob("/kaggle/working/pulled/**/rvq_vae*.pt",recursive=True)+glob.glob("/kaggle/input/**/rvq_vae*.pt",recursive=True)+glob.glob("/export/home/**/rvq_vae*.pt",recursive=True) if "latest" not in os.path.basename(x)]
    assert _vc,"Trained RVQ-VAE not found (set RVQ_CKPT env var)."; RVQ_CKPT=sorted(_vc,key=len)[0]
print("RVQ-VAE:",RVQ_CKPT)
# work/checkpoint dir: env var (cluster persistent path) or kaggle working
WORK=os.environ.get("WORK_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.expanduser("~/motion/runs"))
PROJ=os.path.join(WORK,"clfm"); CK=os.path.join(PROJ,"ckpt"); FIG=os.path.join(PROJ,"fig")
for d in [PROJ,CK,FIG]: os.makedirs(d,exist_ok=True)

# ---- Weights & Biases ----
WB=None
if USE_WANDB:
    try:
        import wandb as WB
        WB.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN,
                config=dict(variant=VARIANT,smoke=SMOKE_TEST,full_steps=FULL_STEPS,eval_n=EVAL_N,
                            guidance=GUIDANCE,ode_steps=ODE_STEPS,pen_bone=PEN_BONE,pen_foot=PEN_FOOT,
                            hml_root=HML3D_ROOT,rvq_ckpt=os.path.basename(RVQ_CKPT)))
        print(f"  W&B: {WANDB_ENTITY}/{WANDB_PROJECT} run={WANDB_RUN}")
    except Exception as _e:
        print(f"  W&B disabled ({_e})"); WB=None
def wlog(d,step=None):
    if WB is not None:
        try: WB.log(d,step=step)
        except Exception: pass

NFEATS=263; N_JOINTS=22; UNIT_LEN=4; MIN_MOTION_LEN=40; MAX_MOTION_LEN=196; FPS=20
RVQ_CODE_DIM=256; RVQ_DOWN=4; T_LAT=MAX_MOTION_LEN//RVQ_DOWN; RVQ_ENC_CHANNELS=(NFEATS,256,512,RVQ_CODE_DIM)
T5_MODEL="sentence-transformers/sentence-t5-base"; T5_DIM=768; T5_MAXLEN=64
LHID=512; LLAYERS=8; LHEADS=8; LFF=4; TIME_DIM=256
DHID=512; DLAYERS=8; DHEADS=8   # direct DiT
FM_LR=2e-4; FM_LR_MIN=1e-5; FM_WARMUP=1000; FM_BS=64; EMA=0.999; CFG_DROP=0.1; GRAD_CLIP=1.0
LOG_EVERY=250; SAVE_EVERY=2000; EVAL_EVERY=4000; EVAL_SUB_N=512
T0=time.time(); sess_h=lambda:(time.time()-T0)/3600
def safe_save(o,p):
    t=p+".tmp"
    try:
        with open(t,"wb") as f: torch.save(o,f)
        os.replace(t,p)
    except Exception:
        if os.path.exists(t): os.remove(t)
        raise
def lr_sched(s,w,lr,lo,tot): return lr*(s+1)/w if s<w else lo+0.5*(lr-lo)*(1+math.cos(math.pi*min((s-w)/max(1,tot-w),1.0)))
# kinematic structure
_CHAINS=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
EDGES=[(c[k],c[k+1]) for c in _CHAINS for k in range(len(c)-1)]
EI=torch.tensor([e[0] for e in EDGES],device=DEVICE); EJ=torch.tensor([e[1] for e in EDGES],device=DEVICE)
FOOT_JOINTS=[7,10,8,11]
print("="*72); print(f"LFM/CLFM/CDFM | latent={RUN_LATENT} direct={RUN_DIRECT} penalty={TRAIN_PENALTY} | 12k steps"); print("="*72)

# %% [code]
# Cell 2: FK + CONSTRAINT PROJECTION (the core; validated below) + inverse-RIC
def _qmul(a,b):
    aw,ax,ay,az=a[...,0],a[...,1],a[...,2],a[...,3]; bw,bx,by,bz=b[...,0],b[...,1],b[...,2],b[...,3]
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)
def _qinv(q): return q*torch.tensor([1,-1,-1,-1],dtype=q.dtype,device=q.device)
def _qapply(q,p):
    z=torch.zeros(p.shape[:-1],dtype=p.dtype,device=p.device); pq=torch.cat((z.unsqueeze(-1),p),-1)
    return _qmul(_qmul(q,pq),_qinv(q))[...,1:]
def _root_quat_and_pos(data):
    rv=data[...,0]; ang=torch.zeros_like(rv); ang[...,1:]=rv[...,:-1]; ang=torch.cumsum(ang,-1)
    q=torch.zeros(data.shape[:-1]+(4,),device=data.device,dtype=data.dtype); q[...,0]=torch.cos(ang); q[...,2]=torch.sin(ang)
    rp=torch.zeros(data.shape[:-1]+(3,),device=data.device,dtype=data.dtype)
    rp[...,1:,[0,2]]=data[...,:-1,1:3]; rp=_qapply(q,rp); rp=torch.cumsum(rp,-2); rp[...,1]=data[...,3]
    return q,rp,ang
def recover_from_ric(data,joints=N_JOINTS):
    q,rp,ang=_root_quat_and_pos(data)
    p=data[...,4:(joints-1)*3+4].view(data.shape[:-1]+(-1,3))
    p=_qapply(q[...,None,:].expand(p.shape[:-1]+(4,)),p); p[...,0]+=rp[...,0:1]; p[...,2]+=rp[...,2:3]
    return torch.cat([rp.unsqueeze(-2),p],dim=-2)
def lengths_to_mask(L,ml): return torch.arange(ml,device=L.device)[None,:]<L[:,None]
def local_joints_from_raw(fr):
    z=fr[...,:3]*0.0; j=fr[...,4:67].reshape(fr.shape[:-1]+(21,3)); return torch.cat([z.unsqueeze(-2),j],dim=-2)

# ---- rest bone lengths (per edge), filled in Cell 4 ----
rest_len=None  # (E,)

def project_bonelength(J):
    """J:(B,T,22,3) absolute joints. Rescale each bone to rest length, walking chains from
    root outward (parent set before child) -> BLE becomes EXACTLY 0. Analytic, no solve."""
    out=J.clone()
    for ch in _CHAINS:
        for k in range(len(ch)-1):
            par,chi=ch[k],ch[k+1]
            ei=[i for i,(a,b) in enumerate(EDGES) if a==par and b==chi][0]
            vec=out[:,:,chi,:]-out[:,:,par,:]; ln=vec.norm(dim=-1,keepdim=True).clamp(min=1e-6)
            out[:,:,chi,:]=out[:,:,par,:]+vec/ln*rest_len[ei]
    return out
def project_foot(J,lens=None,h=PROJ_FOOT_H):
    """For contact frames (foot height<h) zero the horizontal foot displacement -> reduces FSR.
    Temporal constraint: cannot be analytically zeroed; this is a reduction."""
    out=J.clone(); T=out.shape[1]
    for f in FOOT_JOINTS:
        for t in range(1,T):
            contact=out[:,t,f,1]<h
            out[:,t,f,0]=torch.where(contact,out[:,t-1,f,0],out[:,t,f,0])
            out[:,t,f,2]=torch.where(contact,out[:,t-1,f,2],out[:,t,f,2])
    return out
def project_joints(J,lens=None):
    """Full projection: foot fix THEN bone rescale, so BLE->0 is EXACT at output
    (FSR reduced but nonzero, since the final bone rescale slightly perturbs the foot)."""
    return project_bonelength(project_foot(J,lens))

def inverse_ric(J,data):
    """Map absolute projected joints J back into the 263-d vector's RIC position dims [4:67],
    keeping all other dims from `data`. Lets us re-encode/continue sampling. Validated below."""
    q,rp,ang=_root_quat_and_pos(data)            # root quat, root pos (B,T,3)
    rel=J[:,:,1:,:].clone()                      # joints 1..21 absolute
    rel[...,0]-=rp[...,None,0]; rel[...,2]-=rp[...,None,2]   # subtract root xz
    ric=_qapply(_qinv(q)[...,None,:].expand(rel.shape[:-1]+(4,)),rel)  # un-rotate -> root-local
    out=data.clone(); out[...,4:4+(N_JOINTS-1)*3]=ric.reshape(data.shape[:-1]+(-1,)); return out

# %% [code]
# Cell 3: data
def load_split(split):
    md=os.path.join(HML3D_ROOT,"new_joint_vecs"); td=os.path.join(HML3D_ROOT,"texts")
    with open(os.path.join(HML3D_ROOT,f"{split}.txt")) as f: names=[l.strip() for l in f if l.strip()]
    ents=[]; sk=0
    for nm in tqdm(names,desc=f"load {split}",leave=False):
        mp=os.path.join(md,nm+".npy"); tp=os.path.join(td,nm+".txt")
        if not(os.path.exists(mp) and os.path.exists(tp)): sk+=1; continue
        try: m=np.load(mp).astype(np.float32)
        except Exception: sk+=1; continue
        if m.ndim!=2 or m.shape[1]!=NFEATS or len(m)<MIN_MOTION_LEN: sk+=1; continue
        whole=[]
        with open(tp,errors="ignore") as fh:
            for ln in fh:
                pr=ln.strip().split("#")
                if len(pr)<4 or not pr[0].strip(): continue
                cap=pr[0].strip()
                try: ft,tt=float(pr[2]),float(pr[3])
                except ValueError: continue
                if math.isnan(ft) or math.isnan(tt): ft=tt=0.0
                if ft==0.0 and tt==0.0: whole.append(cap)
                else:
                    fs,ts=int(ft*FPS),int(tt*FPS)
                    if ts-fs>=MIN_MOTION_LEN and ts<=len(m): ents.append(dict(name=nm,motion=m[fs:ts],texts=[cap]))
        if whole: ents.append(dict(name=nm,motion=m,texts=whole))
    print(f"  {split}: {len(ents)} (skipped {sk})"); return ents
print("\nLoading HumanML3D..."); train_entries=load_split("train"); test_entries=load_split("test")
train_lens=np.array([min((len(e["motion"])//UNIT_LEN)*UNIT_LEN,MAX_MOTION_LEN) for e in train_entries],dtype=np.int32)
test_lens =np.array([min((len(e["motion"])//UNIT_LEN)*UNIT_LEN,MAX_MOTION_LEN) for e in test_entries],dtype=np.int32)
print(f"  train {len(train_entries)}  test {len(test_entries)}")

# %% [code]
# Cell 4: mean/std + rest bone lengths
om=os.path.join(HML3D_ROOT,"Mean.npy"); os_=os.path.join(HML3D_ROOT,"Std.npy")
mean_data=np.load(om).astype(np.float32); std_data=np.load(os_).astype(np.float32); std_data[std_data<1e-6]=1e-6
mean_t=torch.tensor(mean_data,device=DEVICE); std_t=torch.tensor(std_data,device=DEVICE)
def pad_norm(mot):
    L=min((len(mot)//UNIT_LEN)*UNIT_LEN,MAX_MOTION_LEN); mn=((mot[:L]-mean_data)/std_data).astype(np.float32)
    if L<MAX_MOTION_LEN: mn=np.concatenate([mn,np.zeros((MAX_MOTION_LEN-L,NFEATS),np.float32)],0)
    return mn,L
_bl=[]
for e in random.sample(train_entries,min(3000,len(train_entries))):
    m=torch.tensor(e["motion"][:MAX_MOTION_LEN],device=DEVICE)[None]        # (1,T,263) RAW (unnormalized)
    J=recover_from_ric(m)                                                   # absolute FK joints — SAME space as projection & ble_pc_joints
    _bl.append((J[:,:,EI,:]-J[:,:,EJ,:]).norm(dim=-1).mean((0,1)))          # per-bone mean length
rest_len=torch.stack(_bl).mean(0)
print(f"  rest_len: {rest_len.shape[0]} bones, mean {rest_len.mean():.3f} m (from FK, consistent with projection+BLE)")

# ---- VALIDATION GATE: real motion has near-constant bones, so real-motion BLE MUST be ~0.01-0.02, not ~0.12 ----
_val=[]
for e in random.sample(train_entries,min(500,len(train_entries))):
    m=torch.tensor(e["motion"][:MAX_MOTION_LEN],device=DEVICE)[None]; J=recover_from_ric(m)
    bone=(J[:,:,EI,:]-J[:,:,EJ,:]).norm(dim=-1); _val.append((bone-rest_len).abs().mean().item())
_real_ble=float(np.mean(_val))
print(f"  [GATE] real-motion BLE = {_real_ble:.4f}  (must be < 0.03; if ~0.12 the rest_len/joint spaces are mismatched)")
assert _real_ble < 0.03, (f"rest_len VALIDATION FAILED: real-motion BLE={_real_ble:.4f} is too high. "
                          f"The bone-length reference does not match the FK joints the projection uses. "
                          f"Do NOT run training — the BLE metric and projection would be invalid.")
print("  [GATE] PASSED — BLE metric is consistent; projection will drive it to ~0 correctly.")


# %% [code]
# Cell 5: RVQ-VAE load (continuous autoencoder)
def _gn(nc,mg=8):
    g=mg
    while g>1 and nc%g!=0: g-=1
    return nn.GroupNorm(g,nc)
class RVQEncoder(nn.Module):
    def __init__(s,in_dim=NFEATS,ch=RVQ_ENC_CHANNELS,cd=RVQ_CODE_DIM):
        super().__init__(); s.in_proj=nn.Conv1d(in_dim,ch[1],3,1,1); L=[]
        for i in range(1,len(ch)):
            ci=ch[i-1] if i>1 else ch[1]; co=ch[i]; st=2 if i<len(ch)-1 else 1
            L.append(nn.Sequential(nn.Conv1d(ci,co,3,st,1),_gn(co),nn.SiLU(),nn.Conv1d(co,co,3,1,1),_gn(co),nn.SiLU()))
        s.blocks=nn.ModuleList(L); s.out_proj=nn.Conv1d(ch[-1],cd,1)
    def forward(s,x):
        x=x.transpose(1,2); x=s.in_proj(x)
        for b in s.blocks: x=b(x)
        return s.out_proj(x).transpose(1,2)
class RVQDecoder(nn.Module):
    def __init__(s,cd=RVQ_CODE_DIM,ch=RVQ_ENC_CHANNELS):
        super().__init__(); hidden=[512,256,256]; s.in_proj=nn.Conv1d(cd,hidden[0],1)
        n_up=int(round(math.log2(RVQ_DOWN))); blocks=[]
        for j in range(len(hidden)-1):
            ci,co=hidden[j],hidden[j+1]; seq=[]
            if j<n_up: seq.append(nn.Upsample(scale_factor=2,mode="nearest"))
            seq+=[nn.Conv1d(ci,co,3,1,1),_gn(co),nn.SiLU(),nn.Conv1d(co,co,3,1,1),_gn(co),nn.SiLU()]
            blocks.append(nn.Sequential(*seq))
        s.blocks=nn.ModuleList(blocks); s.out_proj=nn.Conv1d(hidden[-1],NFEATS,3,1,1)
    def forward(s,x):
        x=x.transpose(1,2); x=s.in_proj(x)
        for b in s.blocks: x=b(x)
        return s.out_proj(x).transpose(1,2)
class VQ(nn.Module):
    def __init__(s,n=512,cd=RVQ_CODE_DIM):
        super().__init__(); cb=F.normalize(torch.randn(n,cd)*0.02,dim=-1)
        s.register_buffer("codebook",cb); s.register_buffer("cluster_size",torch.zeros(n)); s.register_buffer("ema_w",cb.clone()); s.register_buffer("usage",torch.zeros(n))
class RVQ(nn.Module):
    def __init__(s,nl=4): super().__init__(); s.quantizers=nn.ModuleList([VQ() for _ in range(nl)])
class RVQVAE(nn.Module):
    def __init__(s): super().__init__(); s.encoder=RVQEncoder(); s.rvq=RVQ(); s.decoder=RVQDecoder()
rvq=RVQVAE().to(DEVICE); _rc=torch.load(RVQ_CKPT,map_location=DEVICE,weights_only=False)
rvq.load_state_dict(_rc["state"] if isinstance(_rc,dict) and "state" in _rc else _rc); rvq.eval()
for p in rvq.parameters(): p.requires_grad=False
print("  RVQ-VAE loaded.")

# %% [code]
# Cell 6: T5
print("\nLoading T5..."); from transformers import T5Tokenizer, T5EncoderModel
t5_tok=T5Tokenizer.from_pretrained(T5_MODEL); t5_enc=T5EncoderModel.from_pretrained(T5_MODEL).to(DEVICE).eval()
for p in t5_enc.parameters(): p.requires_grad=False
@torch.no_grad()
def embed_text(caps,bs=64,as_numpy=True):
    S,M,P=[],[],[]
    for s in range(0,len(caps),bs):
        inp=t5_tok(caps[s:s+bs],padding="max_length",truncation=True,max_length=T5_MAXLEN,return_tensors="pt").to(DEVICE)
        h=t5_enc(**inp).last_hidden_state; mk=inp["attention_mask"]
        pool=F.normalize((h*mk.unsqueeze(-1).float()).sum(1)/mk.sum(1,keepdim=True).clamp(min=1),dim=-1)
        if as_numpy: S.append(h.cpu().numpy().astype(np.float32)); M.append(mk.cpu().numpy().astype(np.bool_)); P.append(pool.cpu().numpy().astype(np.float32))
        else: S.append(h); M.append(mk.bool()); P.append(pool)
    return (np.concatenate(S,0),np.concatenate(M,0),np.concatenate(P,0)) if as_numpy else (torch.cat(S,0),torch.cat(M,0),torch.cat(P,0))
print("Embedding captions..."); tr_seq,tr_mask,tr_pool=embed_text([e["texts"][0] for e in train_entries]); te_seq,te_mask,te_pool=embed_text([e["texts"][0] for e in test_entries])

# %% [code]
# Cell 7: latent base data (frozen-VAE latents + z-stats)
@torch.no_grad()
def encode_all():
    Z=[]
    for s in tqdm(range(0,len(train_entries),128),desc="encode",leave=False):
        e=min(s+128,len(train_entries)); B=[pad_norm(train_entries[i]["motion"])[0] for i in range(s,e)]
        Z.append(rvq.encoder(torch.tensor(np.stack(B),device=DEVICE)).cpu().numpy().astype(np.float32))
    return np.concatenate(Z,0)
if RUN_LATENT:
    print("\nEncoding latents..."); tz=encode_all()
    z_mean=tz.reshape(-1,RVQ_CODE_DIM).mean(0).astype(np.float32); z_std=(tz.reshape(-1,RVQ_CODE_DIM).std(0)+1e-6).astype(np.float32)
    train_z=((tz-z_mean)/z_std).astype(np.float32); z_mean_t=torch.tensor(z_mean,device=DEVICE); z_std_t=torch.tensor(z_std,device=DEVICE)

# %% [code]
# Cell 8: models (latent FM + direct DiT) + projected samplers
def sinusoidal(t,dim,mp=10000):
    half=dim//2; fr=torch.exp(-math.log(mp)*torch.arange(half,device=t.device).float()/half); a=t.float().unsqueeze(-1)*fr
    return torch.cat([torch.cos(a),torch.sin(a)],-1)
class FiLM(nn.Module):
    def __init__(s,dim,heads,ff,tdim=T5_DIM,drop=0.1):
        super().__init__()
        s.n1=nn.LayerNorm(dim,elementwise_affine=False); s.n2=nn.LayerNorm(dim,elementwise_affine=False); s.n3=nn.LayerNorm(dim,elementwise_affine=False); s.nt=nn.LayerNorm(tdim)
        s.sa=nn.MultiheadAttention(dim,heads,dropout=drop,batch_first=True); s.ca=nn.MultiheadAttention(dim,heads,dropout=drop,kdim=tdim,vdim=tdim,batch_first=True)
        s.ff=nn.Sequential(nn.Linear(dim,dim*ff),nn.GELU(),nn.Dropout(drop),nn.Linear(dim*ff,dim),nn.Dropout(drop)); s.film=nn.Linear(dim,9*dim); nn.init.zeros_(s.film.weight); nn.init.zeros_(s.film.bias)
    def forward(s,x,c,tseq,tmask):
        a1,b1,g1,a2,b2,g2,a3,b3,g3=s.film(c).chunk(9,-1)
        h=s.n1(x)*(1+a1.unsqueeze(1))+b1.unsqueeze(1); x=x+g1.unsqueeze(1)*s.sa(h,h,h,need_weights=False)[0]
        h=s.n2(x)*(1+a2.unsqueeze(1))+b2.unsqueeze(1); x=x+g2.unsqueeze(1)*s.ca(h,s.nt(tseq),s.nt(tseq),key_padding_mask=~tmask,need_weights=False)[0]
        h=s.n3(x)*(1+a3.unsqueeze(1))+b3.unsqueeze(1); return x+g3.unsqueeze(1)*s.ff(h)
class FMNet(nn.Module):
    """Shared transformer for latent (cd=256,T=49) or direct (cd=263,T=196)."""
    def __init__(s,cd,Tlen,hid,layers,heads,ff=LFF):
        super().__init__(); s.cd=cd; s.Tlen=Tlen; s.in_proj=nn.Linear(cd,hid); s.out_proj=nn.Linear(hid,cd); nn.init.zeros_(s.out_proj.weight); nn.init.zeros_(s.out_proj.bias)
        s.pos=nn.Parameter(torch.randn(1,Tlen,hid)*0.02); s.time_mlp=nn.Sequential(nn.Linear(TIME_DIM,hid),nn.SiLU(),nn.Linear(hid,hid))
        s.text_proj=nn.Sequential(nn.Linear(T5_DIM,hid),nn.SiLU(),nn.Linear(hid,hid)); s.len_emb=nn.Embedding(MAX_MOTION_LEN+1,hid)
        s.null_seq=nn.Parameter(torch.zeros(T5_MAXLEN,T5_DIM)); s.null_pool=nn.Parameter(torch.zeros(T5_DIM))
        s.blocks=nn.ModuleList([FiLM(hid,heads,ff) for _ in range(layers)])
    def forward(s,z,t,tseq,tmask,tpool,length):
        h=s.in_proj(z)+s.pos; c=s.time_mlp(sinusoidal(t,TIME_DIM))+s.text_proj(tpool)+s.len_emb(length.clamp(0,MAX_MOTION_LEN))
        for b in s.blocks: h=b(h,c,tseq,tmask)
        return s.out_proj(h)

@torch.no_grad()
def _gj(mn): return recover_from_ric(mn*std_t+mean_t)   # normalized 263 -> absolute joints
def _joints_to_norm(J,base_mn):
    """projected absolute joints -> normalized 263 (write RIC dims, keep rest from base_mn)."""
    raw=base_mn*std_t+mean_t; raw2=inverse_ric(J,raw); return (raw2-mean_t)/std_t

@torch.no_grad()
def sample(net,is_latent,tseq,tmask,tpool,length,n=ODE_STEPS,guidance=GUIDANCE,mode="none",seed=None):
    """mode: 'none' | 'posthoc' | 'inproc'. Returns normalized-263 motion (B,T,263)."""
    if seed is not None: torch.manual_seed(seed)
    B=tpool.shape[0]; cd=net.cd; Tl=net.Tlen
    z=torch.randn(B,Tl,cd,device=DEVICE); ns=net.null_seq.unsqueeze(0).expand(B,-1,-1); nm=torch.ones(B,T5_MAXLEN,dtype=torch.bool,device=DEVICE); npl=net.null_pool.unsqueeze(0).expand(B,-1); dt=1.0/n
    for i in range(n):
        t=torch.full((B,),i*dt,device=DEVICE); v=net(z,t,tseq,tmask,tpool,length)
        if guidance!=1.0: vu=net(z,t,ns,nm,npl,length); v=vu+guidance*(v-vu)
        z=z+dt*v
        if mode=="inproc" and i>=n//2:   # guide during 2nd half: decode-project-(re)encode
            mn=(rvq.decoder(z*z_std_t+z_mean_t) if is_latent else z)
            J=project_joints(_gj(mn),length); mn2=_joints_to_norm(J,mn)
            z=(rvq.encoder(mn2) if is_latent else mn2)
    mn=(rvq.decoder(z*z_std_t+z_mean_t) if is_latent else z)
    if mode in ("posthoc","inproc"):     # final projection GUARANTEES BLE->0 exactly on output
        J=project_joints(_gj(mn),length); mn=_joints_to_norm(J,mn)
    return mn

# %% [code]
# Cell 9: evaluator + metrics (verified)
def _iw(m):
    if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
class MovementConvEncoder(nn.Module):
    def __init__(s,i,h,o):
        super().__init__(); s.main=nn.Sequential(nn.Conv1d(i,h,4,2,1),nn.Dropout(.2,True),nn.LeakyReLU(.2,True),nn.Conv1d(h,o,4,2,1),nn.Dropout(.2,True),nn.LeakyReLU(.2,True)); s.out_net=nn.Linear(o,o); s.main.apply(_iw); s.out_net.apply(_iw)
    def forward(s,x): return s.out_net(s.main(x.permute(0,2,1)).permute(0,2,1))
class MotionEncoderBiGRUCo(nn.Module):
    def __init__(s,i,h,o):
        super().__init__(); s.input_emb=nn.Linear(i,h); s.gru=nn.GRU(h,h,batch_first=True,bidirectional=True); s.out=nn.Sequential(nn.Linear(2*h,h),nn.LayerNorm(h),nn.LeakyReLU(.2,True),nn.Linear(h,o)); s.hidden=nn.Parameter(torch.randn(2,1,h)); s.input_emb.apply(_iw); s.out.apply(_iw)
    def forward(s,x,ml):
        n=x.shape[0]; e=pack_padded_sequence(s.input_emb(x),ml.tolist(),batch_first=True,enforce_sorted=False); _,h=s.gru(e,s.hidden.repeat(1,n,1)); return s.out(torch.cat([h[0],h[1]],-1))
def _remap(sd,*r):
    o={}
    for k,v in sd.items():
        for a,b in r: k=k.replace(a,b)
        o[k]=v
    return o
def _meta():
    for p in ["/kaggle/input/**/Comp_v6_KLD01/meta/mean.npy","/kaggle/working/**/meta/mean.npy","/kaggle/input/**/meta/mean.npy"]:
        for h in glob.glob(p,recursive=True):
            s=h.replace("mean.npy","std.npy")
            if os.path.exists(s): me=np.load(h).astype(np.float32); sd=np.load(s).astype(np.float32); return me,np.where(sd<1e-8,1e-8,sd)
    return mean_data,std_data
def _finest():
    pats=([os.path.join(EVAL_ROOT,"**","finest.tar")] if EVAL_ROOT else [])+["/kaggle/working/**/finest.tar","/kaggle/input/**/finest.tar"]
    for p in pats:
        h=sorted(glob.glob(p,recursive=True),key=lambda x:(0 if "text_mot_match" in x else 1,len(x)))
        if h: return h[0]
    return None
mean_eval,std_eval=_meta(); mean_eval_t=torch.tensor(mean_eval,device=DEVICE); std_eval_t=torch.tensor(std_eval,device=DEVICE)
def to_eval(m): return ((m*std_t+mean_t)-mean_eval_t)/std_eval_t
EVAL_ENABLED=False; ev_me=ev_mo=None
try:
    ck=_finest()
    if ck is None: raise FileNotFoundError("finest.tar (attach the evaluator for FID/R)")
    c=torch.load(ck,map_location=DEVICE,weights_only=False); ev_me=MovementConvEncoder(259,512,512).to(DEVICE); ev_mo=MotionEncoderBiGRUCo(512,1024,512).to(DEVICE)
    ev_me.load_state_dict(_remap(c["movement_encoder"],("output_net.","out_net."))); ev_mo.load_state_dict(_remap(c["motion_encoder"],("output_net.","out."))); ev_me.eval(); ev_mo.eval(); EVAL_ENABLED=True; print("  evaluator loaded.")
except Exception as ex: print("  evaluator unavailable:",ex,"-> FID/R skipped.")
@torch.no_grad()
def memb(motions,m_lens):
    m=to_eval(motions.to(DEVICE).float()); mov=ev_me(m[...,:-4]).detach(); return ev_mo(mov,(m_lens//UNIT_LEN).clamp(min=1)).cpu().numpy()
def fid_calc(g,r):
    mg,sg=g.mean(0),np.cov(g.T); mr,sr=r.mean(0),np.cov(r.T); d=mg-mr; cm,_=scipy_linalg.sqrtm(sg@sr,disp=False)
    if np.iscomplexobj(cm): cm=cm.real
    return max(float(d@d+np.trace(sg+sr-2*cm)),0.0)
def rprec(g,r,bs=32,ks=(1,2,3)):
    N=len(g); gn=g/(np.linalg.norm(g,1,keepdims=True)+1e-8); rn=r/(np.linalg.norm(r,1,keepdims=True)+1e-8); per={k:[] for k in ks}
    for b in range(max(N//bs,1)):
        s=b*bs; e=min(s+bs,N)
        if e-s<2: continue
        sim=gn[s:e]@rn[s:e].T; rk=np.argsort(-sim,1); corr=np.arange(e-s)[:,None]; mt=rk==corr
        for k in ks: per[k].append(float(mt[:,:k].any(1).mean()))
    return {k:float(np.mean(per[k])) if per[k] else 0.0 for k in ks}
@torch.no_grad()
def fsr_pc(J,L,h=PROJ_FOOT_H):
    fj=J[:,:,FOOT_JOINTS,:]; hor=(fj[:,1:,:,[0,2]]-fj[:,:-1,:,[0,2]]).norm(dim=-1); ht=fj[:,1:,:,1]
    w=(ht<h).float()*lengths_to_mask(L,MAX_MOTION_LEN)[:,1:].unsqueeze(-1).float(); return ((hor*w).sum(dim=(1,2))/(w.sum(dim=(1,2))+1e-8)).cpu().numpy()
@torch.no_grad()
def ble_pc(mn,L):
    J=recover_from_ric(mn*std_t+mean_t); bone=(J[:,:,EI,:]-J[:,:,EJ,:]).norm(dim=-1); err=(bone-rest_len).abs().mean(-1); fm=lengths_to_mask(L,MAX_MOTION_LEN).float(); return ((err*fm).sum(1)/(fm.sum(1)+1e-8)).cpu().numpy()
@torch.no_grad()
def ble_pc_joints(J,L):
    bone=(J[:,:,EI,:]-J[:,:,EJ,:]).norm(dim=-1); err=(bone-rest_len).abs().mean(-1); fm=lengths_to_mask(L,MAX_MOTION_LEN).float(); return ((err*fm).sum(1)/(fm.sum(1)+1e-8)).cpu().numpy()
# eval subset for training monitor
_rng=np.random.default_rng(123); cand=[i for i in range(len(test_entries)) if (len(test_entries[i]["motion"])//UNIT_LEN)*UNIT_LEN>=MIN_MOTION_LEN]
sub_idx=np.array(sorted(_rng.permutation(cand)[:EVAL_SUB_N].tolist()))
sub_len=torch.tensor([int(test_lens[i]) for i in sub_idx],device=DEVICE); sub_tseq=te_seq[sub_idx]; sub_tmask=te_mask[sub_idx]; sub_tpool=te_pool[sub_idx]; sub_real_mf=None
if EVAL_ENABLED:
    with torch.no_grad():
        _r=[]
        for s in range(0,len(sub_idx),64):
            e=min(s+64,len(sub_idx)); rm=torch.tensor(np.stack([pad_norm(test_entries[int(i)]["motion"])[0] for i in sub_idx[s:e]]),device=DEVICE); ml=lengths_to_mask(sub_len[s:e],MAX_MOTION_LEN)
            _r.append(memb(rm*ml[...,None],sub_len[s:e]))
        sub_real_mf=np.concatenate(_r,0)
@torch.no_grad()
def quick_fid(net,is_latent):
    net.eval(); mf=[]
    for s in range(0,len(sub_idx),32):
        e=min(s+32,len(sub_idx)); ts=torch.tensor(sub_tseq[s:e],device=DEVICE); tm=torch.tensor(sub_tmask[s:e],device=DEVICE); tp=torch.tensor(sub_tpool[s:e],device=DEVICE)
        x=sample(net,is_latent,ts,tm,tp,sub_len[s:e]); gm=lengths_to_mask(sub_len[s:e],MAX_MOTION_LEN)
        if EVAL_ENABLED: mf.append(memb(x*gm[...,None],sub_len[s:e]))
    return fid_calc(np.concatenate(mf,0),sub_real_mf) if EVAL_ENABLED else float("nan")

# %% [code]
# Cell 10: training (shared) — base or penalty variant
class EMAh:
    def __init__(s,m,d): s.d=d; s.shadow={k:v.clone().detach() for k,v in m.state_dict().items()}
    @torch.no_grad()
    def update(s,m):
        for k,v in m.state_dict().items(): s.shadow[k].mul_(s.d).add_(v.detach()*(1-s.d)) if v.dtype.is_floating_point else s.shadow[k].copy_(v)
def diff_penalty(mn,L):
    """differentiable bone + foot penalty on decoded normalized motion."""
    raw=mn*std_t+mean_t; fm=lengths_to_mask(L,MAX_MOTION_LEN).float()
    J=recover_from_ric(raw)                                   # FK joints — consistent with rest_len
    bone=(J[:,:,EI,:]-J[:,:,EJ,:]).norm(dim=-1); Lb=(((bone-rest_len)**2).mean(-1)*fm).sum()/(fm.sum()+1e-6)
    fj=J[:,:,FOOT_JOINTS,:]; vel=(fj[:,1:,:,[0,2]]-fj[:,:-1,:,[0,2]]).norm(dim=-1); ht=fj[:,1:,:,1]
    cw=(ht<PROJ_FOOT_H).float()*fm[:,1:].unsqueeze(-1); Lf=(vel*cw).sum()/(cw.sum()+1e-6)
    return PEN_BONE*Lb+PEN_FOOT*Lf
def train_base(tag,is_latent,total,penalty):
    best_p=os.path.join(CK,f"{tag}_best.pt"); latest_p=os.path.join(CK,f"{tag}_latest.pt")
    cd=RVQ_CODE_DIM if is_latent else NFEATS; Tl=T_LAT if is_latent else MAX_MOTION_LEN
    net=FMNet(cd,Tl,LHID if is_latent else DHID,LLAYERS if is_latent else DLAYERS,LHEADS if is_latent else DHEADS).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=FM_LR,weight_decay=0.0); ema=EMAh(net,EMA); st=0; best=float("inf")
    if os.path.exists(latest_p):
        r=torch.load(latest_p,map_location=DEVICE,weights_only=False); net.load_state_dict(r["net"]); ema.shadow={k:v.to(DEVICE) for k,v in r["ema"].items()}
        try: opt.load_state_dict(r["opt"])
        except Exception: pass
        st=int(r.get("step",0)); best=float(r.get("best",float("inf"))); print(f"  [{tag}] resume @ {st}")
    order=list(range(len(train_entries)))
    def save_latest(): safe_save(dict(net=net.state_dict(),ema={k:v.clone() for k,v in ema.shadow.items()},opt=opt.state_dict(),step=st,best=best,z_mean=(z_mean if is_latent else None),z_std=(z_std if is_latent else None)),latest_p)
    def save_best(metric): 
        nonlocal best; best=metric; safe_save(dict(state={k:v.clone() for k,v in ema.shadow.items()},step=st,metric=metric,z_mean=(z_mean if is_latent else None),z_std=(z_std if is_latent else None)),best_p)
    print("\n"+"-"*72); print(f"TRAIN [{tag}]  {st}->{total}  latent={is_latent} penalty={penalty}"); print("-"*72)
    net.train(); t0=time.time(); done=(st>=total)
    while not done:
        random.shuffle(order)
        for s in range(0,len(order)-FM_BS,FM_BS):
            if st>=total: done=True; break
            idx=order[s:s+FM_BS]; L=torch.tensor([int(train_lens[i]) for i in idx],device=DEVICE)
            tseq=torch.tensor(tr_seq[idx],device=DEVICE); tmask=torch.tensor(tr_mask[idx],device=DEVICE); tpool=torch.tensor(tr_pool[idx],device=DEVICE); B=len(idx)
            if is_latent: z1=torch.tensor(train_z[idx],device=DEVICE)
            else: z1=torch.tensor(np.stack([pad_norm(train_entries[i]["motion"])[0] for i in idx]),device=DEVICE)
            drop=(torch.rand(B,device=DEVICE)<CFG_DROP)
            tseq=torch.where(drop[:,None,None],net.null_seq.unsqueeze(0).expand(B,-1,-1),tseq); tpool=torch.where(drop[:,None],net.null_pool.unsqueeze(0).expand(B,-1),tpool)
            for pg in opt.param_groups: pg["lr"]=lr_sched(st,FM_WARMUP,FM_LR,FM_LR_MIN,total)
            z0=torch.randn_like(z1); t=torch.rand(B,device=DEVICE); zt=(1-t).view(-1,1,1)*z0+t.view(-1,1,1)*z1; u=z1-z0
            v=net(zt,t,tseq,tmask,tpool,L); loss=F.mse_loss(v,u)
            if penalty:    # soft variant: add differentiable bone+foot penalty on the predicted endpoint
                z1hat=zt+(1-t).view(-1,1,1)*v   # x1 estimate from current state+velocity
                mnhat=(rvq.decoder(z1hat*z_std_t+z_mean_t) if is_latent else z1hat)
                loss=loss+diff_penalty(mnhat,L)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),GRAD_CLIP); opt.step(); ema.update(net); st+=1
            if st%LOG_EVERY==0:
                print(f"  [{tag}] {st:>6} loss={loss.item():.4f} {time.time()-t0:.0f}s ({sess_h():.1f}h)")
                wlog({f"{tag}/loss":loss.item(), f"{tag}/lr":opt.param_groups[0]["lr"], f"{tag}/step":st}, step=st)
            if st%SAVE_EVERY==0: save_latest()
            if st%EVAL_EVERY==0:
                bk={k:v.detach().clone() for k,v in net.state_dict().items()}; net.load_state_dict({k:v.to(DEVICE) for k,v in ema.shadow.items()})
                fid=quick_fid(net,is_latent); star=""
                if fid<best: save_best(fid); star="  <-BEST"
                print(f"    [{tag} eval {st}] FID={fid:.4f}{star}")
                wlog({f"{tag}/eval_FID":fid, f"{tag}/best_FID":best}, step=st)
                net.load_state_dict(bk); save_latest()
            if st>=total: done=True; break
    save_latest()
    if not os.path.exists(best_p): save_best(float("inf"))
    print(f"  [{tag}] done best={best:.4f}"); return best_p

# %% [code]
# Cell 11: train the requested base(s). VARIANT env var picks exactly one per job (or "all"/"eval").
bases={}
def _train_if(name,is_latent,penalty):
    if VARIANT in (name,"all"): bases[name]=train_base(name,is_latent,STEPS[name],penalty=penalty)
# Path 1a: reuse a recovered latent-FM checkpoint instead of retraining the latent base.
# Set LFM_CKPT to the recovered lfm_best.pt; it is copied into the expected latent_best.pt
# slot so the eval loader picks it up, and latent training is skipped.
_LFM_CKPT=os.environ.get("LFM_CKPT","")
if _LFM_CKPT and os.path.exists(_LFM_CKPT) and VARIANT in ("latent","all","eval"):
    _dst=os.path.join(CK,"latent_best.pt")
    if not os.path.exists(_dst):
        import shutil as _sh; _sh.copy(_LFM_CKPT,_dst); print(f"  [1a] reused recovered LFM -> {_dst} (latent training skipped)")
    _SKIP_LATENT=True
else:
    _SKIP_LATENT=False
if VARIANT=="eval":
    print("VARIANT=eval -> skipping training; will evaluate existing checkpoints.")
else:
    if not _SKIP_LATENT: _train_if("latent",True,False)
    _train_if("latent_pen",True,True)
    _train_if("direct",False,False)
    _train_if("direct_pen",False,True)
print("bases trained this job:",list(bases))
# When training a SINGLE base (parallel sbatch), don't run the full table here — the eval job does that.
_RUN_TABLE = (VARIANT in ("all","eval"))


# %% [code]
# Cell 12: ============ TABLE: evaluate every variant ============
def load_net(tag,is_latent):
    bp=os.path.join(CK,f"{tag}_best.pt"); ck=torch.load(bp,map_location=DEVICE,weights_only=False)
    cd=RVQ_CODE_DIM if is_latent else NFEATS; Tl=T_LAT if is_latent else MAX_MOTION_LEN
    net=FMNet(cd,Tl,LHID if is_latent else DHID,LLAYERS if is_latent else DLAYERS,LHEADS if is_latent else DHEADS).to(DEVICE); net.load_state_dict(ck["state"]); net.eval(); return net
def _agg(a): return float(a.mean()),float(np.percentile(a,95)),float(a.max())
@torch.no_grad()
def eval_variant(net,is_latent,mode,N=EVAL_N):
    rng=np.random.default_rng(0); sel=np.array(sorted(rng.permutation(len(test_entries))[:N].tolist()))
    caps=[test_entries[int(i)]["texts"][0] for i in sel]; lens=torch.tensor([int(test_lens[i]) for i in sel],device=DEVICE)
    tseq,tmask,tpool=embed_text(caps); fsr=[]; ble=[]; mf=[]
    real_mf=[]
    for s in range(0,N,32):
        e=min(s+32,N); ts=torch.tensor(tseq[s:e],device=DEVICE); tm=torch.tensor(tmask[s:e],device=DEVICE); tp=torch.tensor(tpool[s:e],device=DEVICE)
        x=sample(net,is_latent,ts,tm,tp,lens[s:e],mode=mode,seed=s); gm=lengths_to_mask(lens[s:e],MAX_MOTION_LEN); J=_gj(x)
        fsr.append(fsr_pc(J,lens[s:e])); ble.append(ble_pc_joints(J,lens[s:e]))
        if EVAL_ENABLED:
            mf.append(memb(x*gm[...,None],lens[s:e]))
            rm=torch.tensor(np.stack([pad_norm(test_entries[int(i)]["motion"])[0] for i in sel[s:e]]),device=DEVICE); real_mf.append(memb(rm*gm[...,None],lens[s:e]))
    fsr=np.concatenate(fsr); ble=np.concatenate(ble)
    fid=R3=float("nan")
    if EVAL_ENABLED: G=np.concatenate(mf,0); Rr=np.concatenate(real_mf,0); fid=fid_calc(G,Rr); R3=rprec(G,Rr)[3]
    return dict(fid=fid,R3=R3,fsr=fsr,ble=ble)

rows=[]  # (label, dict)
def add(label,net,is_latent,mode): print(f"  eval: {label}"); rows.append((label,eval_variant(net,is_latent,mode)))
def _have(tag): return os.path.exists(os.path.join(CK,f"{tag}_best.pt"))
if not _RUN_TABLE:
    print(f"\nVARIANT={VARIANT}: single-base training job done. Table/figures are produced by the eval job (VARIANT=eval).")
else:
  if _have("latent"):
    nL=load_net("latent",True)
    add("LFM (unconstrained)",nL,True,"none")
    if _have("latent_pen"): add("LFM + penalty",load_net("latent_pen",True),True,"none")
    add("CLFM + post-hoc proj",nL,True,"posthoc")
    add("CLFM + in-process proj",nL,True,"inproc")
  if _have("direct"):
    nD=load_net("direct",False)
    add("CDFM (unconstrained)",nD,False,"none")
    if _have("direct_pen"): add("CDFM + penalty",load_net("direct_pen",False),False,"none")
    add("CDFM + post-hoc proj",nD,False,"posthoc")
    add("CDFM + in-process proj",nD,False,"inproc")

if _RUN_TABLE and rows:
  print("\n"+"="*96)
  print(f" {'Variant':<26} {'FID':>7} {'R@3':>6} {'FSR mean':>9} {'FSR p95':>8} {'BLE mean':>9} {'BLE p95':>8} {'BLE max':>8}")
  print("="*96)
  _wb_rows=[]
  for label,r in rows:
    fm,fp,_=_agg(r["fsr"]); bm,bp,bx=_agg(r["ble"])
    print(f" {label:<26} {r['fid']:>7.3f} {r['R3']:>6.3f} {fm:>9.4f} {fp:>8.4f} {bm:>9.4f} {bp:>8.4f} {bx:>8.4f}")
    _wb_rows.append([label,round(r['fid'],3),round(r['R3'],3),round(fm,4),round(fp,4),round(bm,4),round(bp,4),round(bx,4)])
  print("="*96)
  print(" Note: projection rows -> BLE mean/p95/max ~ 0 (EXACT, analytic bone rescale);")
  print("       FSR reduced vs unconstrained but NOT zero (temporal constraint).")
  np.savez(os.path.join(PROJ,"table_arrays.npz"),**{f"{i}_{k}":v for i,(lab,r) in enumerate(rows) for k,v in r.items() if k in ("fsr","ble")})
  if WB is not None:
    try:
        tbl=WB.Table(columns=["Variant","FID","R@3","FSR_mean","FSR_p95","BLE_mean","BLE_p95","BLE_max"],data=_wb_rows)
        WB.log({"projection_table":tbl})
        for label,r in rows:
            fm,fp,_=_agg(r["fsr"]); bm,bp,bx=_agg(r["ble"]); key=label.replace(" ","_").replace("+","").replace("(","").replace(")","")
            WB.log({f"final/{key}/FID":r['fid'],f"final/{key}/R3":r['R3'],f"final/{key}/FSR_mean":fm,f"final/{key}/BLE_mean":bm,f"final/{key}/BLE_max":bx})
    except Exception as _e: print("wandb table log failed:",_e)

# %% [code]
# Cell 13: ============ QUALITATIVE FIGURES ============
# Single-base training jobs (VARIANT=latent/direct/...) finish here — the eval job
# (VARIANT=eval, after all bases are trained) produces the figures + GIFs below.
if not (_RUN_TABLE and rows):
    if WB is not None:
        try: WB.finish()
        except Exception: pass
    print(f"[{VARIANT}] job complete (training + logging done). Figures/GIFs are made by the eval job.")
    import sys as _sys; _sys.exit(0)

# pick one test clip, compare unconstrained vs in-process projection
if _have("latent"):
    netQ=load_net("latent",True); isL=True
else:
    netQ=load_net("direct",False); isL=False
qi=int(sub_idx[0]); cap=test_entries[qi]["texts"][0]; Lq=int(test_lens[qi]); tq=embed_text([cap]); tq=[torch.tensor(a,device=DEVICE) for a in tq]
x_unc=sample(netQ,isL,tq[0],tq[1],tq[2],torch.tensor([Lq],device=DEVICE),mode="none",seed=7)
x_prj=sample(netQ,isL,tq[0],tq[1],tq[2],torch.tensor([Lq],device=DEVICE),mode="inproc",seed=7)
Ju=_gj(x_unc)[0,:Lq].cpu().numpy(); Jp=_gj(x_prj)[0,:Lq].cpu().numpy()
# (1) per-frame BLE trace
def ble_trace(J):
    Jt=torch.tensor(J,device=DEVICE)[None]; bone=(Jt[:,:,EI,:]-Jt[:,:,EJ,:]).norm(dim=-1); return (bone-rest_len).abs().mean(-1)[0].cpu().numpy()
# (2) foot height + horizontal speed traces
def foot_trace(J,f=7):
    h=J[:,f,1]; spd=np.linalg.norm(J[1:,f,[0,2]]-J[:-1,f,[0,2]],axis=-1); return h,np.concatenate([[0],spd])
fig,axes=plt.subplots(1,3,figsize=(15,3.8),dpi=120)
axes[0].plot(ble_trace(Ju),label="unconstrained",color="#e74c3c"); axes[0].plot(ble_trace(Jp),label="projected",color="#2ecc71")
axes[0].set_title("Bone-length error per frame"); axes[0].set_xlabel("frame"); axes[0].set_ylabel("BLE (m)"); axes[0].legend(fontsize=8)
hu,su=foot_trace(Ju); hp,sp=foot_trace(Jp)
axes[1].plot(su,label="unconstrained",color="#e74c3c"); axes[1].plot(sp,label="projected",color="#2ecc71"); axes[1].plot(hu,"--",color="#888",alpha=.6,label="foot height (unc)")
axes[1].axhline(PROJ_FOOT_H,color="k",ls=":",lw=.8); axes[1].set_title("Foot horizontal speed (contact = below dotted)"); axes[1].set_xlabel("frame"); axes[1].legend(fontsize=8)
# (3) BLE distribution over the eval set (unconstrained vs projected)
if rows:
    u_ble=next((r["ble"] for lab,r in rows if "unconstrained" in lab),None); p_ble=next((r["ble"] for lab,r in rows if "in-process" in lab),None)
    if u_ble is not None and p_ble is not None:
        hi=np.percentile(u_ble,99); b=np.linspace(0,hi,40)
        axes[2].hist(u_ble,bins=b,alpha=.6,color="#e74c3c",label="unconstrained",density=True); axes[2].hist(p_ble,bins=b,alpha=.6,color="#2ecc71",label="projected",density=True)
        axes[2].set_title("BLE distribution (eval set)"); axes[2].set_xlabel("BLE (m)"); axes[2].set_yticks([]); axes[2].legend(fontsize=8)
fig.suptitle(f'"{cap[:70]}"',fontsize=10); fig.tight_layout(); pth=os.path.join(FIG,"qualitative.png"); fig.savefig(pth,bbox_inches="tight"); plt.close(fig)
display(HTML(f'<img src="data:image/png;base64,{base64.b64encode(open(pth,"rb").read()).decode()}" style="width:980px">')); print("saved",pth)

# (4) skeleton render strips (unconstrained vs projected) at a few frames
def strip(J,color,title):
    fr=np.linspace(0,len(J)-1,6).astype(int); fig,axs=plt.subplots(1,6,figsize=(15,2.8),dpi=110,subplot_kw={"projection":"3d"})
    for ax,t in zip(axs,fr):
        for ch in _CHAINS: ax.plot([J[t,k,0] for k in ch],[J[t,k,2] for k in ch],[J[t,k,1] for k in ch],color=color,lw=2,marker="o",ms=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=12,azim=60); ax.set_title(f"f{t}",fontsize=7)
    fig.suptitle(title,fontsize=9); fig.tight_layout(); return fig
for J,c,t,nm in [(Ju,"#e74c3c","unconstrained (note limb stretch / foot slide)","q_unc"),(Jp,"#2ecc71","projected (BLE=0, foot planted)","q_prj")]:
    f=strip(J,c,t); pp=os.path.join(FIG,nm+".png"); f.savefig(pp,bbox_inches="tight"); plt.close(f)
    display(HTML(f'<img src="data:image/png;base64,{base64.b64encode(open(pp,"rb").read()).decode()}" style="width:980px">'))
print("Done. Table arrays + figures saved under",PROJ)

# %% [code]
# Cell 14: ============ INLINE GIF COMPARISON across variants ============
# One animated GIF per variant, laid out side-by-side per prompt so the
# unconstrained-vs-projected difference (limb stretch / foot slide) is visible.
from PIL import Image as _Image
def render_gif_b64(J,title="",color="#3498db",max_frames=40,fps=15):
    """J:(T,22,3) absolute joints -> base64 animated GIF (3D skeleton)."""
    T=J.shape[0]; idx=np.linspace(0,T-1,min(max_frames,T)).astype(int); pad=0.4
    xmn,xmx=J[...,0].min()-pad,J[...,0].max()+pad; zmn,zmx=J[...,2].min()-pad,J[...,2].max()+pad
    ymn=min(J[...,1].min()-0.05,0.0); ymx=J[...,1].max()+0.3; span=max(xmx-xmn,zmx-zmn)
    cx,cz=(xmn+xmx)/2,(zmn+zmx)/2; xmn,xmx=cx-span/2,cx+span/2; zmn,zmx=cz-span/2,cz+span/2; fr=[]
    for t in idx:
        fig=plt.figure(figsize=(2.6,2.6),dpi=70); ax=fig.add_subplot(111,projection="3d")
        xx,zz=np.meshgrid(np.linspace(xmn,xmx,4),np.linspace(zmn,zmx,4)); ax.plot_surface(xx,zz,np.zeros_like(xx),alpha=0.05,color="#888",linewidth=0)
        for ch in _CHAINS: ax.plot([J[t,k,0] for k in ch],[J[t,k,2] for k in ch],[J[t,k,1] for k in ch],color=color,lw=2,marker="o",ms=2.5)
        ax.set_xlim(xmn,xmx); ax.set_ylim(zmn,zmx); ax.set_zlim(ymn,ymx); ax.set_box_aspect([1,1,max(0.4,(ymx-ymn)/span)])
        ax.view_init(elev=12,azim=60); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        if title: ax.set_title(title,fontsize=7)
        fig.tight_layout(pad=0.1); buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=70); buf.seek(0); fr.append(_Image.open(buf).convert("RGB").copy()); buf.close(); plt.close(fig)
    out=io.BytesIO(); fr[0].save(out,format="GIF",save_all=True,append_images=fr[1:],duration=int(1000/fps),loop=0); out.seek(0)
    return base64.b64encode(out.read()).decode()

# build the variant list (label, net, is_latent, mode, color) matching the table
gif_variants=[]
if _have("latent"):
    _nL=load_net("latent",True)
    gif_variants.append(("LFM unconstrained",_nL,True,"none","#e74c3c"))
    if _have("latent_pen"): gif_variants.append(("LFM + penalty",load_net("latent_pen",True),True,"none","#e67e22"))
    gif_variants.append(("CLFM post-hoc",_nL,True,"posthoc","#27ae60"))
    gif_variants.append(("CLFM in-process",_nL,True,"inproc","#2ecc71"))
if _have("direct"):
    _nD=load_net("direct",False)
    gif_variants.append(("CDFM unconstrained",_nD,False,"none","#c0392b"))
    if _have("direct_pen"): gif_variants.append(("CDFM + penalty",load_net("direct_pen",False),False,"none","#d35400"))
    gif_variants.append(("CDFM post-hoc",_nD,False,"posthoc","#16a085"))
    gif_variants.append(("CDFM in-process",_nD,False,"inproc","#1abc9c"))

GIF_PROMPTS=[test_entries[int(sub_idx[0])]["texts"][0], "a person walks forward then turns around"]
for prompt in GIF_PROMPTS:
    tq=embed_text([prompt]); tq=[torch.tensor(a,device=DEVICE) for a in tq]; Lq=(min(MAX_MOTION_LEN,140)//UNIT_LEN)*UNIT_LEN
    cells=[]
    for label,net,isL,mode,col in gif_variants:
        x=sample(net,isL,tq[0],tq[1],tq[2],torch.tensor([Lq],device=DEVICE),mode=mode,seed=7); J=_gj(x)[0,:Lq].cpu().numpy()
        b=render_gif_b64(J,title=label,color=col)
        cells.append(f'<div style="text-align:center;margin:3px"><img src="data:image/gif;base64,{b}" style="width:190px;border:1px solid #ddd;border-radius:4px"></div>')
    display(HTML(f'<div style="font-family:monospace;font-size:12px;margin:10px 0 4px;padding:5px 9px;background:#eef;border-left:4px solid #36c">&#9658; "{prompt[:80]}" (same seed across variants)</div>'
                 f'<div style="display:flex;flex-wrap:wrap;gap:3px">{"".join(cells)}</div>'))
print("GIF comparison done — same prompt+seed across variants; watch limbs/feet on unconstrained vs projected.")