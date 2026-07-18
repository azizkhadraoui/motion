#!/usr/bin/env python
# Diagnostic: WHY is BLE ~0.12? Run this on the cluster FIRST, before any training.
# It loads real HumanML3D, computes bone-length reference from FK-recovered joints,
# and prints real-motion BLE. Real motion has near-rigid bones, so this MUST be ~0.01-0.02.
# If it prints ~0.12, the problem is in the data/normalization, not the generator, and we
# debug from these numbers. If it prints ~0.016, the 0.12 came from the generated motion
# (or the old run's normalization) and the metric itself is fine.
#
#   python ble_diagnostic.py --hml /export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
import argparse, glob, os, math, random
import numpy as np, torch
ap=argparse.ArgumentParser(); ap.add_argument("--hml",required=True); ap.add_argument("--n",type=int,default=500); a=ap.parse_args()
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
NFEATS=263; N_JOINTS=22; MAXL=196
_CHAINS=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
EDGES=[(c[k],c[k+1]) for c in _CHAINS for k in range(len(c)-1)]
EI=torch.tensor([e[0] for e in EDGES],device=DEV); EJ=torch.tensor([e[1] for e in EDGES],device=DEV)
FEET=[7,10,8,11]

def _qmul(a,b):
    aw,ax,ay,az=a[...,0],a[...,1],a[...,2],a[...,3]; bw,bx,by,bz=b[...,0],b[...,1],b[...,2],b[...,3]
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)
def _qinv(q): return q*torch.tensor([1,-1,-1,-1],dtype=q.dtype,device=q.device)
def _qapply(q,p):
    z=torch.zeros(p.shape[:-1],dtype=p.dtype,device=p.device); pq=torch.cat((z.unsqueeze(-1),p),-1); return _qmul(_qmul(q,pq),_qinv(q))[...,1:]
def recover(data,joints=N_JOINTS):
    rv=data[...,0]; ang=torch.zeros_like(rv); ang[...,1:]=rv[...,:-1]; ang=torch.cumsum(ang,-1)
    q=torch.zeros(data.shape[:-1]+(4,),device=data.device,dtype=data.dtype); q[...,0]=torch.cos(ang); q[...,2]=torch.sin(ang)
    rp=torch.zeros(data.shape[:-1]+(3,),device=data.device,dtype=data.dtype); rp[...,1:,[0,2]]=data[...,:-1,1:3]; rp=_qapply(q,rp); rp=torch.cumsum(rp,-2); rp[...,1]=data[...,3]
    p=data[...,4:(joints-1)*3+4].view(data.shape[:-1]+(-1,3)); p=_qapply(q[...,None,:].expand(p.shape[:-1]+(4,)),p); p[...,0]+=rp[...,0:1]; p[...,2]+=rp[...,2:3]
    return torch.cat([rp.unsqueeze(-2),p],dim=-2)

files=sorted(glob.glob(os.path.join(a.hml,"new_joint_vecs","*.npy")))
assert files, f"no .npy under {a.hml}/new_joint_vecs"
print(f"found {len(files)} motion files")
mean=np.load(os.path.join(a.hml,"Mean.npy")).astype(np.float32); std=np.load(os.path.join(a.hml,"Std.npy")).astype(np.float32); std[std<1e-6]=1e-6
print(f"Mean.npy shape {mean.shape}, Std.npy shape {std.shape}")

# sample real motions, recover FK joints
random.seed(0); sample=random.sample(files,min(a.n,len(files)))
per_bone=[]; all_bones=[]
for f in sample:
    m=np.load(f).astype(np.float32)
    if m.ndim!=2 or m.shape[1]!=NFEATS or len(m)<40: continue
    J=recover(torch.tensor(m[:MAXL],device=DEV)[None])              # (1,T,22,3)
    bl=(J[:,:,EI,:]-J[:,:,EJ,:]).norm(dim=-1)                       # (1,T,E)
    per_bone.append(bl.mean((0,1))); all_bones.append(bl.reshape(-1,len(EDGES)))
rest=torch.stack(per_bone).mean(0)                                  # FK rest lengths
print(f"\nrest_len (FK): {rest.shape[0]} bones, mean {rest.mean():.4f} m, min {rest.min():.4f}, max {rest.max():.4f}")

# real-motion BLE against FK rest_len -> MUST be tiny
bles=[]
for bl in all_bones: bles.append((bl-rest).abs().mean().item())
real_ble=float(np.mean(bles))
print(f"\n>>> real-motion BLE (FK rest_len) = {real_ble:.5f}")
print(f">>> per-bone std across frames (rigidity check) = {torch.cat(all_bones,0).std(0).mean():.5f}")
if real_ble < 0.03:
    print(">>> VERDICT: metric is HEALTHY. Real bones are near-rigid, BLE≈0. The 0.12 you saw")
    print("    came from the GENERATED motion at that checkpoint (or the old run's normalization),")
    print("    NOT from the metric. Projection will correctly drive generated BLE -> 0.")
else:
    print(">>> VERDICT: PROBLEM IS LIVE. Real-motion BLE is too high, so the reference or the")
    print("    FK/units are off. Do NOT trust BLE until this is resolved. Likely suspects:")
    print("    (1) Mean/Std mismatch, (2) motions already normalized on disk, (3) wrong joint count.")
    # extra probe: are the on-disk motions already normalized? (mean~0,std~1 would indicate yes)
    raw=np.load(sample[0]).astype(np.float32)
    print(f"    probe: sample motion raw stats  mean={raw.mean():.3f} std={raw.std():.3f}  (if ~0/~1, data is pre-normalized!)")
    print(f"    probe: Mean.npy stats           mean={mean.mean():.3f} std={std.mean():.3f}")
