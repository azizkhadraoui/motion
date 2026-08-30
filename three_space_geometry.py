#!/usr/bin/env python
# TRAJECTORY GEOMETRY IN THREE SPACES: Z, X AND Q.
#
# WHAT THIS ADDS. Everything so far measures the trajectory in two places, the integration space and
# the joint space reached by Psi = K o D_phi. That conflates two distinct maps, so the reported
# distortion cannot be attributed. This script measures the same 50-step reference trajectory at
# three points along the pipeline
#
#       Z  --D_phi-->  X  --K-->  Q
#
# and reports the arc-over-chord ratio and maximum turning angle in each, so the distortion splits
# into a decoder contribution (Z -> X) and a kinematic contribution (X -> Q).
#
# WHY THIS MATTERS FOR THE PAPER. Section 5 currently attributes the loss of trajectory regularity to
# "the decoder", and Section 3 notes in passing that the direct model's decoded ratio is not unity
# because K is itself nonlinear. Those two statements are in tension: if K alone bends the path
# substantially, part of what we call the decoder's distortion belongs to the representation and
# would be present in ANY model using the 263-d motion parameterization, latent or not. Measuring X
# settles it. The direct model is the control: for CDFM the integration space IS X, so its Z and X
# ratios must coincide up to numerical error, and its X -> Q increment isolates K on its own.
#
# THE NORMS. Euclidean on Z (49x256 flattened) and on X (196x263 flattened, frame-averaged), and the
# frame- and joint-averaged root-mean-square norm on Q, matching the analysis. Arc-over-chord is
# dimensionless, so the three are comparable despite the different units.
#
#   sbatch run_three_space.sh

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[3space] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[3space] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm; load_net = M.load_net

N_EVAL = int(os.environ.get("EVAL_N", "512"))
N_REF  = int(os.environ.get("TS_REF", "50"))
NFES   = [int(x) for x in os.environ.get("TS_NFE", "1,2,4,8,16").split(",")]
BASES  = os.environ.get("TS_BASES", "latent,direct").split(",")
REPS   = int(os.environ.get("TS_REPS", "5"))

TCRIT = {2:4.303,3:3.182,4:2.776,5:2.571,9:2.262,19:2.093}
def ci95(a):
    a=np.asarray(a,float); n=len(a)
    if n<2: return float(a.mean()), float("nan")
    return float(a.mean()), float(TCRIT.get(n-1,2.093)*a.std(ddof=1)/math.sqrt(n))
def spearman(x,y):
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return float("nan")
    return float(np.corrcoef(rx,ry)[0,1])

def decode(y, is_lat):
    """Z -> X. For the direct model this is the identity, since it generates in X already."""
    return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_lat else y

def d_flat(a, b):
    """Euclidean on the flattened state; used for Z and for X."""
    return (a - b).flatten(1).norm(dim=1)

def d_q(A, B, L):
    """Frame- and joint-averaged RMS on Q, restricted to valid frames."""
    fm = lengths_to_mask(L, MAXLEN).float()
    return (((A - B) ** 2).sum(-1).mean(-1) * fm).sum(1).div(fm.sum(1).clamp_min(1)).sqrt()

def mpjpe(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    return ((A - B).norm(dim=-1).mean(-1) * fm).sum(1) / fm.sum(1).clamp_min(1)

def arc_chord(states, metric):
    arc = 0.0
    for a, b in zip(states[:-1], states[1:]): arc = arc + metric(b, a)
    return arc / metric(states[-1], states[0]).clamp_min(1e-8)

def max_turn(states, metric_sub):
    d = [metric_sub(b, a) for a, b in zip(states[:-1], states[1:])]
    worst = torch.zeros(d[0].shape[0], device=d[0].device)
    for a, b in zip(d[:-1], d[1:]):
        c = (a*b).sum(1) / (a.norm(dim=1).clamp_min(1e-8) * b.norm(dim=1).clamp_min(1e-8))
        worst = torch.maximum(worst, torch.arccos(c.clamp(-1+1e-6, 1-1e-6)))
    return worst

@torch.no_grad()
def trace(net, is_lat, tseq, tmask, tpool, L, n, seed, want_trace=False):
    torch.manual_seed(seed); B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B,-1,-1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B,-1)
    ts = _timesteps(n, "linear"); Z, X, Q = [], [], []
    for i in range(n):
        t = torch.full((B,), float(ts[i]), device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        if want_trace:
            x = decode(z, is_lat); Z.append(z.clone()); X.append(x); Q.append(_gj(x))
        z = z + float(ts[i+1]-ts[i]) * v
    x = decode(z, is_lat)
    if want_trace: Z.append(z.clone()); X.append(x); Q.append(_gj(x))
    return z, x, _gj(x), Z, X, Q

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT","motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN","three_space_geometry"), job_type="analysis")

rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)
SP = ["Z", "X", "Q"]

out = {}
for tag in BASES:
    is_lat = (tag in ("latent","latent_pen","reflow"))
    if not M._have(tag): print(f"[3space] {tag} missing"); continue
    net = load_net(tag, is_lat)
    print(f"\n{'='*104}\nMODEL {tag}   {REPS} seeds x {N_EVAL} clips\n{'='*104}")
    if not is_lat:
        print("  note: CDFM generates in X, so its Z and X measurements are the same object;")
        print("        its X -> Q increment isolates the kinematic map on its own.")

    per = {"ac": {s: [] for s in SP}, "turn": {s: [] for s in SP},
           "corr": {s: {n: [] for n in NFES} for s in SP},
           "ratio_dec": [], "ratio_kin": [], "dump": None}
    t0 = time.time()
    for rep in range(REPS):
        AC = {s: [] for s in SP}; TU = {s: [] for s in SP}; deg = {n: [] for n in NFES}
        for s0 in range(0, N_EVAL, 32):
            e = min(s0+32, N_EVAL)
            ts_=torch.tensor(TSEQ[s0:e],device=DEVICE); tm=torch.tensor(TMASK[s0:e],device=DEVICE)
            tp=torch.tensor(TPOOL[s0:e],device=DEVICE); L=lens_all[s0:e]; seed=s0+rep*100000
            _,_,Qref,Z,X,Q = trace(net,is_lat,ts_,tm,tp,L,N_REF,seed,want_trace=True)
            AC["Z"].append(arc_chord(Z, d_flat).cpu())
            AC["X"].append(arc_chord(X, d_flat).cpu())
            AC["Q"].append(arc_chord(Q, lambda a,b: d_q(a,b,L)).cpu())
            TU["Z"].append(max_turn(Z, lambda b,a: (b-a).flatten(1)).cpu())
            TU["X"].append(max_turn(X, lambda b,a: (b-a).flatten(1)).cpu())
            TU["Q"].append(max_turn(Q, lambda b,a: (b-a).flatten(2).flatten(1)).cpu())
            for n in NFES:
                _,_,Qn,_,_,_ = trace(net,is_lat,ts_,tm,tp,L,n,seed)
                deg[n].append(mpjpe(Qn, Qref, L).cpu())
        AC = {s: torch.cat(v).numpy() for s,v in AC.items()}
        TU = {s: torch.cat(v).numpy() for s,v in TU.items()}
        D  = {n: torch.cat(deg[n]).numpy() for n in NFES}
        for s in SP:
            per["ac"][s].append(float(AC[s].mean())); per["turn"][s].append(float(TU[s].mean()))
            for n in NFES: per["corr"][s][n].append(spearman(AC[s], D[n]))
        per["ratio_dec"].append(float((AC["X"]/AC["Z"]).mean()))
        per["ratio_kin"].append(float((AC["Q"]/AC["X"]).mean()))
        if rep == 0: per["dump"] = {s: AC[s].tolist() for s in SP}
        print(f"  rep {rep+1}/{REPS}  ({(time.time()-t0)/60:.1f}m)")

    print(f"\n  {'space':<6}{'arc/chord':>22}{'sd across clips':>18}{'max turn':>14}")
    for s in SP:
        m,h = ci95(per["ac"][s]); tm_,_ = ci95(per["turn"][s])
        sd = float(np.std(per["dump"][s])) if per["dump"] else float("nan")
        print(f"  {s:<6}{m:>14.4f} +/-{h:<6.4f}{sd:>18.4f}{tm_:>14.4f}")
    rd, rdh = ci95(per["ratio_dec"]); rk, rkh = ci95(per["ratio_kin"])
    print(f"\n  distortion decomposition (per-clip ratio of arc/chord, mean over clips and seeds)")
    print(f"    decoder      X/Z  = {rd:.4f} +/- {rdh:.4f}")
    print(f"    kinematic    Q/X  = {rk:.4f} +/- {rkh:.4f}")

    print(f"\n  {'space':<6}" + "".join(f"{'n='+str(n):>16}" for n in NFES))
    for s in SP:
        print(f"  {s:<6}" + "".join(f"{ci95(per['corr'][s][n])[0]:>10.3f}+/-{ci95(per['corr'][s][n])[1]:<5.3f}" for n in NFES))

    if is_lat:
        print("\n  READING")
        print(f"    Of the total distortion from Z to Q, the decoder contributes a factor {rd:.3f}")
        print(f"    and the kinematic map a factor {rk:.3f}.")
        if rk > rd:
            print("    >>> The KINEMATIC MAP is the larger source. Part of what Section 5 attributes to")
            print("        the decoder belongs to the 263-d motion parameterization and would appear in")
            print("        any model using it. Section 5 must be reworded, and the direct model's own")
            print("        X -> Q increment (printed above for CDFM) is the evidence.")
        else:
            print("    >>> The DECODER is the larger source, which supports the current wording. Report")
            print("        both increments so the attribution is explicit rather than assumed.")
        cz = ci95(per["corr"]["Z"][NFES[-1]])[0]; cx = ci95(per["corr"]["X"][NFES[-1]])[0]
        cq = ci95(per["corr"]["Q"][NFES[-1]])[0]
        print(f"    Correlation with degradation at n={NFES[-1]}: Z {cz:.3f}, X {cx:.3f}, Q {cq:.3f}.")
        print("    If X already fails to predict, the predictive signal is lost at the decoder rather")
        print("    than at the kinematic map, which localizes the claim more precisely than we can now.")

    out[tag] = dict(
        arc_chord={s: dict(zip(("mean","ci"), ci95(per["ac"][s]))) for s in SP},
        max_turn={s: dict(zip(("mean","ci"), ci95(per["turn"][s]))) for s in SP},
        sd_across_clips={s: float(np.std(per["dump"][s])) for s in SP} if per["dump"] else {},
        ratio_decoder=dict(zip(("mean","ci"), ci95(per["ratio_dec"]))),
        ratio_kinematic=dict(zip(("mean","ci"), ci95(per["ratio_kin"]))),
        correlations={s: {str(n): dict(zip(("mean","ci"), ci95(per["corr"][s][n]))) for n in NFES} for s in SP},
        per_clip_seed0=per["dump"])

dst = os.path.join(os.environ.get("WORK_DIR","."), "three_space_geometry.json")
json.dump(out, open(dst,"w"), indent=2); print(f"\nraw results -> {dst}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.grid":True,"grid.alpha":.22})
    ms=[m for m in out if out[m].get("per_clip_seed0")]
    f,axs=plt.subplots(1,len(ms),figsize=(2.9*len(ms),2.1),squeeze=False)
    C={"Z":"#1f4e79","X":"#c9772e","Q":"#c0504d"}
    for i,m in enumerate(ms):
        a=axs[0][i]; d=out[m]["per_clip_seed0"]
        hi=max(2.6,np.percentile(d["Q"],99))
        b=np.linspace(1.0,hi,64)
        for s in SP: a.hist(d[s],bins=b,color=C[s],alpha=.62,label=f"$\\mathcal{{{s}}}$")
        a.set_yscale("log"); a.set_xlabel("arc length / chord")
        if i==0: a.set_ylabel("clips"); a.legend(frameon=False,fontsize=6.5)
        a.set_title({"latent":"LFM","direct":"CDFM"}.get(m,m),pad=3)
    p=os.path.join(os.environ.get("WORK_DIR","."),"fig_three_space.pdf")
    f.savefig(p,bbox_inches="tight"); f.savefig(p[:-4]+".png",dpi=220,bbox_inches="tight")
    print(f"figure -> {p}")
    wandb.log({"three_space": wandb.Image(p[:-4]+".png")})
except Exception as e:
    print("figure failed:", e)
wandb.finish()
print("Three-space geometry done.")
