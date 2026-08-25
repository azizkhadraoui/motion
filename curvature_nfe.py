#!/usr/bin/env python
# DOES TRAJECTORY CURVATURE PREDICT FEW-STEP DEGRADATION?  (workshop experiment, phase A)
#
# THE QUESTION. Reducing the number of sampling steps degrades generation quality. The standard
# explanation is discretization error of the generative ODE, which should be governed by how curved
# the solution trajectory is. In a LATENT flow model there are two different curvatures, and they are
# not the same object:
#
#   latent curvature   of the path the sampler integrates. This is what controls Euler error.
#   decoded curvature  of the same path pushed through the frozen decoder. This is the geometry of
#                      the space in which quality is actually measured.
#
# Because the decoder is nonlinear, a path that is nearly straight in Z can be curved in motion
# space, and error introduced in Z is amplified non-uniformly by D_phi. So the paper's question is
# not only WHETHER curvature predicts degradation but WHICH curvature does. If the decoded curvature
# is the better predictor, then methods that straighten the latent path -- reflow, OT couplings --
# are optimizing a geometry that is not the one governing the observed degradation, which is a
# concrete and testable implication.
#
# DESIGN. Cross-model comparison (LFM vs CDFM) is confounded: the two differ in dimensionality,
# architecture input and baseline FID, so a difference in NFE degradation has several candidate
# causes. The predictive claim is therefore made WITHIN each model, across clips:
#
#   per clip, at matched noise:
#     reference       x_ref  = sample at N_REF steps
#     degradation     d_n    = MPJPE(x_n, x_ref) in joint space, for each reduced n
#     predictors      latent arc/chord, decoded arc/chord, and velocity variation along the
#                     reference trajectory
#   then Spearman correlation of each predictor against d_n, per n, per model.
#
# Correlation alone is easy to over-read, so we also stratify: clips are split into curvature
# quartiles and FID is computed per quartile per step count. If the top-curvature quartile degrades
# faster than the bottom one within the same model, the relationship is visible distributionally and
# not only as a rank statistic.
#
# We additionally record the ARC-LENGTH PROFILE along t -- where along the trajectory the motion
# actually happens -- which phase B uses to construct a curvature-matched step schedule.
#
#   sbatch run_curvature.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[curv] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[curv] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
load_net = M.load_net; sample = M.sample

N_EVAL = int(os.environ.get("EVAL_N", "512"))
N_REF  = int(os.environ.get("CV_REF", "50"))
NFES   = [int(x) for x in os.environ.get("CV_NFE", "1,2,4,8,16").split(",")]
BASES  = os.environ.get("CV_BASES", "latent,direct").split(",")

def decode(y, is_lat):
    return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_lat else y

def masked_rms(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    d = ((A - B) ** 2).sum(-1).mean(-1) if A.dim() == 4 else ((A - B) ** 2).mean(-1)
    return ((d * fm).sum(1) / fm.sum(1).clamp_min(1)).sqrt()

def mpjpe(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    return ((A - B).norm(dim=-1).mean(-1) * fm).sum(1) / fm.sum(1).clamp_min(1)

@torch.no_grad()
def trajectory(net, is_lat, tseq, tmask, tpool, L, n, seed, trace=False):
    """Euler integration; optionally record latent and decoded states at every step."""
    torch.manual_seed(seed)
    B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    ts = _timesteps(n, "linear")
    zs, js, vs = [], [], []
    for i in range(n):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        tt = torch.full((B,), t, device=DEVICE)
        v = _cfg(net, z, tt, tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        if trace:
            zs.append(z.clone()); js.append(_gj(decode(z, is_lat))); vs.append(v.clone())
        z = z + dt * v
    mn = decode(z, is_lat)
    if trace:
        zs.append(z.clone()); js.append(_gj(mn))
    return mn, zs, js, vs

def arc_over_chord(states, L, joints):
    """Per-sample arc length divided by chord length, over a list of states along the trajectory."""
    arc = 0.0
    for a, b in zip(states[:-1], states[1:]):
        arc = arc + (masked_rms(b, a, L) if joints else (b - a).flatten(1).norm(dim=1))
    chord = masked_rms(states[-1], states[0], L) if joints else (states[-1] - states[0]).flatten(1).norm(dim=1)
    return (arc / chord.clamp_min(1e-8)), arc

def vel_variation(vs):
    """Mean deviation of the velocity field from its trajectory average, relative to its norm.
    This is the straightness measure used in the rectified-flow literature."""
    V = torch.stack(vs)                       # (n, B, ...)
    mean = V.mean(0, keepdim=True)
    num = (V - mean).flatten(2).norm(dim=2).mean(0)
    den = mean.flatten(2).norm(dim=2).squeeze(0).clamp_min(1e-8)
    return num / den

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "curvature_nfe"), job_type="analysis")

rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

out = {}
for tag in BASES:
    is_lat = (tag in ("latent", "latent_pen", "reflow"))
    if not M._have(tag):
        print(f"[curv] {tag} checkpoint missing, skipped"); continue
    net = load_net(tag, is_lat)
    print(f"\n{'='*104}\nMODEL {tag}  (latent={is_lat}), reference {N_REF} steps, {N_EVAL} clips\n{'='*104}")

    cur_lat, cur_dec, cur_vel = [], [], []
    deg = {n: [] for n in NFES}
    mf = {n: [] for n in NFES + [N_REF]}; real_mf = []
    arc_profile = np.zeros(N_REF); nprof = 0
    t0 = time.time()
    for s in range(0, N_EVAL, 32):
        e = min(s + 32, N_EVAL)
        ts_ = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
        tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
        gm = lengths_to_mask(L, MAXLEN)
        # reference trajectory, fully traced
        x_ref, zs, js, vs = trajectory(net, is_lat, ts_, tm, tp, L, N_REF, seed=s, trace=True)
        J_ref = _gj(x_ref)
        r_lat, _ = arc_over_chord(zs, L, joints=False)
        r_dec, _ = arc_over_chord(js, L, joints=True)
        cur_lat.append(r_lat.cpu()); cur_dec.append(r_dec.cpu()); cur_vel.append(vel_variation(vs).cpu())
        seg = np.array([float(masked_rms(js[k + 1], js[k], L).mean()) for k in range(N_REF)])
        arc_profile += seg; nprof += 1
        mf[N_REF].append(memb(x_ref * gm[..., None], L))
        for n in NFES:
            x_n, _, _, _ = trajectory(net, is_lat, ts_, tm, tp, L, n, seed=s)
            deg[n].append(mpjpe(_gj(x_n), J_ref, L).cpu())
            mf[n].append(memb(x_n * gm[..., None], L))
        rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
        real_mf.append(memb(rm * gm[..., None], L))
    print(f"  sampling done in {(time.time()-t0)/60:.1f}m")

    C = {"latent arc/chord": torch.cat(cur_lat).numpy(),
         "decoded arc/chord": torch.cat(cur_dec).numpy(),
         "velocity variation": torch.cat(cur_vel).numpy()}
    D = {n: torch.cat(deg[n]).numpy() for n in NFES}
    R = np.concatenate(real_mf, 0)
    fid = {n: float(fid_calc(np.concatenate(mf[n], 0), R)) for n in NFES + [N_REF]}

    print(f"\n  curvature summary (mean over clips)")
    for k, v in C.items():
        print(f"    {k:<22} {v.mean():.4f}   sd {v.std():.4f}   [{v.min():.3f}, {v.max():.3f}]")
    print(f"\n  FID vs steps: " + "  ".join(f"n={n}:{fid[n]:.4f}" for n in NFES + [N_REF]))

    print(f"\n  WITHIN-MODEL rank correlation of curvature with per-clip degradation")
    print(f"  {'predictor':<22}" + "".join(f"{'n='+str(n):>10}" for n in NFES))
    print("  " + "-" * (22 + 10 * len(NFES)))
    corr = {}
    for k, v in C.items():
        corr[k] = {n: spearman(v, D[n]) for n in NFES}
        print(f"  {k:<22}" + "".join(f"{corr[k][n]:>10.3f}" for n in NFES))

    # stratified FID by curvature quartile, using the better predictor
    best_pred = max(C, key=lambda k: np.nanmean([abs(corr[k][n]) for n in NFES]))
    q = np.quantile(C[best_pred], [0.25, 0.5, 0.75])
    bucket = np.digitize(C[best_pred], q)
    print(f"\n  stratified by {best_pred} (quartiles), FID per step count")
    print(f"  {'quartile':<12}" + "".join(f"{'n='+str(n):>10}" for n in NFES + [N_REF]))
    strat = {}
    G_all = {n: np.concatenate(mf[n], 0) for n in NFES + [N_REF]}
    for b in range(4):
        m = bucket == b
        if m.sum() < 32: continue
        strat[b] = {n: float(fid_calc(G_all[n][m], R[m])) for n in NFES + [N_REF]}
        print(f"  {'Q'+str(b+1):<12}" + "".join(f"{strat[b][n]:>10.4f}" for n in NFES + [N_REF]))
    if 0 in strat and 3 in strat:
        lo_n = NFES[0]
        gap_lo = strat[3][lo_n] - strat[0][lo_n]
        gap_hi = strat[3][N_REF] - strat[0][N_REF]
        print(f"\n  Q4 minus Q1 FID gap: {gap_lo:+.4f} at n={lo_n}, {gap_hi:+.4f} at n={N_REF}.")
        print("  A gap that widens as steps are removed is the distributional form of the claim:")
        print("  high-curvature clips lose more when the step budget shrinks.")

    out[tag] = dict(curvature={k: dict(mean=float(v.mean()), sd=float(v.std())) for k, v in C.items()},
                    fid=fid, correlations=corr, stratified=strat, best_predictor=best_pred,
                    arc_profile=(arc_profile / max(nprof, 1)).tolist(),
                    per_clip={k: v.tolist() for k, v in C.items()},
                    degradation={str(n): D[n].tolist() for n in NFES})

# ------------------------------------------------------------------ cross-model reading
print(f"\n{'='*104}\nREADING\n{'='*104}")
for tag, o in out.items():
    bp = o["best_predictor"]
    print(f"  [{tag}] strongest predictor: {bp}; "
          f"latent curvature {o['curvature']['latent arc/chord']['mean']:.4f}, "
          f"decoded {o['curvature']['decoded arc/chord']['mean']:.4f}")
if "latent" in out and "direct" in out:
    ll = out["latent"]["correlations"]["latent arc/chord"]
    ld = out["latent"]["correlations"]["decoded arc/chord"]
    n0 = NFES[0]
    print(f"\n  For LFM at n={n0}: latent-curvature correlation {ll[n0]:.3f}, "
          f"decoded-curvature correlation {ld[n0]:.3f}.")
    if abs(ld[n0]) > abs(ll[n0]) + 0.05:
        print("  Decoded curvature is the better predictor. The degradation is therefore governed by")
        print("  the geometry after decoding, not by the geometry the sampler integrates in. The")
        print("  implication worth stating: methods that straighten the LATENT path (reflow, OT")
        print("  couplings) target a curvature that is not the one predicting the observed loss.")
    elif abs(ll[n0]) > abs(ld[n0]) + 0.05:
        print("  Latent curvature is the better predictor, i.e. degradation tracks the integration")
        print("  error where it is generated. Latent-path straightening should then help, which is a")
        print("  clean prediction to test against the reflow student.")
    else:
        print("  The two predictors are comparable; report both and avoid claiming either governs.")
    print("\n  NOTE the cross-model comparison of absolute FID degradation between LFM and CDFM is")
    print("  confounded by architecture and dimensionality; the predictive claims above are all")
    print("  within-model across clips, which is why they are stated that way.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "curvature_nfe.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.1))
    for tag, o in out.items():
        ns = sorted(int(k) for k in o["fid"])
        axs[0].plot(ns, [o["fid"][n] for n in ns], "o-", ms=3, lw=1.1, label=tag)
        axs[2].plot(np.linspace(0, 1, len(o["arc_profile"])), o["arc_profile"], lw=1.1, label=tag)
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("sampling steps"); axs[0].set_ylabel("FID"); axs[0].legend(frameon=False, fontsize=6)
    axs[0].set_title("degradation vs step budget", fontsize=8, pad=3)
    for tag, o in out.items():
        for k, st in [("latent arc/chord", "o-"), ("decoded arc/chord", "s--")]:
            axs[1].plot(NFES, [o["correlations"][k][n] for n in NFES], st, ms=3, lw=1.1, label=f"{tag}: {k.split()[0]}")
    axs[1].axhline(0, color="0.5", lw=0.7)
    axs[1].set_xscale("log"); axs[1].set_xlabel("sampling steps")
    axs[1].set_ylabel("Spearman with degradation"); axs[1].legend(frameon=False, fontsize=5)
    axs[1].set_title("which curvature predicts?", fontsize=8, pad=3)
    axs[2].set_xlabel("$t$"); axs[2].set_ylabel("arc length per step (decoded)")
    axs[2].set_title("where the motion happens", fontsize=8, pad=3); axs[2].legend(frameon=False, fontsize=6)
    p = os.path.join(os.environ.get("WORK_DIR", "."), "fig_curvature_nfe.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"figure -> {p}")
    wandb.log({"curvature_fig": wandb.Image(p.replace(".pdf", ".png"))})
except Exception as e:
    print("figure failed:", e)
wandb.finish()
print("Curvature/NFE analysis done.")
