#!/usr/bin/env python
# CURVATURE AND FEW-STEP DEGRADATION, REPLICATED, WITH TWO ADDITIONAL PREDICTORS.
#
# Supersedes curvature_nfe.py. Three changes, each closing a gap in the paper:
#
# 1. REPLICATION. Every statistic is computed independently per sampling seed and reported as a mean
#    with a t-based 95% interval across seeds. The correlations in the single-seed run (0.25-0.42)
#    are the weakest link in the argument and cannot be published without error bars.
#
# 2. DECODER SENSITIVITY, the predictor the error analysis asks for. The Euler bound
#
#        || Psi(Yhat_1) - Psi(Y_1) ||  <=  L_Psi * (h/2) * C * sup_t || Y''_t ||
#
#    contains the decoding map only through its Lipschitz constant L_Psi, and contains no curvature
#    of the DECODED path at all. That is why we expect integration-space curvature to predict and
#    decoded curvature not to. But the bound treats L_Psi as a single constant, whereas the relevant
#    quantity is the LOCAL sensitivity of the decoder where each trajectory terminates. If that
#    varied strongly across samples, the decoder could still modulate degradation without decoded
#    curvature mattering. We therefore estimate, per clip, the operator norm of the decoding Jacobian
#    at the terminal state by finite-difference Jacobian-vector products with random probes, and test
#    it as a third predictor. This is the measurement that completes the theoretical argument rather
#    than leaving L_Psi as an unexamined constant.
#
# 3. LOCAL CURVATURE. Arc over chord summarizes a whole trajectory in one number and is blind to
#    where the bending happens; the paper lists this as a limitation. We add the maximum turning
#    angle between consecutive integration steps, which is local and dimensionless, and report it
#    alongside. If it predicts better than arc/chord, the limitation is addressed rather than merely
#    acknowledged.
#
# Also dumps per-clip predictor values for the distribution figure, and emits that figure directly.
#
#   sbatch run_curvature_v2.sh

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[curv2] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[curv2] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
load_net = M.load_net

N_EVAL = int(os.environ.get("EVAL_N", "512"))
N_REF  = int(os.environ.get("CV_REF", "50"))
NFES   = [int(x) for x in os.environ.get("CV_NFE", "1,2,4,8,16").split(",")]
BASES  = os.environ.get("CV_BASES", "latent,direct").split(",")
REPS   = int(os.environ.get("CV_REPS", "5"))
JAC_PROBES = int(os.environ.get("CV_JAC_PROBES", "4"))
JAC_EPS = float(os.environ.get("CV_JAC_EPS", "1e-2"))

TCRIT = {2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,9:2.262,19:2.093}
def ci95(a):
    a = np.asarray(a, float); n = len(a)
    if n < 2: return float(a.mean()), float("nan")
    return float(a.mean()), float(TCRIT.get(n - 1, 2.093) * a.std(ddof=1) / math.sqrt(n))

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

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
def decoder_sensitivity(z, is_lat, L, probes=JAC_PROBES, eps=JAC_EPS):
    """Per-clip estimate of the local operator norm of Psi = K o D at the terminal state, by
    finite-difference Jacobian-vector products with random unit probes. Returns (B,)."""
    base = _gj(decode(z, is_lat))
    acc = torch.zeros(z.shape[0], device=z.device)
    for _ in range(probes):
        v = torch.randn_like(z)
        v = v / v.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, *([1] * (z.dim() - 1)))
        pert = _gj(decode(z + eps * v, is_lat))
        acc += masked_rms(pert, base, L) / eps
    return acc / probes

@torch.no_grad()
def trajectory(net, is_lat, tseq, tmask, tpool, L, n, seed, trace=False):
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
        if trace: zs.append(z.clone()); js.append(_gj(decode(z, is_lat))); vs.append(v.clone())
        z = z + dt * v
    if trace: zs.append(z.clone()); js.append(_gj(decode(z, is_lat)))
    return z, zs, js, vs

def arc_over_chord(states, L, joints):
    arc = 0.0
    for a, b in zip(states[:-1], states[1:]):
        arc = arc + (masked_rms(b, a, L) if joints else (b - a).flatten(1).norm(dim=1))
    chord = masked_rms(states[-1], states[0], L) if joints else (states[-1] - states[0]).flatten(1).norm(dim=1)
    return arc / chord.clamp_min(1e-8)

def max_turn(states):
    """Largest angle (radians) between consecutive step displacements: a LOCAL curvature measure."""
    d = [ (b - a).flatten(1) for a, b in zip(states[:-1], states[1:]) ]
    worst = torch.zeros(d[0].shape[0], device=d[0].device)
    for a, b in zip(d[:-1], d[1:]):
        cos = (a * b).sum(1) / (a.norm(dim=1).clamp_min(1e-8) * b.norm(dim=1).clamp_min(1e-8))
        worst = torch.maximum(worst, torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)))
    return worst

def vel_variation(vs):
    V = torch.stack(vs); mean = V.mean(0, keepdim=True)
    return V.sub(mean).flatten(2).norm(dim=2).mean(0) / mean.flatten(2).norm(dim=2).squeeze(0).clamp_min(1e-8)

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "curvature_nfe_v2"), job_type="analysis")

rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)
PRED = ["latent arc/chord", "decoded arc/chord", "velocity variation",
        "max turn (local)", "decoder sensitivity"]

out = {}
for tag in BASES:
    is_lat = (tag in ("latent", "latent_pen", "reflow"))
    if not M._have(tag):
        print(f"[curv2] {tag} missing, skipped"); continue
    net = load_net(tag, is_lat)
    print(f"\n{'='*104}\nMODEL {tag}   {REPS} seeds x {N_EVAL} clips\n{'='*104}")

    per_rep = {"curv": {k: [] for k in PRED}, "corr": {k: {n: [] for n in NFES} for k in PRED},
               "fid": {n: [] for n in NFES + [N_REF]}, "strat": {q: {n: [] for n in NFES + [N_REF]} for q in range(4)}}
    dump0 = None
    t0 = time.time()
    for rep in range(REPS):
        C = {k: [] for k in PRED}; deg = {n: [] for n in NFES}
        mf = {n: [] for n in NFES + [N_REF]}; real_mf = []
        for s in range(0, N_EVAL, 32):
            e = min(s + 32, N_EVAL)
            ts_ = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
            tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
            gm = lengths_to_mask(L, MAXLEN)
            seed = s + rep * 100000
            z1, zs, js, vs = trajectory(net, is_lat, ts_, tm, tp, L, N_REF, seed=seed, trace=True)
            x_ref = decode(z1, is_lat); J_ref = _gj(x_ref)
            C["latent arc/chord"].append(arc_over_chord(zs, L, False).cpu())
            C["decoded arc/chord"].append(arc_over_chord(js, L, True).cpu())
            C["velocity variation"].append(vel_variation(vs).cpu())
            C["max turn (local)"].append(max_turn(zs).cpu())
            C["decoder sensitivity"].append(decoder_sensitivity(z1, is_lat, L).cpu())
            mf[N_REF].append(memb(x_ref * gm[..., None], L))
            for n in NFES:
                zn, _, _, _ = trajectory(net, is_lat, ts_, tm, tp, L, n, seed=seed)
                xn = decode(zn, is_lat)
                deg[n].append(mpjpe(_gj(xn), J_ref, L).cpu())
                mf[n].append(memb(xn * gm[..., None], L))
            rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
            real_mf.append(memb(rm * gm[..., None], L))

        C = {k: torch.cat(v).numpy() for k, v in C.items()}
        D = {n: torch.cat(deg[n]).numpy() for n in NFES}
        R = np.concatenate(real_mf, 0); G = {n: np.concatenate(mf[n], 0) for n in NFES + [N_REF]}
        for k in PRED:
            per_rep["curv"][k].append(float(C[k].mean()))
            for n in NFES: per_rep["corr"][k][n].append(spearman(C[k], D[n]))
        for n in NFES + [N_REF]: per_rep["fid"][n].append(float(fid_calc(G[n], R)))
        q = np.quantile(C["latent arc/chord"], [0.25, 0.5, 0.75]); b = np.digitize(C["latent arc/chord"], q)
        for qi in range(4):
            m = b == qi
            if m.sum() >= 32:
                for n in NFES + [N_REF]: per_rep["strat"][qi][n].append(float(fid_calc(G[n][m], R[m])))
        if rep == 0: dump0 = {k: C[k].tolist() for k in PRED}
        print(f"  rep {rep+1}/{REPS}  ({(time.time()-t0)/60:.1f}m)")

    print(f"\n  predictor means over {REPS} seeds")
    for k in PRED:
        m, h = ci95(per_rep["curv"][k]); print(f"    {k:<22} {m:.4f} +/- {h:.4f}")
    print(f"\n  Spearman with degradation (mean +/- 95% CI over seeds)")
    print(f"  {'predictor':<22}" + "".join(f"{'n='+str(n):>18}" for n in NFES))
    for k in PRED:
        row = "".join(f"{ci95(per_rep['corr'][k][n])[0]:>11.3f} +/-{ci95(per_rep['corr'][k][n])[1]:>5.3f}" for n in NFES)
        print(f"  {k:<22}{row}")
    print(f"\n  stratified FID by latent-curvature quartile (mean over seeds)")
    print(f"  {'quartile':<10}" + "".join(f"{'n='+str(n):>10}" for n in NFES + [N_REF]))
    for qi in range(4):
        if per_rep["strat"][qi][NFES[0]]:
            print(f"  {'Q'+str(qi+1):<10}" + "".join(f"{ci95(per_rep['strat'][qi][n])[0]:>10.3f}" for n in NFES + [N_REF]))

    best = max(PRED, key=lambda k: abs(ci95(per_rep["corr"][k][NFES[-1]])[0]))
    print(f"\n  strongest predictor at n={NFES[-1]}: {best}")
    ds = ci95(per_rep["corr"]["decoder sensitivity"][NFES[-1]])
    print(f"  decoder sensitivity correlation: {ds[0]:.3f} +/- {ds[1]:.3f}")
    if abs(ds[0]) < 0.1:
        print("  -> local decoder sensitivity does not predict degradation either. Combined with the")
        print("     decoded-curvature null, the decoder influences the outcome through neither the")
        print("     shape of the decoded path nor its local gain, which is the strongest form of the")
        print("     claim the error analysis permits.")
    else:
        print("  -> local decoder sensitivity DOES predict. Report it: the decoder modulates")
        print("     degradation through its local gain (the L_Psi factor) even though decoded")
        print("     curvature is irrelevant. This refines rather than contradicts the analysis.")
    mt = ci95(per_rep["corr"]["max turn (local)"][NFES[-1]])[0]
    la = ci95(per_rep["corr"]["latent arc/chord"][NFES[-1]])[0]
    print(f"  local turning angle {mt:.3f} vs global arc/chord {la:.3f}: "
          f"{'local is the better summary' if abs(mt) > abs(la) + 0.03 else 'global summary suffices'}")

    out[tag] = dict(reps=REPS, n_eval=N_EVAL,
                    curvature={k: dict(zip(("mean", "ci"), ci95(per_rep["curv"][k]))) for k in PRED},
                    correlations={k: {str(n): dict(zip(("mean", "ci"), ci95(per_rep["corr"][k][n]))) for n in NFES} for k in PRED},
                    fid={str(n): dict(zip(("mean", "ci"), ci95(per_rep["fid"][n]))) for n in NFES + [N_REF]},
                    stratified={str(q): {str(n): ci95(per_rep["strat"][q][n])[0] for n in NFES + [N_REF]}
                                for q in range(4) if per_rep["strat"][q][NFES[0]]},
                    per_clip_seed0=dump0)

dst = os.path.join(os.environ.get("WORK_DIR", "."), "curvature_nfe_v2.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")

# ---- distribution figure: the regularity claim, shown rather than tabulated ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, axs = plt.subplots(1, len(out), figsize=(3.1 * len(out), 2.2), squeeze=False)
    for i, (tag, o) in enumerate(out.items()):
        ax = axs[0][i]; d = o["per_clip_seed0"]
        bins = np.linspace(1.0, max(2.6, np.percentile(d["decoded arc/chord"], 99)), 60)
        ax.hist(d["latent arc/chord"], bins=bins, color="#1f4e79", alpha=0.85, label="integration space")
        ax.hist(d["decoded arc/chord"], bins=bins, color="#c0504d", alpha=0.6, label="decoded")
        ax.set_yscale("log"); ax.set_xlabel("arc length / chord"); ax.set_ylabel("clips")
        ax.set_title(tag.upper(), fontsize=8.5, pad=3)
        if i == 0: ax.legend(frameon=False, fontsize=6.5)
    p = os.path.join(os.environ.get("WORK_DIR", "."), "fig_curvature_dist.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"figure -> {p}")
    wandb.log({"curvature_dist": wandb.Image(p.replace(".pdf", ".png"))})
except Exception as e:
    print("figure failed:", e)
wandb.finish()
print("Replicated curvature study done.")
