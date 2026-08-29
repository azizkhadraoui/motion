#!/usr/bin/env python
# IS THE OUTPUT-MAP GAIN A PREDICTOR, OR JUST AN AMPLIFIER?
#
# THE PROBLEM. The replicated run found that the local gain of the output map,
# ||J_Psi|| with Psi = K o D, is the strongest single predictor of few-step degradation
# (Spearman 0.737 +/- 0.018 for LFM at n=16, 0.600 +/- 0.033 for CDFM). Taken at face value that
# would be a striking result. But it is partly definitional. The Euler bound reads
#
#     || Psi(Yhat_1) - Psi(Y_1) ||  <=  L_Psi * (h/2) * C * sup_t || Y''_t ||,
#
# and degradation is measured on the LEFT, in output space. A clip whose output map has high local
# gain therefore has amplified degradation whatever its trajectory did. Reporting 0.737 without
# addressing this would invite the obvious objection that we have measured a multiplication.
#
# THE TEST. Measure degradation in the INTEGRATION space as well,
#
#     d_Z(n) = || Yhat^(n)_1 - Y^(50)_1 ||   (Euclidean, the space the solver acts in),
#
# where the output map plays no part and no amplification can occur by construction. Then:
#
#   ||J_Psi|| still predicts d_Z   ->  the gain carries information about the trajectory itself.
#                                      High-gain regions are also regions the flow traverses
#                                      differently. Report it as a genuine predictor and say why.
#   ||J_Psi|| does not predict d_Z ->  the gain acts purely as the multiplicative constant of the
#                                      bound. That is CONFIRMATION of the analysis, not a new
#                                      mechanism, and the honest statement is that the output map
#                                      contributes exactly and only the factor L_Psi.
#
# Both outcomes are reportable and both are stated before seeing the numbers, which is the point.
#
# A partial correlation is also computed: the correlation between ||J_Psi|| and output-space
# degradation after removing the linear contribution of integration-space degradation. If the gain
# were pure amplification this would remain high (it is the amplification), so it is reported as a
# descriptive quantity rather than as evidence either way; the d_Z correlation is the decisive one.
#
# SECOND PURPOSE: DUAL-NORM DEGRADATION. The analysis is stated in the frame- and joint-averaged RMS
# norm on Q, while degradation was reported as MPJPE, an averaged l_{2,1} mixed norm. They are
# equivalent up to a constant and MPJPE <= RMS by Cauchy-Schwarz, but the paper should not measure
# one side of an inequality in one norm and the ingredients of the other side in a different one.
# This script emits both, so the correspondence can be shown rather than asserted.
#
#   sbatch run_disentangle.sh

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[disent] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[disent] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; load_net = M.load_net

N_EVAL = int(os.environ.get("EVAL_N", "512"))
N_REF  = int(os.environ.get("DS_REF", "50"))
NFES   = [int(x) for x in os.environ.get("DS_NFE", "1,2,4,8,16").split(",")]
BASES  = os.environ.get("DS_BASES", "latent,direct").split(",")
REPS   = int(os.environ.get("DS_REPS", "5"))
PROBES = int(os.environ.get("DS_PROBES", "4"))
EPS    = float(os.environ.get("DS_EPS", "1e-2"))

TCRIT = {2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,9:2.262,19:2.093}
def ci95(a):
    a = np.asarray(a, float); n = len(a)
    if n < 2: return float(a.mean()), float("nan")
    return float(a.mean()), float(TCRIT.get(n - 1, 2.093) * a.std(ddof=1) / math.sqrt(n))

def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def partial_spearman(x, y, z):
    """Rank correlation of x and y with the linear effect of z removed (on ranks)."""
    rx, ry, rz = rank(x), rank(y), rank(z)
    def resid(a, b):
        b0 = b - b.mean(); denom = (b0 ** 2).sum()
        return a - a.mean() - (((a - a.mean()) * b0).sum() / max(denom, 1e-12)) * b0
    ex, ey = resid(rx, rz), resid(ry, rz)
    if ex.std() == 0 or ey.std() == 0: return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])

def decode(y, is_lat):
    return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_lat else y

def rms_q(A, B, L):
    """Frame- and joint-averaged RMS on Q: the norm the analysis is stated in."""
    fm = lengths_to_mask(L, MAXLEN).float()
    d = ((A - B) ** 2).sum(-1).mean(-1)
    return ((d * fm).sum(1) / fm.sum(1).clamp_min(1)).sqrt()

def mpjpe_q(A, B, L):
    """Averaged l_{2,1} on Q: the standard motion metric."""
    fm = lengths_to_mask(L, MAXLEN).float()
    return ((A - B).norm(dim=-1).mean(-1) * fm).sum(1) / fm.sum(1).clamp_min(1)

def euclid_z(a, b):
    """Euclidean norm in the integration space -- no output map, so no amplification."""
    return (a - b).flatten(1).norm(dim=1)

@torch.no_grad()
def gain(z, is_lat, L, probes=PROBES, eps=EPS):
    """Finite-difference estimate of the local gain of Psi at the state z.

    NOTE ON WHAT THIS ESTIMATES. Averaging the response to isotropic random probes approximates
    E_v ||J_Psi v||, closer to a normalized Frobenius norm than to the operator norm ||J_Psi||_2.
    Report it as an estimate g-hat, not as ||J_Psi||."""
    base = _gj(decode(z, is_lat)); acc = torch.zeros(z.shape[0], device=z.device)
    for _ in range(probes):
        v = torch.randn_like(z)
        v = v / v.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, *([1] * (z.dim() - 1)))
        acc += rms_q(_gj(decode(z + eps * v, is_lat)), base, L) / eps
    return acc / probes

@torch.no_grad()
def gain_trajectory(net, is_lat, tseq, tmask, tpool, L, seed, ts_eval=(0.25, 0.5, 0.75, 1.0),
                    n=None):
    """Gain evaluated at several points ALONG the reference trajectory. Returns (mean, max).

    Equation (2)'s constant is a supremum over a region, not a point evaluation, so a trajectory
    maximum is closer to it than the terminal value alone -- though still over the trajectory rather
    than over the endpoint segment, which `gain_segment` handles."""
    n = n or N_REF
    torch.manual_seed(seed); B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    ts = _timesteps(n, "linear"); marks = set(int(round(x * n)) for x in ts_eval)
    vals = []
    for i in range(n):
        if i in marks: vals.append(gain(z, is_lat, L))
        t = torch.full((B,), float(ts[i]), device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        z = z + float(ts[i + 1] - ts[i]) * v
    vals.append(gain(z, is_lat, L))
    V = torch.stack(vals)
    return V.mean(0), V.max(0).values, z

@torch.no_grad()
def gain_segment(z_approx, z_ref, is_lat, L):
    """Gain at the midpoint of the segment joining the approximate and exact endpoints -- the region
    Equation (2) actually concerns, and the only variant that depends on the step budget n."""
    return gain(0.5 * (z_approx + z_ref), is_lat, L)

@torch.no_grad()
def terminal(net, is_lat, tseq, tmask, tpool, L, n, seed):
    torch.manual_seed(seed); B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    ts = _timesteps(n, "linear")
    for i in range(n):
        t = torch.full((B,), float(ts[i]), device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        z = z + float(ts[i + 1] - ts[i]) * v
    return z

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "gain_disentangle"), job_type="analysis")

rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

out = {}
for tag in BASES:
    is_lat = (tag in ("latent", "latent_pen", "reflow"))
    if not M._have(tag):
        print(f"[disent] {tag} missing, skipped"); continue
    net = load_net(tag, is_lat)
    print(f"\n{'='*104}\nMODEL {tag}   {REPS} seeds x {N_EVAL} clips\n{'='*104}")

    KEYS = ("gain_vs_Q_rms", "gain_vs_Q_mpjpe", "gain_vs_Z", "partial_gain", "rms_vs_mpjpe",
            "gmean_vs_Z", "gmax_vs_Z", "gseg_vs_Z", "gseg_vs_Q_rms", "gmax_vs_Q_rms")
    acc = {k: {n: [] for n in NFES} for k in KEYS}
    t0 = time.time()
    for rep in range(REPS):
        G, GM, GX = [], [], []
        GS = {n: [] for n in NFES}
        dQ_rms = {n: [] for n in NFES}; dQ_mp = {n: [] for n in NFES}; dZ = {n: [] for n in NFES}
        for s in range(0, N_EVAL, 32):
            e = min(s + 32, N_EVAL)
            ts_ = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
            tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
            seed = s + rep * 100000
            gmean, gmax, zref = gain_trajectory(net, is_lat, ts_, tm, tp, L, seed)
            Jref = _gj(decode(zref, is_lat))
            G.append(gain(zref, is_lat, L).cpu()); GM.append(gmean.cpu()); GX.append(gmax.cpu())
            for n in NFES:
                zn = terminal(net, is_lat, ts_, tm, tp, L, n, seed)
                Jn = _gj(decode(zn, is_lat))
                dQ_rms[n].append(rms_q(Jn, Jref, L).cpu())
                dQ_mp[n].append(mpjpe_q(Jn, Jref, L).cpu())
                dZ[n].append(euclid_z(zn, zref).cpu())
                GS[n].append(gain_segment(zn, zref, is_lat, L).cpu())
        G = torch.cat(G).numpy(); GM = torch.cat(GM).numpy(); GX = torch.cat(GX).numpy()
        for n in NFES:
            a = torch.cat(dQ_rms[n]).numpy(); b = torch.cat(dQ_mp[n]).numpy(); c = torch.cat(dZ[n]).numpy()
            gs = torch.cat(GS[n]).numpy()
            acc["gain_vs_Q_rms"][n].append(spearman(G, a))
            acc["gain_vs_Q_mpjpe"][n].append(spearman(G, b))
            acc["gain_vs_Z"][n].append(spearman(G, c))
            acc["partial_gain"][n].append(partial_spearman(G, a, c))
            acc["rms_vs_mpjpe"][n].append(spearman(a, b))
            acc["gmean_vs_Z"][n].append(spearman(GM, c))
            acc["gmax_vs_Z"][n].append(spearman(GX, c))
            acc["gseg_vs_Z"][n].append(spearman(gs, c))
            acc["gseg_vs_Q_rms"][n].append(spearman(gs, a))
            acc["gmax_vs_Q_rms"][n].append(spearman(GX, a))
        print(f"  rep {rep+1}/{REPS}  ({(time.time()-t0)/60:.1f}m)")

    print(f"\n  {'quantity':<46}" + "".join(f"{'n='+str(n):>16}" for n in NFES))
    print("  " + "-" * (46 + 16 * len(NFES)))
    rows = [("TERMINAL gain vs degradation in Q (rms)",   "gain_vs_Q_rms"),
            ("TERMINAL gain vs degradation in Q (MPJPE)", "gain_vs_Q_mpjpe"),
            ("TERMINAL gain vs degradation in Z  <-- decisive", "gain_vs_Z"),
            ("TRAJ-MEAN gain vs degradation in Z",       "gmean_vs_Z"),
            ("TRAJ-MAX  gain vs degradation in Z",       "gmax_vs_Z"),
            ("SEGMENT   gain vs degradation in Z",       "gseg_vs_Z"),
            ("SEGMENT   gain vs degradation in Q (rms)", "gseg_vs_Q_rms"),
            ("TRAJ-MAX  gain vs degradation in Q (rms)", "gmax_vs_Q_rms"),
            ("TERMINAL gain vs Q, controlling for Z",    "partial_gain"),
            ("rms vs MPJPE agreement (sanity)",          "rms_vs_mpjpe")]
    for lab, k in rows:
        print(f"  {lab:<46}" + "".join(f"{ci95(acc[k][n])[0]:>10.3f}+/-{ci95(acc[k][n])[1]:<5.3f}" for n in NFES))

    nz = NFES[-1]
    gq, _ = ci95(acc["gain_vs_Q_rms"][nz]); gz, gzc = ci95(acc["gain_vs_Z"][nz])
    agree, _ = ci95(acc["rms_vs_mpjpe"][nz])
    print(f"\n  VERDICT at n={nz}")
    print(f"    gain vs output-space degradation: {gq:.3f}")
    print(f"    gain vs integration-space degradation: {gz:.3f} +/- {gzc:.3f}")
    if abs(gz) < 0.15:
        print("    >>> The gain does NOT predict degradation where no amplification is possible.")
        print("        It therefore acts purely as the multiplicative factor L_Psi of the bound.")
        print("        Write it that way: the output map contributes exactly and only the constant,")
        print("        and the 0.74 correlation in output space is that constant doing its work.")
        print("        This is confirmation of the analysis, not a separate mechanism, and it must")
        print("        not be presented as the paper's headline predictor.")
    elif abs(gz) > 0.4:
        print("    >>> The gain predicts degradation even in the integration space, where it cannot")
        print("        amplify anything. It therefore carries information about the trajectory")
        print("        itself: regions of high output-map gain are regions the flow traverses")
        print("        differently. This is a genuine finding and can be reported as a predictor,")
        print("        with the amplification caveat acknowledged and answered by this control.")
    else:
        print("    >>> Intermediate. Part amplification, part trajectory information. Report both")
        print("        columns and describe it as partially confounded rather than picking a side.")
    print(f"\n    rms/MPJPE rank agreement: {agree:.3f}. The two norms order clips near-identically,")
    print("    so stating the analysis in rms and reporting MPJPE changes no conclusion.")

    # which evaluation point best matches the constant in Equation (2)?
    lo, hi = NFES[0], NFES[-1]
    seg_lo = ci95(acc["gseg_vs_Q_rms"][lo])[0]; ter_lo = ci95(acc["gain_vs_Q_rms"][lo])[0]
    seg_hi = ci95(acc["gseg_vs_Q_rms"][hi])[0]; ter_hi = ci95(acc["gain_vs_Q_rms"][hi])[0]
    print(f"\n    EVALUATION POINT. Equation (2)'s constant is a supremum over the region between")
    print(f"    the exact and approximate endpoints, so the SEGMENT variant is the faithful one and")
    print(f"    is the only variant that depends on n. Its advantage over TERMINAL should GROW as n")
    print(f"    shrinks, because that is where the endpoints separate.")
    print(f"      n={lo}:  segment {seg_lo:.3f} vs terminal {ter_lo:.3f}   (delta {seg_lo-ter_lo:+.3f})")
    print(f"      n={hi}: segment {seg_hi:.3f} vs terminal {ter_hi:.3f}   (delta {seg_hi-ter_hi:+.3f})")
    if (seg_lo - ter_lo) > (seg_hi - ter_hi) + 0.03:
        print("      >>> The predicted pattern holds: the faithful variant gains most where the")
        print("          region is largest. Report SEGMENT as the primary measurement; this is")
        print("          evidence the bound describes the mechanism rather than merely bounding it.")
    else:
        print("      >>> The predicted pattern does NOT hold. The terminal value is then telling us")
        print("          about a property of the sample rather than about the constant in Eq. (2),")
        print("          and must be described that way rather than as a measurement of L_Psi.")

    out[tag] = {k: {str(n): dict(zip(("mean", "ci"), ci95(acc[k][n]))) for n in NFES}
                for _, k in rows}

dst = os.path.join(os.environ.get("WORK_DIR", "."), "gain_disentangle.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.finish()
print("Disentangling check done.")
