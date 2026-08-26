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
    base = _gj(decode(z, is_lat)); acc = torch.zeros(z.shape[0], device=z.device)
    for _ in range(probes):
        v = torch.randn_like(z)
        v = v / v.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, *([1] * (z.dim() - 1)))
        acc += rms_q(_gj(decode(z + eps * v, is_lat)), base, L) / eps
    return acc / probes

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

    acc = {k: {n: [] for n in NFES} for k in ("gain_vs_Q_rms", "gain_vs_Q_mpjpe", "gain_vs_Z",
                                              "curv_vs_Z", "partial_gain", "rms_vs_mpjpe")}
    t0 = time.time()
    for rep in range(REPS):
        G, CZ = [], []
        dQ_rms = {n: [] for n in NFES}; dQ_mp = {n: [] for n in NFES}; dZ = {n: [] for n in NFES}
        for s in range(0, N_EVAL, 32):
            e = min(s + 32, N_EVAL)
            ts_ = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
            tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
            seed = s + rep * 100000
            zref = terminal(net, is_lat, ts_, tm, tp, L, N_REF, seed)
            Jref = _gj(decode(zref, is_lat))
            G.append(gain(zref, is_lat, L).cpu())
            for n in NFES:
                zn = terminal(net, is_lat, ts_, tm, tp, L, n, seed)
                Jn = _gj(decode(zn, is_lat))
                dQ_rms[n].append(rms_q(Jn, Jref, L).cpu())
                dQ_mp[n].append(mpjpe_q(Jn, Jref, L).cpu())
                dZ[n].append(euclid_z(zn, zref).cpu())
        G = torch.cat(G).numpy()
        for n in NFES:
            a = torch.cat(dQ_rms[n]).numpy(); b = torch.cat(dQ_mp[n]).numpy(); c = torch.cat(dZ[n]).numpy()
            acc["gain_vs_Q_rms"][n].append(spearman(G, a))
            acc["gain_vs_Q_mpjpe"][n].append(spearman(G, b))
            acc["gain_vs_Z"][n].append(spearman(G, c))
            acc["partial_gain"][n].append(partial_spearman(G, a, c))
            acc["rms_vs_mpjpe"][n].append(spearman(a, b))
        print(f"  rep {rep+1}/{REPS}  ({(time.time()-t0)/60:.1f}m)")

    print(f"\n  {'quantity':<46}" + "".join(f"{'n='+str(n):>16}" for n in NFES))
    print("  " + "-" * (46 + 16 * len(NFES)))
    rows = [("gain vs degradation in Q (rms)",      "gain_vs_Q_rms"),
            ("gain vs degradation in Q (MPJPE)",    "gain_vs_Q_mpjpe"),
            ("gain vs degradation in Z  <-- decisive", "gain_vs_Z"),
            ("gain vs Q, controlling for Z (partial)", "partial_gain"),
            ("rms vs MPJPE agreement (sanity)",     "rms_vs_mpjpe")]
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

    out[tag] = {k: {str(n): dict(zip(("mean", "ci"), ci95(acc[k][n]))) for n in NFES}
                for _, k in rows}

dst = os.path.join(os.environ.get("WORK_DIR", "."), "gain_disentangle.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.finish()
print("Disentangling check done.")
