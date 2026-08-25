#!/usr/bin/env python
# COMPOSING MULTIPLE EXACT GEOMETRIC CONSTRAINTS ON GENERATED MOTION.
#
# Candidate contribution for a short, self-contained paper, disjoint from the constraint-placement
# study: that work asks WHERE a single exact correction should sit relative to learned
# transformations; this asks what happens when SEVERAL exact constraints must hold at once.
#
# The seed observation: applying the bone-length rescale after the foot-contact correction restores
# exact bone validity but reintroduces foot skate, and applying the foot correction last does the
# reverse (FSR 0.0000, BLE 0.00233). The two operators do not commute, and alternating them does not
# settle into a motion satisfying both. That is a statement about the constraint sets themselves.
#
# THREE CONSTRAINT SETS, all defined on Cartesian joint trajectories:
#   C_bone   every segment has its prescribed length at every frame        (non-convex)
#   C_foot   a foot in contact has zero horizontal displacement            (affine in positions)
#   C_gnd    no joint lies below the floor plane                           (convex)
#
# FOUR ALGORITHMS. Each is a standard way to find a point in an intersection of sets, and each makes
# a different assumption; comparing them is what turns the observation into an analysis.
#   SEQ      one pass in a fixed order. What the paper currently does.
#   POCS     cyclic projection, repeated. Converges to the intersection for convex sets; for
#            non-convex sets it can cycle, which is what we appear to be seeing.
#   DYKSTRA  cyclic projection with per-set correction terms. Unlike POCS it converges to the
#            projection ONTO the intersection for convex sets, so it removes the ordering bias.
#            If Dykstra also fails to satisfy both, ordering was not the problem and the evidence
#            points to the intersection being empty or nearly so.
#   AVG      simultaneous projection onto all sets, averaged. Converges to a minimizer of the sum of
#            squared distances to the sets even when the intersection IS empty, so it produces a
#            principled compromise point rather than an oscillation.
#
# WHAT WOULD MAKE THE PAPER. Any of these outcomes is reportable:
#   - Dykstra reaches both constraints  -> the failure was operator ordering, and there is a correct
#     algorithm for composing exact geometric constraints on generated motion. Positive result.
#   - No algorithm reaches both, and AVG settles at a nonzero compromise -> the two exact constraint
#     sets are (nearly) disjoint on generated motion, and we quantify the incompatibility with the
#     relaxation sweep below. Negative result, but a sharp and useful one.
#
# THE RELAXATION SWEEP gives the headline figure either way: applying the foot correction with a
# damping factor beta and the bone rescale exactly, for beta in [0,1], traces a Pareto front between
# residual foot skate and residual bone error. Its shape is the quantitative statement of how
# incompatible the two constraints are, and it tells a practitioner what is purchasable.
#
# All post-processing on decoded samples: no training, no sampler changes, and the compute is
# dominated by generating the samples once.
#
#   sbatch run_composition.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[compose] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[compose] models loaded.")

DEVICE = M.DEVICE; MAXLEN = M.MAX_MOTION_LEN; EVAL_N = M.EVAL_N
sample = M.sample; load_net = M.load_net; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; ble_pc_joints = M.ble_pc_joints; fsr_pc = M.fsr_pc
lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
project_foot = M.project_foot; project_bonelength = M.project_bonelength
_joints_to_norm = M._joints_to_norm

BASE = os.environ.get("COMP_BASE", "latent"); IS_LAT = (BASE in ("latent", "latent_pen"))
N_EVAL = int(os.environ.get("EVAL_N", "512"))
K_ITER = int(os.environ.get("COMP_ITERS", "12"))
BETAS = [float(x) for x in os.environ.get("COMP_BETAS", "0,0.25,0.5,0.75,0.9,1.0").split(",")]
USE_GND = os.environ.get("COMP_GND", "1") == "1"

def P_bone(J, L): return project_bonelength(J)
def P_foot(J, L): return project_foot(J, L)
def P_gnd(J, L):
    out = J.clone(); out[..., 1] = out[..., 1].clamp_min(0.0); return out

SETS = [("bone", P_bone), ("foot", P_foot)] + ([("gnd", P_gnd)] if USE_GND else [])

def penetration(J, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    return float(((-J[..., 1]).clamp_min(0.0) * fm.unsqueeze(-1)).sum() / (fm.sum() * J.shape[2] + 1e-8))

def displacement(J, J0, L):
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    return float(((J - J0).norm(dim=-1) * fm).sum() / (fm.sum() * J.shape[2] + 1e-8))

def residuals(J, L, J0):
    return dict(BLE=float(ble_pc_joints(J, L).mean()), FSR=float(fsr_pc(J, L).mean()),
                PEN=penetration(J, L), DISP=displacement(J, J0, L))

# ---------------------------------------------------------------- algorithms
def alg_seq(J0, L, order=("foot", "bone"), **kw):
    J = J0
    for nm in order:
        J = dict(SETS)[nm](J, L)
    return J, [residuals(J, L, J0)]

def alg_pocs(J0, L, iters=K_ITER, **kw):
    J = J0; hist = []
    for _ in range(iters):
        for nm, P in SETS: J = P(J, L)
        hist.append(residuals(J, L, J0))
    return J, hist

def alg_dykstra(J0, L, iters=K_ITER, **kw):
    """Cyclic projection with per-set correction terms; removes the ordering bias of POCS."""
    J = J0.clone(); corr = [torch.zeros_like(J0) for _ in SETS]; hist = []
    for _ in range(iters):
        for i, (nm, P) in enumerate(SETS):
            y = P(J + corr[i], L)
            corr[i] = J + corr[i] - y
            J = y
        hist.append(residuals(J, L, J0))
    return J, hist

def alg_avg(J0, L, iters=K_ITER, **kw):
    """Simultaneous (Cimmino-style) projection: converges to a compromise even if the sets are disjoint."""
    J = J0.clone(); hist = []
    for _ in range(iters):
        J = torch.stack([P(J, L) for _, P in SETS]).mean(0)
        hist.append(residuals(J, L, J0))
    return J, hist

def alg_relaxed(J0, L, beta=1.0, **kw):
    """Damped foot correction, exact bone rescale last. beta=0 is bone only, beta=1 is the paper's O2."""
    Jf = project_foot(J0, L)
    J = (1 - beta) * J0 + beta * Jf
    J = project_bonelength(J)
    return J, [residuals(J, L, J0)]

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", f"constraint_composition_{BASE}"), job_type="analysis")

net = load_net(BASE, IS_LAT)
rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

VARIANTS = [("SEQ foot->bone", alg_seq, dict(order=("foot", "bone"))),
            ("SEQ bone->foot", alg_seq, dict(order=("bone", "foot"))),
            ("POCS", alg_pocs, {}),
            ("DYKSTRA", alg_dykstra, {}),
            ("AVG simultaneous", alg_avg, {})] + \
           [(f"RELAXED beta={b}", alg_relaxed, dict(beta=b)) for b in BETAS]

acc = {v[0]: dict(res=[], hist=[], mf=[]) for v in VARIANTS}
acc["NONE"] = dict(res=[], hist=[], mf=[]); real_mf = []
print(f"\n{'='*100}\nCONSTRAINT COMPOSITION — base={BASE}, {N_EVAL} clips, sets={[s[0] for s in SETS]}\n{'='*100}")
t0 = time.time()
with torch.no_grad():
    for s in range(0, N_EVAL, 32):
        e = min(s + 32, N_EVAL)
        ts = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
        tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
        gm = lengths_to_mask(L, MAXLEN)
        x = sample(net, IS_LAT, ts, tm, tp, L, seed=s)
        J0 = _gj(x)
        acc["NONE"]["res"].append(residuals(J0, L, J0)); acc["NONE"]["mf"].append(memb(x * gm[..., None], L))
        for name, fn, kw in VARIANTS:
            J, hist = fn(J0, L, **kw)
            xn = _joints_to_norm(J, x)
            acc[name]["res"].append(residuals(_gj(xn), L, J0))
            acc[name]["hist"].append(hist)
            acc[name]["mf"].append(memb(xn * gm[..., None], L))
        rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
        real_mf.append(memb(rm * gm[..., None], L))
print(f"  done in {(time.time()-t0)/60:.1f}m")

R = np.concatenate(real_mf, 0)
def agg(name):
    d = acc[name]
    m = {k: float(np.mean([r[k] for r in d["res"]])) for k in ("BLE", "FSR", "PEN", "DISP")}
    m["FID"] = float(fid_calc(np.concatenate(d["mf"], 0), R))
    return m

rows = [("NONE", agg("NONE"))] + [(n, agg(n)) for n, _, _ in VARIANTS]
print(f"\n{'='*100}")
print(f" {'method':<24}{'FID':>10}{'BLE':>12}{'FSR':>11}{'PENETR':>11}{'displ (m)':>12}")
print("-" * 100)
for n, m in rows:
    print(f" {n:<24}{m['FID']:>10.4f}{m['BLE']:>12.5f}{m['FSR']:>11.5f}{m['PEN']:>11.5f}{m['DISP']:>12.5f}")
print("=" * 100)

# iteration histories for the cyclic algorithms
print(f"\nIteration behaviour (mean over clips), {K_ITER} sweeps:")
for name in ("POCS", "DYKSTRA", "AVG simultaneous"):
    H = acc[name]["hist"]
    per_it = [{k: float(np.mean([b[i][k] for b in H])) for k in ("BLE", "FSR")} for i in range(K_ITER)]
    print(f"  {name}")
    print("    BLE " + " ".join(f"{p['BLE']:.5f}" for p in per_it))
    print("    FSR " + " ".join(f"{p['FSR']:.5f}" for p in per_it))
    tail = per_it[-4:]
    osc_b = max(p["BLE"] for p in tail) - min(p["BLE"] for p in tail)
    osc_f = max(p["FSR"] for p in tail) - min(p["FSR"] for p in tail)
    both = per_it[-1]["BLE"] < 1e-6 and per_it[-1]["FSR"] < 1e-6
    print(f"    -> final BLE {per_it[-1]['BLE']:.5f}, FSR {per_it[-1]['FSR']:.5f}; "
          f"tail oscillation {osc_b:.5f} / {osc_f:.5f}"
          + ("   BOTH SATISFIED" if both else "   NOT both satisfied"))
    acc[name]["per_it"] = per_it

print("\nREADING")
dyk = agg("DYKSTRA")
if dyk["BLE"] < 1e-6 and dyk["FSR"] < 1e-6:
    print("  Dykstra satisfies both constraints where sequential projection does not. The failure was")
    print("  operator ordering, and the contribution is a correct algorithm for composing exact")
    print("  geometric constraints on generated motion. Report the FID and displacement cost.")
else:
    print("  No cyclic scheme satisfies both constraints. Since Dykstra removes the ordering bias that")
    print("  POCS suffers from, ordering is not the explanation, and the evidence points to the two")
    print("  exact sets being disjoint or nearly disjoint on generated motion. The relaxation sweep")
    print("  below then defines what is actually purchasable, and the averaged projection gives the")
    print("  principled compromise point.")
print("\n  Pareto front (relaxed foot correction, exact bone rescale):")
for b in BETAS:
    m = agg(f"RELAXED beta={b}")
    print(f"    beta={b:<5} FSR {m['FSR']:.5f}  BLE {m['BLE']:.5f}  FID {m['FID']:.4f}")
print("\n  Caveat for the writeup: the bone rescale is applied along current segment directions and is")
print("  not the Euclidean projection onto C_bone, so POCS/Dykstra guarantees do not transfer even")
print("  setting non-convexity aside. State this; it is a limitation, and it also suggests the")
print("  natural follow-up of deriving the true projection.")

out = dict(base=BASE, n=N_EVAL, sets=[s[0] for s in SETS],
           table={n: m for n, m in rows},
           iters={k: acc[k].get("per_it") for k in ("POCS", "DYKSTRA", "AVG simultaneous")})
dst = os.path.join(os.environ.get("WORK_DIR", "."), f"constraint_composition_{BASE}.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.1))
    for name, st in [("POCS", "o-"), ("DYKSTRA", "s--"), ("AVG simultaneous", "^:")]:
        p = acc[name].get("per_it") or []
        axs[0].plot(range(1, len(p) + 1), [q["BLE"] * 1e3 for q in p], st, ms=3, lw=1.1, label=name)
        axs[1].plot(range(1, len(p) + 1), [q["FSR"] * 1e3 for q in p], st, ms=3, lw=1.1)
    axs[0].set_xlabel("projection sweep"); axs[0].set_ylabel(r"BLE ($\times10^{-3}$)")
    axs[1].set_xlabel("projection sweep"); axs[1].set_ylabel(r"FSR ($\times10^{-3}$)")
    axs[0].legend(frameon=False, fontsize=6)
    p1 = os.path.join(os.environ.get("WORK_DIR", "."), "fig_composition_iters.pdf")
    fig.savefig(p1, bbox_inches="tight"); fig.savefig(p1.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(3.2, 2.2))
    pts = [(agg(f"RELAXED beta={b}")["FSR"] * 1e3, agg(f"RELAXED beta={b}")["BLE"] * 1e3, b) for b in BETAS]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#1f4e79", ms=3.5, lw=1.2)
    for fs, bl, b in pts: ax.annotate(f"{b}", (fs, bl), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(r"foot-skate residual ($\times10^{-3}$)")
    ax.set_ylabel(r"bone-length residual ($\times10^{-3}$)")
    ax.set_title("attainable pairs under damped composition", fontsize=8, pad=4)
    p2 = os.path.join(os.environ.get("WORK_DIR", "."), "fig_composition_pareto.pdf")
    fig2.savefig(p2, bbox_inches="tight"); fig2.savefig(p2.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"figures -> {p1}, {p2}")
    wandb.log({"composition_iters": wandb.Image(p1.replace(".pdf", ".png")),
               "composition_pareto": wandb.Image(p2.replace(".pdf", ".png"))})
except Exception as e:
    print("figure failed:", e)

wandb.log({"composition_table": wandb.Table(columns=["method", "FID", "BLE", "FSR", "PENETR", "displacement"],
                                            data=[[n, m["FID"], m["BLE"], m["FSR"], m["PEN"], m["DISP"]] for n, m in rows])})
wandb.finish()
print("Constraint composition analysis done.")
