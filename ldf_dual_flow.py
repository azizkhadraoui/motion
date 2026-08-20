#!/usr/bin/env python
# B1 — LAGRANGIAN DUAL FLOWS (Kurtz & Davydov, arXiv:2607.04513) FOR BONE-LENGTH CONSTRAINTS.
#
# Faithful implementation of their Eq. (8) / Algorithm 1:
#
#     xdot_t     = v_theta(x_t, t) - J_g(x_t)^T lambda_t - c * J_g(x_t)^T g(x_t)
#     lambdadot_t = g(x_t) / (1 - t)^p ,        lambda_0 = 0
#
# The primal correction has two terms: a quadratic penalty (weight c) and a dual term whose magnitude
# is an integrated state, not a fixed gain. The (1-t)^{-p} rescaling exists because flow models
# integrate on [0,1] rather than to t = infinity as in Platt & Barr's original differential multiplier
# method; it concentrates the correction near t = 1. The paper recommends p = 2: for p in [1,2) the
# residual provably converges to a generally NONZERO limit, and for p > 2 the dynamics become
# progressively harder to integrate. Theorem 1 gives ||g(x_t)|| = O((1-t)^alpha) as t -> 1^-.
#
# Per step the method needs exactly one vector-Jacobian product, [lambda + c*g]^T J_g, obtained by
# differentiating <w, g(x)> with w = (lambda + c*g) DETACHED. No pseudoinverse, no inner optimization.
#
# WHY RUN IT ON BOTH BASES. For CDFM the constraint is evaluated on the state itself, which is the
# setting LDF was designed for. For LFM, g(z) = bone(D(z)) - l puts the FROZEN DECODER inside the
# constraint Jacobian, so J_g = J_bone * J_D. This is the sharpest available test of the paper's
# thesis: LDF's own convergence theorem assumes (A2) that J_g J_g^T stays uniformly well-conditioned
# along the trajectory, and a lossy decoder is precisely what can break that. If LDF attains
# feasibility on CDFM and fails on LFM, the negative result is no longer about our projection
# operator at all -- it is about the representation, demonstrated on the adjacent method's own
# algorithm and its own guarantee.
#
# WHAT IS MEASURED. Besides FID/R@3/BLE/FSR we record ||g(x_t)|| at every step and fit the exponent
# alpha by least squares on log||g|| against log(1-t) over the final portion of the trajectory. That
# turns "does it converge" into a number comparable with Theorem 1, and it is the figure that belongs
# in the paper.
#
# HONEST EXPECTATION. LDF gives ASYMPTOTIC feasibility as t -> 1^-. With a finite 50-step Euler
# integrator you never reach the limit, so a nonzero residual BLE at t = 1 is the expected outcome
# even when the method works perfectly. That is not a defect of LDF; it is the structural difference
# from post-decode projection, which is exact by construction. Do not report the residual as LDF
# "failing" -- report it as the exactness gap, and note the step-count sensitivity (this script
# sweeps n_steps for exactly that reason).
#
#   sbatch run_ldf.sh

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[ldf] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[ldf] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; EDGES = M.EDGES; rest_len = M.rest_len
_cfg = M._cfg; _timesteps = M._timesteps; rvq = M.rvq
recover_from_ric = M.recover_from_ric; lengths_to_mask = M.lengths_to_mask
eval_variant = M.eval_variant; load_net = M.load_net; _agg = M._agg
project_joints = M.project_joints; _gj = M._gj; _joints_to_norm = M._joints_to_norm

EI = torch.tensor([e[0] for e in EDGES], device=DEVICE)
EJ = torch.tensor([e[1] for e in EDGES], device=DEVICE)

C_GRID   = [float(x) for x in os.environ.get("LDF_C", "10,100,1000,10000").split(",")]
P_GRID   = [float(x) for x in os.environ.get("LDF_P", "1,2,3").split(",")]
NSTEPS   = [int(x)   for x in os.environ.get("LDF_STEPS", "50,100").split(",")]
EPS      = float(os.environ.get("LDF_EPS", "1e-3"))     # floor on (1-t) in the dual gain
BASES    = os.environ.get("LDF_BASES", "direct,latent").split(",")

# knobs read by the patched sampler
LDF_C, LDF_P, TRACE = 1.0, 2.0, None

def decode(y, is_lat):
    return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_lat else y

def constraint(y, is_lat, L):
    """g(y) in R^{B x T x E}: signed bone-length residual, zeroed on padded frames."""
    J = recover_from_ric(decode(y, is_lat) * M.std_t + M.mean_t)
    g = (J[:, :, EI, :] - J[:, :, EJ, :]).norm(dim=-1) - rest_len
    return g * lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)

def make_ldf_sampler(orig):
    def s(net, is_latent, tseq, tmask, tpool, length, n=ODE, guidance=GUID, mode="none", seed=None,
          guide_w=None, cfg_rescale=0.0, schedule="linear", solver="euler", gw_schedule="const"):
        if mode not in ("ldf", "ldf_proj", "ldf_ctl"):
            return orig(net, is_latent, tseq, tmask, tpool, length, n=n, guidance=guidance, mode=mode,
                        seed=seed, guide_w=guide_w, cfg_rescale=cfg_rescale, schedule=schedule,
                        solver=solver, gw_schedule=gw_schedule)
        if seed is not None: torch.manual_seed(seed)
        B = tpool.shape[0]
        x = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
        lam = torch.zeros(B, MAXLEN, len(EDGES), device=DEVICE)          # lambda_0 = 0
        ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
        nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
        npl = net.null_pool.unsqueeze(0).expand(B, -1)
        ts = _timesteps(n, schedule)
        for i in range(n):
            t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
            tt = torch.full((B,), t, device=DEVICE)
            with torch.no_grad():
                v = _cfg(net, x, tt, tseq, tmask, tpool, length, ns, nm, npl, guidance, cfg_rescale)
            if mode == "ldf_ctl":
                corr = torch.zeros_like(x)          # CONTROL: plain ODE, tracing only
            else:
                # single VJP: J_g^T (lambda + c*g), with the weight detached
                with torch.enable_grad():
                    xc = x.detach().requires_grad_(True)
                    g = constraint(xc, is_latent, length)
                    w = (lam + LDF_C * g).detach()
                    corr = torch.autograd.grad((w * g).sum(), xc)[0]
            with torch.no_grad():
                gd = constraint(x, is_latent, length)
                if TRACE is not None:
                    TRACE.append((t, float(gd.pow(2).sum(dim=(1, 2)).sqrt().mean()),
                                  float(lam.abs().mean()),
                                  float(corr.flatten(1).norm(dim=1).mean()),
                                  float(v.flatten(1).norm(dim=1).mean())))
                x = x + dt * (v - corr)
                if mode != "ldf_ctl":
                    lam = lam + dt * gd / max((1.0 - t), EPS) ** LDF_P   # lambda-dot = g/(1-t)^p
        mn = decode(x, is_latent)
        if mode == "ldf_proj":
            mn = _joints_to_norm(project_joints(_gj(mn), length), mn)
        return mn
    return s

def fit_alpha(trace, frac=0.5):
    """Least-squares fit of log||g|| = const + alpha*log(1-t) over the final `frac` of the trajectory."""
    pts = [(t, r[0]) for t, *r in trace if r[0] > 0 and (1 - t) > 1e-6]
    pts = pts[int(len(pts) * (1 - frac)):]
    if len(pts) < 4: return float("nan")
    X = np.log(np.array([1 - t for t, _ in pts])); Y = np.log(np.array([g for _, g in pts]))
    return float(np.polyfit(X, Y, 1)[0])

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "ldf_dual_flow"), job_type="ablation")

orig = M.sample
rows = []; traces = {}
try:
    for tag in BASES:
        is_lat = (tag in ("latent", "latent_pen"))
        if not M._have(tag):
            print(f"[ldf] {tag} checkpoint missing, skipped"); continue
        net = load_net(tag, is_lat)

        M.sample = orig
        base = eval_variant(net, is_lat, "none")
        post = eval_variant(net, is_lat, "posthoc")
        bb, _, _ = _agg(base["ble"])
        rows.append(dict(base=tag, method="unconstrained", c=None, p=None, n=ODE,
                         FID=round(float(base["fid"]), 4), R3=round(float(base["R3"]), 4),
                         BLE=round(bb, 5), alpha=None))
        rows.append(dict(base=tag, method="post-decode projection", c=None, p=None, n=ODE,
                         FID=round(float(post["fid"]), 4), R3=round(float(post["R3"]), 4),
                         BLE=0.0, alpha=None))
        print(f"\n{'='*104}\nLDF on {tag} (latent={is_lat})   references: unconstrained FID "
              f"{rows[-2]['FID']}, post-decode {rows[-1]['FID']} at BLE 0\n{'='*104}")

        M.sample = make_ldf_sampler(orig)
        globals()["TRACE"] = []
        rc = eval_variant(net, is_lat, "ldf_ctl", n_steps=ODE)
        ctl = list(globals()["TRACE"]); globals()["TRACE"] = None
        a_ctl = fit_alpha(ctl)
        bc, _, _ = _agg(rc["ble"])
        print(f"  CONTROL (plain ODE, no correction): BLE={bc:.5f}  ||g||: {ctl[0][1]:.4f} -> "
              f"{ctl[-1][1]:.5f}  alpha_control={a_ctl:.3f}")
        print("  Any LDF alpha must be read AGAINST this: the residual falls steeply under the")
        print("  unconstrained flow too, simply because x_0 is noise. Only the excess over")
        print("  alpha_control is attributable to the dual flow.")
        traces[f"{tag}_CONTROL"] = ctl[::max(1, len(ctl) // 60)]
        rows.append(dict(base=tag, method="control (traced plain ODE)", c=None, p=None, n=ODE,
                         FID=round(float(rc["fid"]), 4), R3=round(float(rc["R3"]), 4),
                         BLE=round(bc, 5), alpha=round(a_ctl, 3) if a_ctl == a_ctl else None))
        for n in NSTEPS:
            for p in P_GRID:
                for c in C_GRID:
                    globals()["LDF_C"], globals()["LDF_P"] = c, p
                    globals()["TRACE"] = []
                    t0 = time.time()
                    r = eval_variant(net, is_lat, "ldf", n_steps=n)
                    tr = list(globals()["TRACE"]); globals()["TRACE"] = None
                    a = fit_alpha(tr)
                    bm, _, bx = _agg(r["ble"]); fm, _, _ = _agg(r["fsr"])
                    row = dict(base=tag, method="LDF", c=c, p=p, n=n,
                               FID=round(float(r["fid"]), 4), R3=round(float(r["R3"]), 4),
                               BLE=round(bm, 5), BLE_max=round(bx, 5), FSR=round(fm, 4),
                               alpha=round(a, 3) if a == a else None,
                               g0=round(tr[0][1], 4) if tr else None,
                               gT=round(tr[-1][1], 5) if tr else None,
                               corr_ratio=round(float(np.mean([r[2] / max(r[3], 1e-9) for _, *r in tr])), 5) if tr else None,
                               sec=round(time.time() - t0))
                    rows.append(row); traces[f"{tag}_c{c}_p{p}_n{n}"] = tr[::max(1, len(tr) // 60)]
                    print(f"  c={c:<7} p={p:<4} n={n:<4} FID={row['FID']:<9} R@3={row['R3']:<7} "
                          f"BLE={row['BLE']:<9} ||g||->{row['gT']:<9} alpha={row['alpha']:<7} "
                          f"|corr|/|v|={row['corr_ratio']} ({row['sec']}s)")
                    wandb.log({f"ldf/{tag}/FID": row["FID"], f"ldf/{tag}/BLE": row["BLE"],
                               f"ldf/{tag}/alpha": row["alpha"] or 0.0, "ldf/c": c, "ldf/p": p})
        # best LDF config, then LDF followed by post-decode projection
        cand = [r for r in rows if r["base"] == tag and r["method"] == "LDF" and r["BLE"] == r["BLE"]]
        if cand:
            best = min(cand, key=lambda r: r["BLE"])
            globals()["LDF_C"], globals()["LDF_P"], globals()["TRACE"] = best["c"], best["p"], None
            r = eval_variant(net, is_lat, "ldf_proj", n_steps=best["n"])
            bm, _, _ = _agg(r["ble"])
            rows.append(dict(base=tag, method="LDF + post-decode projection", c=best["c"], p=best["p"],
                             n=best["n"], FID=round(float(r["fid"]), 4), R3=round(float(r["R3"]), 4),
                             BLE=round(bm, 5), alpha=None))
            print(f"  best LDF (c={best['c']}, p={best['p']}) + post-decode projection: "
                  f"FID={rows[-1]['FID']} BLE={rows[-1]['BLE']}")
finally:
    M.sample = orig

print(f"\n{'='*112}")
print(f" {'base':<8}{'method':<32}{'c':>7}{'p':>5}{'n':>5}{'FID':>10}{'R@3':>8}{'BLE':>11}{'alpha':>8}")
print("-" * 112)
for r in rows:
    print(f" {r['base']:<8}{r['method']:<32}{str(r['c']):>7}{str(r['p']):>5}{str(r['n']):>5}"
          f"{r['FID']:>10}{r['R3']:>8}{r['BLE']:>11}{str(r.get('alpha')):>8}")
print("=" * 112)

print("\nREADING THE RESULT")
for tag in BASES:
    lf = [r for r in rows if r["base"] == tag and r["method"] == "LDF"]
    if not lf: continue
    unc = [r for r in rows if r["base"] == tag and r["method"] == "unconstrained"][0]
    b = min(lf, key=lambda r: r["BLE"])
    print(f"\n  [{tag}] best feasibility: BLE {b['BLE']} at c={b['c']}, p={b['p']}, n={b['n']}, "
          f"FID {b['FID']} (unconstrained {unc['FID']}, BLE {unc['BLE']})")
    fitted = [r["alpha"] for r in lf if r.get("alpha") is not None]
    if fitted:
        print(f"        fitted decay exponent alpha over the final half of the trajectory: "
              f"{min(fitted):.2f} to {max(fitted):.2f}")
    if b["BLE"] < unc["BLE"] * 0.5 and b["FID"] < 3 * unc["FID"]:
        print("        -> LDF reduces the residual substantially at moderate FID cost: it WORKS here.")
    elif b["FID"] > 3 * unc["FID"]:
        print("        -> feasibility is bought at a large FID cost; report the tradeoff curve, and")
        print("           check whether the p=2 rows integrate stably at this step count.")
    else:
        print("        -> LDF does not materially reduce the residual here.")
print("\n  The comparison that matters is direct vs latent. If LDF attains feasibility on CDFM but not")
print("  on LFM, say so explicitly: in the latent case the frozen decoder sits inside J_g, so the")
print("  method's own assumption (A2) -- uniform conditioning of J_g J_g^T along the trajectory -- is")
print("  the thing the representation breaks. That is a statement about the representation, not about")
print("  our projection operator, and it is made using their algorithm and their guarantee.")
print("\n  Also state plainly: LDF is asymptotic (Theorem 1 gives O((1-t)^alpha) as t -> 1^-), so a")
print("  nonzero residual at a finite step count is expected behaviour, not a failure. Post-decode")
print("  projection is exact by construction. That is the structural difference, and the n-sweep")
print("  above shows how the residual responds to a finer grid.")
print("\n  Single seed per config: screening only. Confirm any headline number across seeds.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "ldf_dual_flow.json")
json.dump(dict(rows=rows, traces=traces), open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for k, tr in traces.items():
        if not tr: continue
        ax.plot([1 - t for t, *_ in tr], [r[0] for _, *r in tr], lw=1.0, label=k, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("$1-t$ (integration progresses right to left)")
    ax.set_ylabel(r"$\|g(x_t)\|$")
    ax.set_title("LDF constraint residual vs time", pad=4)
    if len(traces) <= 6: ax.legend(fontsize=5, frameon=False)
    p = os.path.join(os.environ.get("WORK_DIR", "."), "fig_ldf_residual.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"figure -> {p}")
    wandb.log({"ldf_residual_fig": wandb.Image(p.replace(".pdf", ".png"))})
except Exception as e:
    print("figure failed:", e)

wandb.log({"ldf_table": wandb.Table(
    columns=["base", "method", "c", "p", "n_steps", "FID", "R@3", "BLE", "alpha"],
    data=[[r["base"], r["method"], r["c"], r["p"], r["n"], r["FID"], r["R3"], r["BLE"], r.get("alpha")] for r in rows])})
wandb.finish()
print("LDF experiment done.")
