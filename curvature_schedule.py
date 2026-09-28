#!/usr/bin/env python
# CURVATURE-MATCHED STEP SCHEDULES — v2 (BeNTo camera-ready, reviewers R1 + R2)
#
# WHY v1 USED THE WRONG STATISTIC. For explicit Euler the local truncation error of step k is
#       LTE_k ~ 1/2 h_k^2 ||Y''(t_k)||,
# so equalising error across steps needs h_k ~ ||Y''||^(-1/2). v1 equalised ARC LENGTH, i.e.
# h_k ~ ||Y'||^(-1): a different derivative and a different power. Arc length allocates steps by
# speed; the bound allocates them by difficulty. That is the mismatch both reviewers named.
#
# THE CONSTRUCTION. Error density rho(t) = ||Y''(t)||^(1/2); choose t_k with
#       int_0^{t_k} rho = (k/n) int_0^1 rho,
# using the same reparameterisation as v1 (grid_from_profile, unchanged; only the integrand changes).
#
# ESTIMATING Y'' FOR FREE. Along the 50-step reference trajectory the states are available, and the
# TOTAL derivative along the trajectory is the second difference
#       Y''(t_k) ~ (Y_{k+1} - 2 Y_k + Y_{k-1}) / h^2  ( = (v_k - v_{k-1}) / h in the solver space ),
# no extra network evaluations. VALIDATION (GITS-style, no extra NFEs either): the error of one big
# Euler step of size H = m*h from Y_k against the fine trajectory's Y_{k+m} is measured directly and
# compared with the prediction 1/2 H^2 ||Y''(t_k)|| (Spearman across steps and clips, and the ratio).
#
# ARMS
#   A      curvature in Z (the space the solver integrates), shared grid (mean profile)
#   B      curvature in Q (decoded joints, where quality is measured), shared grid
#   C      curvature in Q, PER-CLIP grid, calibrated on an INDEPENDENT noise draw of the same prompt
#          (a usable method: costs one 50-step calibration trajectory per prompt)
#   C-Z    per-clip grid from Z curvature (control: Z is regular, so per-clip should not matter there)
#   C-orc  per-clip Q grid calibrated on the SAME noise as the sample: an ORACLE upper bound, not a
#          method (it needs the 50-step trajectory of the very sample it then draws in n steps)
# baselines: linear, cosine, power, arc-matched (decoded), arc-matched (latent) — as in Table 4.
#
# PRE-COMMITTED READINGS (BeNTo camera-ready plan §3.5), decided per step budget with PAIRED 95% CIs
# over seeds (every schedule sees the same noise in a replication):
#   A > B, either beats cosine -> practical payoff; lead §7 with it
#   A > B, both lose to cosine -> internal comparison supports the thesis; stronger negative vs cosine
#   A ~ B                      -> curvature does not transfer to scheduling; report as a limitation
#   C > B                      -> the destroyed Q-regularity has a practical cost; links §5 to §7
#
# GO / NO-GO. SC_PROFILE_ONLY=1 calibrates, prints and plots ||Y''|| against t and exits. If the
# profile is nearly flat the curvature grid is nearly linear and the sweep would measure noise.
#
#   SC_PROFILE_ONLY=1 sbatch run_curv_schedule.sh      # 5-minute go/no-go
#   sbatch run_curv_schedule.sh                         # full sweep

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[sched] importing {MAIN} ...", flush=True)
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[sched] models loaded.", flush=True)

DEVICE = M.DEVICE; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm; load_net = M.load_net

N_EVAL = int(os.environ.get("EVAL_N", "512"))
NFES = [int(x) for x in os.environ.get("SC_NFE", "2,4,8,16").split(",")]
BASES = os.environ.get("SC_BASES", "latent,direct").split(",")
CAL_N = int(os.environ.get("SC_CAL", "128"))
REPS = int(os.environ.get("SC_REPS", "5"))
CAL_STEPS = int(os.environ.get("SC_CAL_STEPS", "50"))
VAL_M = int(os.environ.get("SC_VAL_M", "5"))                  # big step = VAL_M fine steps (validation)
PROFILE_ONLY = os.environ.get("SC_PROFILE_ONLY", "0") == "1"
WORK_DIR = os.environ.get("WORK_DIR", ".")
CAL_SEED_OFFSET = 7_777_777                                    # independent calibration noise for arm C

TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262, 19: 2.093}
def ci95(a):
    a = np.asarray(a, float); n = len(a)
    if n < 2: return float(a.mean()), float("nan")
    return float(a.mean()), float(TCRIT.get(n - 1, 2.093) * a.std(ddof=1) / math.sqrt(n))
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")

def decode(y, is_lat):
    return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_lat else y

def masked_rms(A, L):
    """Per-clip RMS of a joint-space tensor (B,T,J,3) over valid frames."""
    fm = lengths_to_mask(L, MAXLEN).float()
    d = (A ** 2).sum(-1).mean(-1)
    return ((d * fm).sum(1) / fm.sum(1).clamp_min(1)).sqrt()


@torch.no_grad()
def reference(net, is_lat, tseq, tmask, tpool, L, seeds, steps=CAL_STEPS):
    """Fine linear-grid trajectory; per clip: arc and curvature profiles in Z and Q, plus the
    GITS-style direct truncation-error validation. seeds: (B,) per-clip seeds."""
    B = tpool.shape[0]
    z = torch.stack([torch.randn(net.Tlen, net.cd, generator=torch.Generator(device="cpu").manual_seed(int(s)))
                     for s in seeds]).to(DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    ts = _timesteps(steps, "linear"); h = 1.0 / steps
    Z, Q, V = [z.clone()], [_gj(decode(z, is_lat))], []
    for i in range(steps):
        v = _cfg(net, z, torch.full((B,), float(ts[i]), device=DEVICE), tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        V.append(v); z = z + h * v
        Z.append(z.clone()); Q.append(_gj(decode(z, is_lat)))
    arcZ = torch.stack([(Z[k + 1] - Z[k]).flatten(1).norm(dim=1) for k in range(steps)], 1)      # (B,steps)
    arcQ = torch.stack([masked_rms(Q[k + 1] - Q[k], L) for k in range(steps)], 1)
    # curvature at interior nodes k = 1..steps-1 (total derivative along the trajectory)
    accZ = torch.stack([(Z[k + 1] - 2 * Z[k] + Z[k - 1]).flatten(1).norm(dim=1) / h ** 2 for k in range(1, steps)], 1)
    accQ = torch.stack([masked_rms(Q[k + 1] - 2 * Q[k] + Q[k - 1], L) / h ** 2 for k in range(1, steps)], 1)
    # validation: one Euler step of size H = VAL_M*h from node k vs the fine trajectory
    H = VAL_M * h; val = dict(pred_Z=[], meas_Z=[], pred_Q=[], meas_Q=[])
    for k in range(1, steps - VAL_M):
        zb = Z[k] + H * V[k]
        val["meas_Z"].append((zb - Z[k + VAL_M]).flatten(1).norm(dim=1)); val["pred_Z"].append(0.5 * H * H * accZ[:, k - 1])
        val["meas_Q"].append(masked_rms(_gj(decode(zb, is_lat)) - Q[k + VAL_M], L)); val["pred_Q"].append(0.5 * H * H * accQ[:, k - 1])
    val = {k: torch.stack(v, 1).cpu().numpy() for k, v in val.items()}
    to = lambda x: x.cpu().numpy()
    return dict(arcZ=to(arcZ), arcQ=to(arcQ), accZ=to(accZ), accQ=to(accQ), val=val)


def density_per_interval(acc):
    """acc: (..., steps-1) curvature at interior nodes -> (..., steps) error density rho = acc^(1/2)
    per fine interval (average of its end-node values; end intervals take their single node)."""
    rho = np.sqrt(np.maximum(acc, 0.0))
    left = np.concatenate([rho[..., :1], rho], -1); right = np.concatenate([rho, rho[..., -1:]], -1)
    return 0.5 * (left + right)


def grid_from_profile(seg, n):
    """Timestep grid equalizing the cumulative integral of a per-interval profile (unchanged from v1)."""
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = s / max(s[-1], 1e-12)
    t_fine = np.linspace(0.0, 1.0, len(s))
    targets = np.linspace(0.0, 1.0, n + 1)
    return np.interp(targets, s, t_fine).astype(np.float32)


@torch.no_grad()
def sample_grid(net, is_lat, tseq, tmask, tpool, L, grids, seed):
    """grids: (n+1,) shared grid or (B, n+1) per-clip grids. Per-clip steps batch naturally since the
    network already takes a per-sample t."""
    torch.manual_seed(seed)
    B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    G = torch.tensor(np.broadcast_to(grids, (B, grids.shape[-1])).copy(), device=DEVICE, dtype=torch.float32)
    for i in range(G.shape[1] - 1):
        v = _cfg(net, z, G[:, i], tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        z = z + (G[:, i + 1] - G[:, i]).view(-1, 1, 1) * v
    return decode(z, is_lat)


# ------------------------------------------------------------------ data
rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)
bt = lambda a, s, e: torch.tensor(a[s:e], device=DEVICE)

# The sampler seeds a batch with torch.manual_seed(s + 100000*rep) and draws randn(B,...). For the
# ORACLE arm the calibration trajectory must use exactly that noise, so it is drawn the same way.
@torch.no_grad()
def batch_noise(net, B, seed):
    torch.manual_seed(seed); return torch.randn(B, net.Tlen, net.cd, device=DEVICE)

@torch.no_grad()
def reference_from_noise(net, is_lat, ts_, tm, tp, L, z0):
    B = tp.shape[0]; ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE); npl = net.null_pool.unsqueeze(0).expand(B, -1)
    steps = CAL_STEPS; ts = _timesteps(steps, "linear"); h = 1.0 / steps; z = z0.clone()
    Z, Q = [z.clone()], [_gj(decode(z, is_lat))]
    for i in range(steps):
        z = z + h * _cfg(net, z, torch.full((B,), float(ts[i]), device=DEVICE), ts_, tm, tp, L, ns, nm, npl, GUID, 0.0)
        Z.append(z.clone()); Q.append(_gj(decode(z, is_lat)))
    accZ = torch.stack([(Z[k + 1] - 2 * Z[k] + Z[k - 1]).flatten(1).norm(dim=1) / h ** 2 for k in range(1, steps)], 1)
    accQ = torch.stack([masked_rms(Q[k + 1] - 2 * Q[k] + Q[k - 1], L) / h ** 2 for k in range(1, steps)], 1)
    return accZ.cpu().numpy(), accQ.cpu().numpy()


def regularity(acc):
    """Across-clip dispersion of the normalised curvature profile: mean over t of the sd across clips
    of (profile / its own mean). Low = clips share a profile shape (a shared grid is justified)."""
    p = acc / np.maximum(acc.mean(-1, keepdims=True), 1e-12)
    return float(p.std(0).mean())


out = {}; t_start = time.time()
for tag in BASES:
    is_lat = tag in ("latent", "latent_pen", "reflow")
    if not M._have(tag): print(f"[sched] {tag} missing, skipped"); continue
    net = load_net(tag, is_lat)
    print(f"\n{'='*110}\nCALIBRATION — {tag}: {CAL_N} clips, {CAL_STEPS}-step linear reference\n{'='*110}", flush=True)
    cal = {k: [] for k in ["arcZ", "arcQ", "accZ", "accQ"]}; vals = {k: [] for k in ["pred_Z", "meas_Z", "pred_Q", "meas_Q"]}
    for s in range(0, CAL_N, 32):
        e = min(s + 32, CAL_N)
        r = reference(net, is_lat, bt(TSEQ, s, e), bt(TMASK, s, e), bt(TPOOL, s, e), lens_all[s:e],
                      seeds=[CAL_SEED_OFFSET + i for i in range(s, e)])
        for k in cal: cal[k].append(r[k])
        for k in vals: vals[k].append(r["val"][k])
    cal = {k: np.concatenate(v, 0) for k, v in cal.items()}; vals = {k: np.concatenate(v, 0) for k, v in vals.items()}
    tnode = np.arange(1, CAL_STEPS) / CAL_STEPS
    mZ, mQ = cal["accZ"].mean(0), cal["accQ"].mean(0)

    # ---- go / no-go
    cv = lambda x: float(x.std() / max(x.mean(), 1e-12))
    rhoZ_shared = density_per_interval(mZ); rhoQ_shared = density_per_interval(mQ)
    dev = {}
    for name, seg in [("A (Z curvature)", rhoZ_shared), ("B (Q curvature)", rhoQ_shared)]:
        dev[name] = {n: float(np.abs(grid_from_profile(seg, n) - np.linspace(0, 1, n + 1)).max()) for n in NFES}
    regZ, regQ = regularity(cal["accZ"]), regularity(cal["accQ"])
    print(f"  ||Y''|| profile, Z: coefficient of variation over t = {cv(mZ):.3f}; peak at t = {tnode[mZ.argmax()]:.2f}")
    print(f"  ||Y''|| profile, Q: coefficient of variation over t = {cv(mQ):.3f}; peak at t = {tnode[mQ.argmax()]:.2f}")
    print(f"  across-clip dispersion of the normalised profile: Z {regZ:.3f}   Q {regQ:.3f}   (low = shared grid justified)")
    for name, d in dev.items():
        print(f"  max |t_k - k/n| of the {name} grid: " + "  ".join(f"n={n}:{v:.3f}" for n, v in d.items()))
    print("  profile (t, ||Z''||, ||Q''||) at 10 points: " +
          "  ".join(f"({tnode[i]:.2f},{mZ[i]:.3g},{mQ[i]:.3g})" for i in np.linspace(0, len(tnode) - 1, 10).astype(int)))
    flat = all(v < 0.02 for d in dev.values() for v in d.values())
    print("  GO/NO-GO: " + ("NO-GO — both curvature grids are within 0.02 of linear at every n; the sweep would measure noise."
                            if flat else "GO — the curvature grids differ materially from linear."), flush=True)
    # ---- validation of the finite-difference estimate
    vres = {}
    for sp in ["Z", "Q"]:
        p, m_ = vals[f"pred_{sp}"], vals[f"meas_{sp}"]
        vres[sp] = dict(spearman_all=spearman(p.ravel(), m_.ravel()),
                        spearman_over_t=spearman(p.mean(0), m_.mean(0)),
                        median_ratio=float(np.median(m_ / np.maximum(p, 1e-12))))
        print(f"  validation ({sp}): measured big-step error vs 1/2 H^2 ||Y''||: Spearman {vres[sp]['spearman_all']:.3f} "
              f"(all), {vres[sp]['spearman_over_t']:.3f} (profile over t); median measured/predicted {vres[sp]['median_ratio']:.2f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
        for a, y, sp in [(ax[0], cal["accZ"], "Z (integration)"), (ax[1], cal["accQ"], "Q (decoded)")]:
            q = np.percentile(y, [25, 50, 75], 0)
            a.fill_between(tnode, q[0], q[2], alpha=0.25, color="#2a78d6"); a.plot(tnode, y.mean(0), color="#2a78d6", lw=2, label="mean")
            a.plot(tnode, q[1], color="#2a78d6", lw=1, ls="--", label="median"); a.set_yscale("log")
            a.set_title(f"{tag}: ||Y''|| in {sp}", fontsize=8); a.set_xlabel("t"); a.legend(fontsize=6, frameon=False)
        fig.tight_layout(); pth = os.path.join(WORK_DIR, f"fig_curvature_profile_{tag}.png"); fig.savefig(pth, dpi=200); plt.close(fig)
        print(f"  figure -> {pth}")
    except Exception as ex:
        print("  figure failed:", ex)
    rec = dict(profile=dict(t=tnode.tolist(), accZ_mean=mZ.tolist(), accQ_mean=mQ.tolist(),
                            accZ_q=np.percentile(cal["accZ"], [25, 50, 75], 0).tolist(),
                            accQ_q=np.percentile(cal["accQ"], [25, 50, 75], 0).tolist()),
               cv=dict(Z=cv(mZ), Q=cv(mQ)), regularity=dict(Z=regZ, Q=regQ), grid_dev=dev, validation=vres, go=not flat)
    out[tag] = rec
    if PROFILE_ONLY or flat:
        continue

    # ---- schedules
    SHARED = {"linear": None, "cosine": None, "power": None,
              "arc-matched (decoded)": cal["arcQ"].mean(0),
              "A: curvature Z, shared": rhoZ_shared, "B: curvature Q, shared": rhoQ_shared}
    if is_lat: SHARED["arc-matched (latent)"] = cal["arcZ"].mean(0)
    PERCLIP = ["C: curvature Q, per-clip", "C-Z: curvature Z, per-clip (control)", "C-orc: curvature Q, per-clip ORACLE"]
    names = list(SHARED) + PERCLIP
    fids = {(n, k): [] for n in NFES for k in names}; r3s = {(n, k): [] for n in NFES for k in names}
    real_mf = None
    print(f"\n{'='*110}\nSCHEDULES — {tag}, {N_EVAL} clips x {REPS} seeds, NFE {NFES}\n{'='*110}", flush=True)
    for rep in range(REPS):
        mf = {(n, k): [] for n in NFES for k in names}; rmf = []
        for s in range(0, N_EVAL, 32):
            e = min(s + 32, N_EVAL); ts_, tm, tp, L = bt(TSEQ, s, e), bt(TMASK, s, e), bt(TPOOL, s, e), lens_all[s:e]
            gm = lengths_to_mask(L, MAXLEN); seed = s + rep * 100000
            # per-clip calibration: independent noise (method) and the sample's own noise (oracle)
            aZi, aQi = reference_from_noise(net, is_lat, ts_, tm, tp, L, batch_noise(net, e - s, seed + CAL_SEED_OFFSET))
            _, aQo = reference_from_noise(net, is_lat, ts_, tm, tp, L, batch_noise(net, e - s, seed))
            pc = {"C: curvature Q, per-clip": density_per_interval(aQi),
                  "C-Z: curvature Z, per-clip (control)": density_per_interval(aZi),
                  "C-orc: curvature Q, per-clip ORACLE": density_per_interval(aQo)}
            for n in NFES:
                for k in names:
                    if k in SHARED:
                        g = np.asarray(_timesteps(n, k) if SHARED[k] is None else grid_from_profile(SHARED[k], n), np.float32)
                    else:
                        g = np.stack([grid_from_profile(p, n) for p in pc[k]])
                    x = sample_grid(net, is_lat, ts_, tm, tp, L, g, seed)
                    mf[(n, k)].append(memb(x * gm[..., None], L))
            if real_mf is None:
                rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
                rmf.append(memb(rm * gm[..., None], L))
        if real_mf is None: real_mf = np.concatenate(rmf, 0)
        for key, v in mf.items():
            G = np.concatenate(v, 0); fids[key].append(float(fid_calc(G, real_mf))); r3s[key].append(float(rprec(G, real_mf)[3]))
        print(f"  rep {rep+1}/{REPS} done ({(time.time()-t_start)/60:.1f} min)", flush=True)

    res = {f"{n}|{k}": dict(FID=ci95(fids[(n, k)]), R3=float(np.mean(r3s[(n, k)])), per_seed=fids[(n, k)])
           for n in NFES for k in names}
    print(f"\n  FID, mean ± 95% CI over {REPS} seeds ({N_EVAL}-clip protocol)")
    print(f"  {'schedule':<40}" + "".join(f"{'n='+str(n):>18}" for n in NFES))
    for k in names:
        print(f"  {k:<40}" + "".join(f"{res[f'{n}|{k}']['FID'][0]:>11.3f}±{res[f'{n}|{k}']['FID'][1]:<6.3f}" for n in NFES))

    # ---- pre-committed readings, paired over seeds
    def pd(a, b, n):   # paired FID(a) - FID(b); negative = a better
        return ci95(np.array(fids[(n, a)]) - np.array(fids[(n, b)]))
    better = lambda d: d[0] + d[1] < 0 if d[1] == d[1] else d[0] < 0
    tied = lambda d: (d[0] - d[1] <= 0 <= d[0] + d[1]) if d[1] == d[1] else False
    A_, B_, C_, COS = "A: curvature Z, shared", "B: curvature Q, shared", "C: curvature Q, per-clip", "cosine"
    reading = {}
    print(f"\n  READING (paired 95% CIs over seeds; negative = first schedule better)")
    for n in NFES:
        dAB, dCB = pd(A_, B_, n), pd(C_, B_, n)
        dAc, dBc = pd(A_, COS, n), pd(B_, COS, n)
        dAl = pd(A_, "linear", n); dOB = pd("C-orc: curvature Q, per-clip ORACLE", B_, n)
        if better(dAB) and (better(dAc) or better(dBc)): verdict = "A > B, and a curvature grid beats cosine: PRACTICAL PAYOFF"
        elif better(dAB): verdict = "A > B, both lose to (or tie) cosine: internal comparison supports the thesis"
        elif tied(dAB): verdict = "A ~ B: curvature does not transfer to scheduling here"
        else: verdict = "B > A: the decoded-space profile schedules better (contrary to the thesis at this n)"
        cvd = ("C > B: destroyed Q-regularity has a practical cost (links §5 to §7)" if better(dCB)
               else "C ~ B: per-clip calibration does not help" if tied(dCB) else "C worse than B")
        reading[n] = dict(A_minus_B=dAB, C_minus_B=dCB, A_minus_cosine=dAc, B_minus_cosine=dBc, A_minus_linear=dAl,
                          oracle_minus_B=dOB, verdict_AB=verdict, verdict_CB=cvd)
        f = lambda d: f"{d[0]:+.3f}±{d[1]:.3f}"
        print(f"   n={n:<3} A−B {f(dAB)}  C−B {f(dCB)}  A−cos {f(dAc)}  B−cos {f(dBc)}  A−lin {f(dAl)}  oracle−B {f(dOB)}")
        print(f"         -> {verdict}\n         -> {cvd}")
    out[tag].update(results=res, reading={str(k): v for k, v in reading.items()}, schedules=names)
    json.dump(out, open(os.path.join(WORK_DIR, "curvature_schedule_v2.json"), "w"), indent=2, default=float)

dst = os.path.join(WORK_DIR, "curvature_profile_check.json" if PROFILE_ONLY else "curvature_schedule_v2.json")
json.dump(out, open(dst, "w"), indent=2, default=float); print(f"\nraw results -> {dst}")
print("  Attribution: this is the analytic equal-error allocation; GITS (Chen et al., ICML 2024) solves the")
print("  general dynamic program over a measured cost matrix. Cite them for the DP formulation.")
print("Schedule experiment v2 done.")
