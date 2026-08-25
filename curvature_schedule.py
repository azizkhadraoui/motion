#!/usr/bin/env python
# CURVATURE-MATCHED STEP SCHEDULES  (workshop experiment, phase B)
#
# Phase A asks whether curvature predicts few-step degradation. If it does, the immediate practical
# question is whether that knowledge buys anything, which is what turns the analysis into a method.
#
# THE CONSTRUCTION. Phase A records the arc-length profile of the decoded trajectory: how much motion
# space is traversed per unit of t. If quality loss comes from taking large steps through regions
# where the decoded path moves fastest, then steps should be placed to equalize ARC LENGTH rather
# than to equalize t. Given the cumulative decoded arc length s(t), we choose the grid
#
#       t_k  such that  s(t_k) = (k/n) * s(1),
#
# so every Euler step covers the same distance in the space where quality is measured. This is a
# one-dimensional reparameterization computed once from a small calibration batch, costs nothing at
# sampling time, and requires no retraining.
#
# WHY THIS IS NOT THE EARLIER NULL RESULT. The sampling-strategy ablation compared cosine and power
# timestep grids at 50 steps and found no improvement. At 50 steps the discretization error is
# already negligible, so no schedule can help. The regime that matters here is 2 to 8 steps, where
# discretization error dominates, and where the choice of grid is the difference between a usable
# sample and a broken one. The two experiments are asking different questions, and the earlier null
# is the reason we can be confident the baseline is not simply under-tuned.
#
# WHAT IS COMPARED, at each step budget: linear (the baseline grid), cosine, power, and the
# arc-length-matched grid derived from the decoded profile. For the latent model we additionally
# derive a grid from the LATENT arc-length profile, since which of the two works better is the
# actionable form of phase A's question about which curvature governs degradation.
#
#   sbatch run_curv_schedule.sh          # reads curvature_nfe.json if present, else calibrates

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[sched] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[sched] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
load_net = M.load_net; ble_pc_joints = M.ble_pc_joints

N_EVAL = int(os.environ.get("EVAL_N", "512"))
NFES   = [int(x) for x in os.environ.get("SC_NFE", "2,4,8,16").split(",")]
BASES  = os.environ.get("SC_BASES", "latent,direct").split(",")
CAL_N  = int(os.environ.get("SC_CAL", "64"))       # clips used to calibrate the profile
CAL_STEPS = int(os.environ.get("SC_CAL_STEPS", "50"))
PROFILE_JSON = os.environ.get("SC_PROFILE", os.path.join(os.environ.get("WORK_DIR", "."), "curvature_nfe.json"))

def decode(y, is_lat):
    return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_lat else y

def masked_rms(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    d = ((A - B) ** 2).sum(-1).mean(-1) if A.dim() == 4 else ((A - B) ** 2).mean(-1)
    return ((d * fm).sum(1) / fm.sum(1).clamp_min(1)).sqrt()

@torch.no_grad()
def calibrate(net, is_lat, tseq, tmask, tpool, L, steps=CAL_STEPS, seed=0):
    """Mean per-step arc length along a fine-grid trajectory, in decoded and latent space."""
    torch.manual_seed(seed)
    B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    ts = _timesteps(steps, "linear")
    dec_seg, lat_seg = [], []
    prev_j = _gj(decode(z, is_lat)); prev_z = z.clone()
    for i in range(steps):
        t = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        tt = torch.full((B,), t, device=DEVICE)
        v = _cfg(net, z, tt, tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        z = z + dt * v
        j = _gj(decode(z, is_lat))
        dec_seg.append(float(masked_rms(j, prev_j, L).mean()))
        lat_seg.append(float((z - prev_z).flatten(1).norm(dim=1).mean()))
        prev_j, prev_z = j, z.clone()
    return np.array(dec_seg), np.array(lat_seg)

def grid_from_profile(seg, n):
    """Timestep grid equalizing cumulative arc length; returns n+1 points in [0,1]."""
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = s / max(s[-1], 1e-12)
    t_fine = np.linspace(0.0, 1.0, len(s))
    targets = np.linspace(0.0, 1.0, n + 1)
    return np.interp(targets, s, t_fine).astype(np.float32)

@torch.no_grad()
def sample_grid(net, is_lat, tseq, tmask, tpool, L, grid, seed):
    torch.manual_seed(seed)
    B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    for i in range(len(grid) - 1):
        t = float(grid[i]); dt = float(grid[i + 1] - grid[i])
        tt = torch.full((B,), t, device=DEVICE)
        v = _cfg(net, z, tt, tseq, tmask, tpool, L, ns, nm, npl, GUID, 0.0)
        z = z + dt * v
    return decode(z, is_lat)

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "curvature_schedule"), job_type="ablation")

rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

prof_cache = {}
if os.path.exists(PROFILE_JSON):
    try:
        prof_cache = {k: np.array(v["arc_profile"]) for k, v in json.load(open(PROFILE_JSON)).items()
                      if "arc_profile" in v}
        print(f"[sched] loaded decoded arc profiles from {PROFILE_JSON}: {list(prof_cache)}")
    except Exception as e:
        print("[sched] could not read profile json:", e)

out = {}
for tag in BASES:
    is_lat = (tag in ("latent", "latent_pen", "reflow"))
    if not M._have(tag):
        print(f"[sched] {tag} missing, skipped"); continue
    net = load_net(tag, is_lat)

    ts_c = torch.tensor(TSEQ[:CAL_N], device=DEVICE); tm_c = torch.tensor(TMASK[:CAL_N], device=DEVICE)
    tp_c = torch.tensor(TPOOL[:CAL_N], device=DEVICE); L_c = lens_all[:CAL_N]
    dec_seg, lat_seg = calibrate(net, is_lat, ts_c, tm_c, tp_c, L_c)
    if tag in prof_cache and len(prof_cache[tag]) == len(dec_seg):
        dec_seg = prof_cache[tag]          # prefer phase A's profile, measured on more clips
        print(f"[sched] using phase-A decoded profile for {tag}")

    SCHEDULES = {"linear": None, "cosine": None, "power": None,
                 "arc-matched (decoded)": dec_seg}
    if is_lat: SCHEDULES["arc-matched (latent)"] = lat_seg

    print(f"\n{'='*104}\nSTEP SCHEDULES — {tag}, {N_EVAL} clips\n{'='*104}")
    res = {}
    real_mf = None
    for n in NFES:
        for name, seg in SCHEDULES.items():
            grid = _timesteps(n, name) if seg is None else grid_from_profile(seg, n)
            grid = np.asarray(grid, dtype=np.float32)
            mf, rmf, bl = [], [], []
            for s in range(0, N_EVAL, 32):
                e = min(s + 32, N_EVAL)
                ts_ = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
                tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
                gm = lengths_to_mask(L, MAXLEN)
                x = sample_grid(net, is_lat, ts_, tm, tp, L, grid, seed=s)
                mf.append(memb(x * gm[..., None], L)); bl.append(np.asarray(ble_pc_joints(_gj(x), L)))
                if real_mf is None:
                    rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
                    rmf.append(memb(rm * gm[..., None], L))
            if real_mf is None: real_mf = np.concatenate(rmf, 0)
            G = np.concatenate(mf, 0)
            res[(n, name)] = dict(FID=float(fid_calc(G, real_mf)), R3=float(rprec(G, real_mf)[3]),
                                  BLE=float(np.concatenate(bl).mean()))
            print(f"  n={n:<4} {name:<24} FID={res[(n,name)]['FID']:.4f}  R@3={res[(n,name)]['R3']:.4f}")
            wandb.log({f"sched/{tag}/{name}/FID": res[(n, name)]["FID"], "sched/n": n})

    print(f"\n  {'steps':<8}" + "".join(f"{k:>26}" for k in SCHEDULES))
    print("  " + "-" * (8 + 26 * len(SCHEDULES)))
    for n in NFES:
        print(f"  {n:<8}" + "".join(f"{res[(n,k)]['FID']:>26.4f}" for k in SCHEDULES))
    print("\n  improvement of arc-matched over linear:")
    for n in NFES:
        d = res[(n, "linear")]["FID"] - res[(n, "arc-matched (decoded)")]["FID"]
        print(f"    n={n:<4} {d:+.4f}" + ("   arc-matched better" if d > 0 else "   linear better"))
    out[tag] = {f"{n}|{k}": v for (n, k), v in res.items()}

print("\nREADING")
print("  If the arc-matched grid beats linear at 2-8 steps and the advantage shrinks as steps grow,")
print("  that is the schedule-side confirmation of phase A: the loss comes from stepping too coarsely")
print("  through the fast part of the decoded path, and reallocating steps recovers it at no cost.")
print("  For the latent model, whichever of the two arc-matched grids wins identifies the geometry")
print("  that actually governs the error, which is the actionable version of phase A's question.")
print("  Single seed per cell: screen here, then replicate the winning configuration before claiming.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "curvature_schedule.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.finish()
print("Schedule experiment done.")
