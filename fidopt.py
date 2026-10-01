#!/usr/bin/env python
# FID-optimisation experiments for LFM and CDFM — shared sampler and protocol.
#
# Techniques (all from the recent flow-matching literature; see FIDOPT_README in the report):
#   cfg       standard classifier-free guidance, v = v_u + g (v_c - v_u)
#   cfgzero   CFG-Zero* (Fan et al., arXiv:2503.18886): optimised scale s* = <v_c, v_u>/|v_u|^2 per sample,
#             v = (1 - g) s* v_u + g v_c, and zero-init: no movement on the first K solver steps
#   apg       Adaptive Projected Guidance (Sadat et al., ICLR 2025, arXiv:2410.02416) applied, as the paper
#             prescribes, to the DENOISED prediction. For flow matching that is the endpoint estimate
#             x1 = z + (1 - t) v. diff = x1_c - x1_u, reverse momentum beta, optional norm cap r,
#             parallel part (w.r.t. x1_c) scaled by eta; x1_g = x1_c + (g - 1) diff; v = (x1_g - z)/(1 - t)
#   auto      Autoguidance (Karras et al., NeurIPS 2024, arXiv:2406.02507): v = v_g + w (v_m - v_g) with a
#             smaller, shorter-trained CONDITIONAL guide model g; optional extra CFG term (g_cfg - 1)(v_c - v_u)
# Weight-space techniques (Neon, contrastive FM) produce new checkpoints and are sampled with 'cfg'.
#
# PROTOCOL. Hyperparameters are selected on a SCREENING set of 512 test clips disjoint from the standard
# 512-clip REPORT set (ClipSet start=512). Selected configurations are then reported on the report set
# with 5 noise seeds, PAIRED against the baseline (CFG 2.5, 50 Euler steps) under identical noise, with the
# full metric panel (FID, SW, MMD, precision/recall/coverage, R@3) so that an FID gain obtained by
# narrowing the distribution is visible.

import os, json, time
import numpy as np, torch
from iclr_common import ClipSet, run_protocol, decode, null_cond, ci95, fmt

BASELINE = dict(mode="cfg", g=2.5)


def guided_velocity(M, net, z, t, ts, tm, tp, L, nc, cfg, state, guide=None):
    mode = cfg["mode"]; g = float(cfg.get("g", 2.5))
    v_c = net(z, t, ts, tm, tp, L)
    if mode == "auto":
        v_g = guide(z, t, ts, tm, tp, L)
        v = v_g + float(cfg["w"]) * (v_c - v_g)
        if cfg.get("g", 1.0) != 1.0:
            v = v + (g - 1.0) * (v_c - net(z, t, nc[0], nc[1], nc[2], L))
        return v
    if g == 1.0:
        return v_c
    v_u = net(z, t, nc[0], nc[1], nc[2], L)
    if mode == "cfg":
        return v_u + g * (v_c - v_u)
    if mode == "cfgzero":
        B = z.shape[0]; fc, fu = v_c.reshape(B, -1), v_u.reshape(B, -1)
        s = ((fc * fu).sum(1) / (fu * fu).sum(1).clamp_min(1e-8)).view(-1, 1, 1)
        return (1.0 - g) * s * v_u + g * v_c
    if mode == "apg":
        tt = t.view(-1, 1, 1)
        x_c = z + (1 - tt) * v_c; x_u = z + (1 - tt) * v_u
        diff = x_c - x_u
        beta = float(cfg.get("beta", 0.0))
        if beta != 0.0:
            state["buf"] = diff + beta * state["buf"] if "buf" in state else diff
            diff = state["buf"]
        r = cfg.get("r")
        B = z.shape[0]
        if r is not None:
            nrm = diff.reshape(B, -1).norm(dim=1).view(-1, 1, 1)
            diff = diff * torch.clamp(float(r) / nrm.clamp_min(1e-8), max=1.0)
        fc = x_c.reshape(B, -1); fd = diff.reshape(B, -1)
        par = ((fd * fc).sum(1) / (fc * fc).sum(1).clamp_min(1e-8)).view(-1, 1, 1) * x_c
        diff = (diff - par) + float(cfg.get("eta", 0.0)) * par
        x_g = x_c + (g - 1.0) * diff
        return (x_g - z) / (1 - tt)
    raise ValueError(mode)


@torch.no_grad()
def sample(M, net, is_lat, ts, tm, tp, L, seed, cfg, n=50, guide=None):
    torch.manual_seed(seed)
    B = tp.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=M.DEVICE)
    nc = null_cond(net, B, M)
    grid = M._timesteps(n, "linear"); K = int(cfg.get("zero_init", 0)); state = {}
    for i in range(n):
        t = float(grid[i]); dt = float(grid[i + 1] - grid[i])
        if i < K:
            continue                                   # CFG-Zero* zero-init: no movement on the first K steps
        v = guided_velocity(M, net, z, torch.full((B,), t, device=M.DEVICE), ts, tm, tp, L, nc, cfg, state, guide)
        z = z + dt * v
    return decode(M, z, is_lat)


def label(cfg):
    m = cfg["mode"]
    if m == "cfg": return f"CFG g={cfg['g']}"
    if m == "cfgzero": return f"CFG-Zero* g={cfg['g']} K={cfg.get('zero_init', 0)}"
    if m == "apg": return f"APG g={cfg['g']} eta={cfg.get('eta', 0)} beta={cfg.get('beta', 0)}" + (f" r={cfg['r']}" if cfg.get("r") else "")
    if m == "auto": return f"autoguide w={cfg['w']}" + (f" +CFG {cfg['g']}" if cfg.get("g", 1.0) != 1.0 else "")
    return str(cfg)


def make_gen(M, net, is_lat, cfg, guide=None, zstats=None):
    def gen(s, e, ts, tm, tp, L, seed):
        if zstats is not None: M.z_mean_t, M.z_std_t = zstats
        return sample(M, net, is_lat, ts, tm, tp, L, seed, cfg, guide=guide), {}
    return gen


def screen(M, clips, items, tag):
    """items: list of (name, gen). One replication on the SCREENING set (seed offset 900) -> FID per item."""
    out = {}
    for name, gen in items:
        r = run_protocol(M, clips, lambda s, e, a, b, c, L, sd, gen=gen: gen(s, e, a, b, c, L, sd + 900 * 100000), 1)
        out[name] = r["FID"][0]
        print(f"  [screen {tag}] {name:<52} FID={out[name]:.4f}  R@3={r['R3'][0]:.3f}  ({r['_secs']}s)", flush=True)
    return out


def report(M, clips, items, reps, tag, baseline_name):
    """items: list of (name, gen) including the baseline. 'reps' seeds on the REPORT set, full metrics,
    paired differences against the baseline."""
    res = {}
    for name, gen in items:
        res[name] = run_protocol(M, clips, gen, reps, label=f"{tag} | {name}", full_metrics=True)
    base = res[baseline_name]; rows = {}
    keys = ["FID", "SW", "MMD", "precision", "recall", "coverage", "R3"]
    print(f"\n  REPORT [{tag}] — {clips.n}-clip report set, {reps} seeds, paired vs '{baseline_name}' (negative dFID = better)")
    print(f"  {'configuration':<52}{'FID':>18}{'dFID (paired)':>22}{'dSW':>18}{'dPrec':>10}{'dRec':>10}{'dCov':>10}")
    for name, r in res.items():
        d = {k: ci95(np.array(r["_per_rep"][k]) - np.array(base["_per_rep"][k])) for k in keys}
        sig = d["FID"][1] == d["FID"][1] and d["FID"][0] + d["FID"][1] < 0
        rows[name] = dict(metrics={k: r[k] for k in keys + ["BLE", "FSR"]}, paired=d, per_rep=r["_per_rep"],
                          better_than_baseline=bool(sig))
        print(f"  {name:<52}{fmt(*r['FID']):>18}{fmt(*d['FID']):>22}{fmt(*d['SW'], p=4):>18}"
              f"{d['precision'][0]:>+10.3f}{d['recall'][0]:>+10.3f}{d['coverage'][0]:>+10.3f}" + ("   <- BETTER" if sig else ""), flush=True)
    return rows


def save(obj, name):
    p = os.path.join(os.environ.get("WORK_DIR", "."), name)
    json.dump(obj, open(p, "w"), indent=2, default=float); print(f"\nraw results -> {p}", flush=True)
