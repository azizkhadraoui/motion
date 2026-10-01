#!/usr/bin/env python
# FID-optimisation experiments — the training half. Produces checkpoints that fidopt_eval.py consumes.
#
#   --task neon     Neon (Alemohammad, Wang & Baraniuk, ICLR 2026, arXiv:2510.03597), self-training half:
#                   draw NEON_NS samples from the base model with its own test-time sampler (CFG 2.5, 50
#                   Euler steps) on training prompts, then briefly fine-tune the base on them with the usual
#                   FM loss at reduced lr. Saves theta_s at several budgets. The merge
#                       theta_Neon = (1 + w) theta_r - w theta_s ,  w > 0
#                   is done at evaluation time (fidopt_eval.py), jointly searched with the CFG scale as the
#                   paper prescribes.
#   --task auto     Autoguidance (Karras et al., NeurIPS 2024, arXiv:2406.02507): train a SMALLER conditional
#                   model (hid 256, 4 layers) for a SHORT schedule on the same data and task; it is the guide.
#   --task dfm      Contrastive Flow Matching (Stoica et al., ICCV 2025, arXiv:2506.05350) as a fine-tune of
#                   the converged base:  L = |v - (x1 - x0)|^2 - lambda |v - (x1' - x0')|^2 , negatives from a
#                   different sample of the batch, lambda = 0.05.
#   --task ftctl    The CONTROL for dfm: identical fine-tune with lambda = 0. Without it, any dfm gain could
#                   be an artefact of simply training longer at a lower learning rate.
#
#   sbatch run_fidopt_train.sh <task> <base>      base in {latent, direct}

import os, sys, time, random, argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, null_cond, decode

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["neon", "auto", "dfm", "ftctl"], required=True)
ap.add_argument("--base", choices=["latent", "direct"], required=True)
A = ap.parse_args()
M = load_main(f"fidopt-train-{A.task}-{A.base}")
DEVICE = M.DEVICE; IS_LAT = A.base == "latent"; BS = 64
SAVE_AT = {"neon": [500, 1000, 2000, 4000], "auto": [5000, 10000, 20000], "dfm": [2500, 5000, 10000],
           "ftctl": [2500, 5000, 10000]}[A.task]
if os.environ.get("FO_SAVE_AT"): SAVE_AT = [int(x) for x in os.environ["FO_SAVE_AT"].split(",")]   # smoke tests
TAG = os.environ.get("FO_TAG", "")                                                                    # e.g. "_smoke"
STEPS = max(SAVE_AT)
LR = {"neon": 2e-5, "auto": 2e-4, "dfm": 2e-5, "ftctl": 2e-5}[A.task]
LAMBDA = float(os.environ.get("DFM_LAMBDA", "0.05")) if A.task == "dfm" else 0.0
NEON_NS = int(os.environ.get("NEON_NS", "16000" if IS_LAT else "8000"))
SMALL = dict(hid=256, layers=4, heads=4)
ckname = lambda step: os.path.join(M.CK, f"fidopt_{A.task}_{A.base}{TAG}_s{step}.pt")
random.seed(0); np.random.seed(0); torch.manual_seed(0)

# ------------------------------------------------------------------ base model and its latent statistics
base_net = M.load_net(A.base, IS_LAT)                    # restores the base's z-stats for latent
ZM = M.z_mean_t.detach().cpu().numpy() if IS_LAT else None; ZS = M.z_std_t.detach().cpu().numpy() if IS_LAT else None
cd = M.RVQ_CODE_DIM if IS_LAT else M.NFEATS; Tl = M.T_LAT if IS_LAT else M.MAX_MOTION_LEN
NTR = len(M.train_entries)

# ------------------------------------------------------------------ real training targets
if IS_LAT:
    print("[fidopt] encoding training latents with the base model's standardisation ...", flush=True)
    with torch.no_grad():
        Z = []
        for s in range(0, NTR, 128):
            b = [M.pad_norm(M.train_entries[i]["motion"])[0] for i in range(s, min(s + 128, NTR))]
            Z.append(((M.rvq.encoder(torch.tensor(np.stack(b), device=DEVICE)) - M.z_mean_t) / M.z_std_t).cpu())
        REAL = torch.cat(Z, 0)
def real_x1(idx):
    if IS_LAT: return REAL[idx].to(DEVICE)
    return torch.tensor(np.stack([M.pad_norm(M.train_entries[i]["motion"])[0] for i in idx]), device=DEVICE)

# ------------------------------------------------------------------ Neon: synthesise the self-training set
if A.task == "neon":
    syn_p = os.path.join(M.CK, f"fidopt_neon_{A.base}{TAG}_synthetic.pt")
    if os.path.exists(syn_p):
        d = torch.load(syn_p, weights_only=False); SYN_X, SYN_IDX = d["x"], d["idx"]
        print(f"[fidopt] reusing {len(SYN_IDX)} synthetic samples", flush=True)
    else:
        rng = np.random.default_rng(1); SYN_IDX = rng.integers(0, NTR, NEON_NS)
        X = []; t0 = time.time(); grid = M._timesteps(M.ODE_STEPS, "linear")
        with torch.no_grad():
            for s in range(0, NEON_NS, 64):
                idx = SYN_IDX[s:s + 64]; B = len(idx)
                L = torch.tensor([int(M.train_lens[i]) for i in idx], device=DEVICE)
                ts = torch.tensor(M.tr_seq[idx], device=DEVICE); tm = torch.tensor(M.tr_mask[idx], device=DEVICE)
                tp = torch.tensor(M.tr_pool[idx], device=DEVICE); ns, nm, npl = null_cond(base_net, B, M)
                torch.manual_seed(10_000_000 + s); z = torch.randn(B, Tl, cd, device=DEVICE)
                for i in range(M.ODE_STEPS):
                    tt = torch.full((B,), float(grid[i]), device=DEVICE)
                    z = z + float(grid[i + 1] - grid[i]) * M._cfg(base_net, z, tt, ts, tm, tp, L, ns, nm, npl, M.GUIDANCE, 0.0)
                X.append(z.cpu())
                if s % 2048 == 0: print(f"  synthesised {s + B}/{NEON_NS} ({(time.time()-t0)/60:.1f} min)", flush=True)
        SYN_X = torch.cat(X, 0); torch.save(dict(x=SYN_X, idx=SYN_IDX), syn_p)
        print(f"[fidopt] {NEON_NS} synthetic samples -> {syn_p}", flush=True)

# ------------------------------------------------------------------ the network being trained
if A.task == "auto":
    net = M.FMNet(cd, Tl, SMALL["hid"], SMALL["layers"], SMALL["heads"]).to(DEVICE)
    print(f"[fidopt] guide model: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params "
          f"(base {sum(p.numel() for p in base_net.parameters())/1e6:.1f}M)", flush=True)
else:
    net = base_net                                          # fine-tune starts from the base EMA weights
net.train()
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
ema = M.EMAh(net, M.EMA)


def batch():
    if A.task == "neon":
        j = np.random.randint(0, len(SYN_IDX), BS); idx = SYN_IDX[j]; x1 = SYN_X[j].to(DEVICE)
    else:
        idx = np.random.randint(0, NTR, BS); x1 = real_x1(idx)
    L = torch.tensor([int(M.train_lens[i]) for i in idx], device=DEVICE)
    ts = torch.tensor(M.tr_seq[idx], device=DEVICE); tm = torch.tensor(M.tr_mask[idx], device=DEVICE)
    tp = torch.tensor(M.tr_pool[idx], device=DEVICE)
    drop = torch.rand(BS, device=DEVICE) < M.CFG_DROP
    ts = torch.where(drop[:, None, None], net.null_seq.unsqueeze(0).expand(BS, -1, -1), ts)
    tp = torch.where(drop[:, None], net.null_pool.unsqueeze(0).expand(BS, -1), tp)
    return x1, ts, tm, tp, L


print(f"\n{'='*96}\nFIDOPT TRAIN task={A.task} base={A.base} steps={STEPS} lr={LR} lambda={LAMBDA} save at {SAVE_AT}\n{'='*96}", flush=True)
t0 = time.time(); run = []
for st in range(1, STEPS + 1):
    for pg in opt.param_groups: pg["lr"] = LR * min(1.0, st / 200)
    x1, ts, tm, tp, L = batch()
    x0 = torch.randn_like(x1); t = torch.rand(BS, device=DEVICE); T3 = t.view(-1, 1, 1)
    zt = (1 - T3) * x0 + T3 * x1; u = x1 - x0
    v = net(zt, t, ts, tm, tp, L)
    loss = F.mse_loss(v, u)
    if LAMBDA > 0:
        perm = torch.roll(torch.arange(BS, device=DEVICE), 1)          # a different sample of the batch
        loss = loss - LAMBDA * F.mse_loss(v, u[perm])
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), M.GRAD_CLIP); opt.step(); ema.update(net); run.append(loss.item())
    if st % 250 == 0:
        print(f"  [{A.task}/{A.base}] {st:>6} loss={np.mean(run):.4f} {(time.time()-t0)/60:.1f} min", flush=True); run = []
    if st in SAVE_AT:
        M.safe_save(dict(state={k: v_.clone() for k, v_ in ema.shadow.items()}, step=st, task=A.task, base=A.base,
                         lr=LR, lam=LAMBDA, arch=(SMALL if A.task == "auto" else None),
                         z_mean=ZM, z_std=ZS), ckname(st))
        print(f"  saved {ckname(st)}", flush=True)
print("=== FIDOPT TRAIN DONE ===")
