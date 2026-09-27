#!/usr/bin/env python
# 5.1 (MF-1) — MEANFLOW AND IMPROVED-MEANFLOW HEADS ON THE LATENT SPACE.
#
#   MeanFlow  (Geng, Deng, Bai, Kolter & He, arXiv:2505.13447)
#   iMF       (Geng, Lu, Wu, Shechtman, Kolter & He, arXiv:2512.02012)
#
# Warm start from the converged LFM checkpoint; the network gains a second time argument (interval
# length, zero-initialised embedding, so at step 0 the head reproduces the teacher's instantaneous
# field for every interval). Train the average-velocity objective. One-step sampling:
#       z1 = z0 + u(z0, 0, 1)            (this codebase: t = 0 noise, t = 1 data)
#
# THE IDENTITY, in this codebase's convention. u(z_t, t, s) = 1/(s-t) int_t^s v dtau for s >= t.
# Differentiating (s - t) u = int_t^s v with respect to the START time t along the trajectory:
#       u = v + (s - t) du/dt ,     du/dt = d_z u . v + d_t u     (a JVP with tangent (v, 1, 0))
# (MeanFlow's u = v - (t - r) du/dt is the same identity with time reversed.)
#
#   --variant mf : regress u onto the stop-gradient target  v~ + (s - t) du/dt,  JVP tangent v~
#                  (the conditional velocity, CFG-mixed). The target contains the network: MF's
#                  self-referential objective.
#   --variant imf: regress the INSTANTANEOUS velocity. The network parameterises u; the compound
#                  prediction  V = u - (s - t) sg(du/dt)  is regressed onto v~, with the JVP tangent
#                  taken along the network's own instantaneous velocity u(z, t, t). The regression
#                  target no longer depends on the network through du/dt. (Our reading of iMF's
#                  reformulation; iMF's in-context guidance conditioning is NOT reproduced -- CFG is
#                  baked in with a fixed omega, as for MF.)
#
# CFG (MeanFlow Sec. 4.2): v~ = omega * v + (1 - omega) * u(z_t, t, t | null), stop-gradient, for
# conditioned rows; v for the 10% condition-dropped rows. Sampled at guidance 1.0 afterwards.
# omega defaults to 2.5, the teacher's and the reflow pairs' guidance.
#
# JVP CHECK. torch.func.jvp through the FiLM/attention path is verified on a small batch against a
# central finite difference before training. If forward-mode AD is unsupported or disagrees
# (rel. err > 5e-2), training falls back to the finite-difference JVP and says so in the log/JSON.
#
# HONEST EXPECTATION. These are ImageNet-scale methods; on 13,840 clips with a short fine-tune, a
# 1-NFE FID competitive with reflow's 0.75 would be a good outcome and failure to converge is a
# plausible, reportable result. A poorly trained head is NOT evidence against MeanFlow. Stability is
# logged (loss, non-finite steps) because MF's self-referential target is what iMF exists to fix.
#
#   sbatch run_meanflow.sh mf     |     sbatch run_meanflow.sh imf
#   env: MF_STEPS (60000) MF_LR (1e-4) MF_BS (64) MF_OMEGA (2.5) MF_RATIO (0.25) MF_EVAL_EVERY (4000)

import os, sys, json, time, math, random, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_common import load_main, new_mfnet, mf_ckpt_path, null_cond, ClipSet, run_protocol, fmt, decode

ap = argparse.ArgumentParser(); ap.add_argument("--variant", choices=["mf", "imf"], default=os.environ.get("MF_VARIANT", "mf"))
VAR = ap.parse_args().variant
VTAG = VAR + os.environ.get("MF_SUFFIX", "")                  # file-name tag (smoke runs use a suffix)

M = load_main(f"meanflow-{VAR}")
DEVICE = M.DEVICE
STEPS = int(os.environ.get("MF_STEPS", "60000")); LR = float(os.environ.get("MF_LR", "1e-4"))
BS = int(os.environ.get("MF_BS", "64")); OMEGA = float(os.environ.get("MF_OMEGA", "2.5"))
RATIO = float(os.environ.get("MF_RATIO", "0.25"))           # fraction of samples with s != t
EVAL_EVERY = int(os.environ.get("MF_EVAL_EVERY", "4000")); WARM = 1000
ADAPT_P, ADAPT_C = 1.0, 1e-3                                  # MeanFlow adaptive loss weighting
WORK_DIR = os.environ.get("WORK_DIR", ".")
torch.backends.cuda.matmul.allow_tf32 = False                 # keep the JVP check meaningful

# ------------------------------------------------------------------ data: teacher latents
ck = torch.load(os.path.join(M.CK, "latent_best.pt"), map_location=DEVICE, weights_only=False)
M.z_mean_t = torch.tensor(ck["z_mean"], device=DEVICE).float(); M.z_std_t = torch.tensor(ck["z_std"], device=DEVICE).float()
print("[meanflow] encoding training latents with the teacher's standardisation ...", flush=True)
with torch.no_grad():
    Z = []
    for s in range(0, len(M.train_entries), 128):
        b = [M.pad_norm(M.train_entries[i]["motion"])[0] for i in range(s, min(s + 128, len(M.train_entries)))]
        Z.append(((M.rvq.encoder(torch.tensor(np.stack(b), device=DEVICE)) - M.z_mean_t) / M.z_std_t).cpu())
    TRAIN_Z = torch.cat(Z, 0)
print(f"[meanflow] {TRAIN_Z.shape[0]} latents {tuple(TRAIN_Z.shape[1:])}, per-dim std {TRAIN_Z.std().item():.3f}")

# ------------------------------------------------------------------ model (warm start)
net = new_mfnet(M)
missing, unexpected = net.load_state_dict(ck["state"], strict=False)
assert not unexpected and all(k.startswith("h_mlp") for k in missing), (missing, unexpected)
for m in net.modules():                                      # deterministic forward for the JVP
    if isinstance(m, nn.Dropout): m.p = 0.0
    if isinstance(m, nn.MultiheadAttention): m.dropout = 0.0
net.train()


def fwd(z, t, s, c):
    return net(z, t, s, c[0], c[1], c[2], c[3])


try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    MATH_SDPA = lambda: sdpa_kernel(SDPBackend.MATH)          # forward-mode AD needs the math kernel
except Exception:
    MATH_SDPA = lambda: torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


def jvp_func(z, t, s, c, tz):
    """du/dt along (tz, 1, 0) by forward-mode AD. Detached: both objectives stop its gradient."""
    with torch.no_grad(), MATH_SDPA():
        return torch.func.jvp(lambda a, b, d: fwd(a, b, d, c), (z, t, s), (tz, torch.ones_like(t), torch.zeros_like(s)))[1]


def jvp_fd(z, t, s, c, tz, eps=1e-3):
    """Central finite-difference fallback for the same derivative."""
    with torch.no_grad():
        return (fwd(z + eps * tz, t + eps, s, c) - fwd(z - eps * tz, t - eps, s, c)) / (2 * eps)


# ------------------------------------------------------------------ batches
order = list(range(len(M.train_entries)))
def batch():
    idx = random.sample(order, BS)
    L = torch.tensor([int(M.train_lens[i]) for i in idx], device=DEVICE)
    ts = torch.tensor(M.tr_seq[idx], device=DEVICE); tm = torch.tensor(M.tr_mask[idx], device=DEVICE)
    tp = torch.tensor(M.tr_pool[idx], device=DEVICE)
    drop = torch.rand(BS, device=DEVICE) < M.CFG_DROP
    ns, nm, npl = null_cond(net, BS, M)
    ts = torch.where(drop[:, None, None], ns, ts); tm = torch.where(drop[:, None], nm, tm)
    tp = torch.where(drop[:, None], npl, tp)
    return TRAIN_Z[idx].to(DEVICE), (ts, tm, tp, L), drop, (ns, nm, npl, L)


def sample_ts(B):
    a = torch.sigmoid(torch.randn(B, device=DEVICE) + 0.4); b = torch.sigmoid(torch.randn(B, device=DEVICE) + 0.4)
    t, s = torch.minimum(a, b), torch.maximum(a, b)           # logit-normal, mirrored to t=0 noise
    same = torch.rand(B, device=DEVICE) >= RATIO
    return t, torch.where(same, t, s)


# ------------------------------------------------------------------ JVP verification
z1, c, drop, cn = batch()
sub = slice(0, 8); cs = tuple(x[sub] for x in c)
z0 = torch.randn_like(z1[sub]); t, s = sample_ts(8); s = torch.maximum(s, t + 0.2).clamp(max=1.0)
zt = (1 - t)[:, None, None] * z0 + t[:, None, None] * z1[sub]; tz = z1[sub] - z0
JVP_MODE = "func"
try:
    # train mode on purpose: eval() routes nn.MultiheadAttention to the fused fast path
    # (_native_multi_head_attention), which has no forward-mode AD. Dropout is zeroed, so train mode
    # is deterministic, and it is the mode training itself uses.
    net.train()
    d_func = jvp_func(zt, t, s, cs, tz)
    d_fd = jvp_fd(zt, t, s, cs, tz)
    rel = float((d_func - d_fd).norm() / d_fd.norm().clamp_min(1e-12))
    print(f"[meanflow] JVP check: torch.func vs central difference, rel. err = {rel:.2e}  "
          f"(|du/dt| = {d_fd.norm().item():.3f})", flush=True)
    if not (rel < 5e-2): JVP_MODE = "fd"
except Exception as e:
    print(f"[meanflow] torch.func.jvp failed through this network ({type(e).__name__}: {str(e)[:200]})", flush=True)
    rel = float("nan"); JVP_MODE = "fd"
net.train()
if os.environ.get("MF_JVP") in ("func", "fd"):              # explicit override (see meanflow_jvp_diag.py)
    JVP_MODE = os.environ["MF_JVP"]; print(f"[meanflow] JVP mode forced by MF_JVP={JVP_MODE}")
print(f"[meanflow] JVP mode: {JVP_MODE}" + ("  (FALLBACK: finite difference)" if JVP_MODE == "fd" else ""), flush=True)
jvp = jvp_func if JVP_MODE == "func" else jvp_fd

# ------------------------------------------------------------------ training
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
ema = M.EMAh(net, M.EMA)
latest_p = os.path.join(M.CK, f"meanflow_{VTAG}_latest.pt"); best_p = mf_ckpt_path(M, VTAG)
st, best, log, nonfinite = 0, float("inf"), [], 0
diverging, DIVERGED = 0, False
if os.path.exists(latest_p):
    r = torch.load(latest_p, map_location=DEVICE, weights_only=False)
    net.load_state_dict(r["net"]); ema.shadow = {k: v.to(DEVICE) for k, v in r["ema"].items()}
    opt.load_state_dict(r["opt"]); st, best, log = int(r["step"]), float(r["best"]), r.get("log", [])
    print(f"[meanflow] resuming at step {st}")
zm = M.z_mean_t.cpu().numpy(); zsd = M.z_std_t.cpu().numpy()


def save_latest():
    M.safe_save(dict(net=net.state_dict(), ema={k: v.clone() for k, v in ema.shadow.items()}, opt=opt.state_dict(),
                     step=st, best=best, log=log, z_mean=zm, z_std=zsd), latest_p)


@torch.no_grad()
def quick_fid_1nfe(model, nfe=1):
    """1-NFE FID on the main script's 512-clip training-monitor subset (not the paper protocol)."""
    model.eval(); mf = []
    for a in range(0, len(M.sub_idx), 32):
        b = min(a + 32, len(M.sub_idx)); L = M.sub_len[a:b]
        ts = torch.tensor(M.sub_tseq[a:b], device=DEVICE); tm = torch.tensor(M.sub_tmask[a:b], device=DEVICE)
        tp = torch.tensor(M.sub_tpool[a:b], device=DEVICE); torch.manual_seed(a)
        z = torch.randn(b - a, M.T_LAT, M.RVQ_CODE_DIM, device=DEVICE); g = np.linspace(0, 1, nfe + 1)
        for i in range(nfe):
            z = z + float(g[i + 1] - g[i]) * model(z, torch.full((b - a,), float(g[i]), device=DEVICE),
                                                   torch.full((b - a,), float(g[i + 1]), device=DEVICE), ts, tm, tp, L)
        x = decode(M, z, True); gm = M.lengths_to_mask(L, M.MAX_MOTION_LEN)
        mf.append(M.memb(x * gm[..., None], L))
    return float(M.fid_calc(np.concatenate(mf, 0), M.sub_real_mf))


def eval_ema(nfe=1):
    bk = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(ema.shadow); f = quick_fid_1nfe(net, nfe); net.load_state_dict(bk); net.train()
    return f


print(f"\n{'='*96}\nMEANFLOW [{VAR}] {st} -> {STEPS} steps, lr {LR}, bs {BS}, omega {OMEGA}, s!=t ratio {RATIO}, "
      f"jvp {JVP_MODE}\n{'='*96}", flush=True)
if st == 0:
    f0 = eval_ema(1); log.append(dict(step=0, fid1=f0))
    print(f"  [warm start] 1-NFE FID = {f0:.4f}  (teacher field used as a 1-step map)", flush=True)
t0 = time.time(); run_loss, run_raw = [], []
while st < STEPS:
    z1, c, drop, cn = batch()
    for pg in opt.param_groups: pg["lr"] = LR * min(1.0, (st + 1) / WARM)
    z0 = torch.randn_like(z1); t, s = sample_ts(BS)
    T3 = lambda x: x[:, None, None]
    zt = (1 - T3(t)) * z0 + T3(t) * z1; v = z1 - z0
    with torch.no_grad():
        if OMEGA != 1.0:
            u_unc = net(zt, t, t, cn[0], cn[1], cn[2], cn[3])
            vt = torch.where(T3(drop), v, OMEGA * v + (1 - OMEGA) * u_unc)
        else:
            vt = v
    u = fwd(zt, t, s, c)                                   # the only forward that carries gradient
    if VAR == "mf":
        dudt = jvp(zt, t, s, c, vt)
        tgt = (vt + T3(s - t) * dudt).detach()
        raw = ((u - tgt) ** 2).mean((1, 2))
    else:
        with torch.no_grad():
            v_self = fwd(zt, t, t, c)                      # the network's own instantaneous velocity
        dudt = jvp(zt, t, s, c, v_self)
        V = u - T3(s - t) * dudt
        raw = ((V - vt) ** 2).mean((1, 2))
    w = 1.0 / (raw.detach() + ADAPT_C) ** ADAPT_P
    loss = (w * raw).mean()
    if not torch.isfinite(loss):
        nonfinite += 1; opt.zero_grad(set_to_none=True)
        if nonfinite >= 50:
            print(f"  [{VAR}] 50 non-finite losses by step {st}: training UNSTABLE, stopping.", flush=True); break
        continue
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), M.GRAD_CLIP); opt.step(); ema.update(net); st += 1
    run_loss.append(loss.item()); run_raw.append(raw.mean().item())
    if st % 250 == 0:
        print(f"  [{VAR}] {st:>6} loss={np.mean(run_loss):.4f} mse={np.mean(run_raw):.4f} |g|={float(gn):.2f} "
              f"nonfinite={nonfinite} {(time.time()-t0)/60:.1f}m", flush=True)
        log.append(dict(step=st, loss=float(np.mean(run_loss)), mse=float(np.mean(run_raw)), gnorm=float(gn)))
        # divergence guard: the adaptive weight keeps the weighted loss near 1 even while the raw error
        # explodes, so watch the RAW mse. 3 consecutive logs above 100x the first logged value -> stop.
        mse0 = next((r["mse"] for r in log if "mse" in r), None)
        diverging = diverging + 1 if (mse0 and np.mean(run_raw) > 100 * mse0) else 0
        run_loss, run_raw = [], []
        if diverging >= 3:
            DIVERGED = True
            print(f"  [{VAR}] raw MSE > 100x its initial value for 3 consecutive logs: training DIVERGED, stopping.", flush=True)
            break
    if st % 2000 == 0: save_latest()
    if st % EVAL_EVERY == 0 or st == STEPS:
        f1 = eval_ema(1); f2 = eval_ema(2); star = ""
        if f1 < best:
            best = f1; star = "  <-BEST"
            M.safe_save(dict(state={k: v.clone() for k, v in ema.shadow.items()}, step=st, metric=f1, variant=VAR,
                             omega=OMEGA, jvp=JVP_MODE, h_scale=net.H_SCALE, z_mean=zm, z_std=zsd), best_p)
        log.append(dict(step=st, fid1=f1, fid2=f2))
        print(f"    [{VAR} eval {st}] 1-NFE FID={f1:.4f}  2-NFE FID={f2:.4f}{star}", flush=True)
        save_latest()
save_latest()

# ------------------------------------------------------------------ final: paper protocol, replicated
print(f"\n{'='*96}\nFINAL [{VAR}] best checkpoint, 512-clip protocol, 3 reps, guidance baked in (1.0)\n{'='*96}", flush=True)
final = {}
if os.path.exists(best_p):
    bk = torch.load(best_p, map_location=DEVICE, weights_only=False); net.load_state_dict(bk["state"]); net.eval()
    clips = ClipSet(M, 512)
    for nfe in (1, 2, 4):
        def gen(a, b, ts, tm, tp, L, seed, nfe=nfe):
            torch.manual_seed(seed); z = torch.randn(b - a, M.T_LAT, M.RVQ_CODE_DIM, device=DEVICE); g = np.linspace(0, 1, nfe + 1)
            for i in range(nfe):
                z = z + float(g[i + 1] - g[i]) * net(z, torch.full((b - a,), float(g[i]), device=DEVICE),
                                                     torch.full((b - a,), float(g[i + 1]), device=DEVICE), ts, tm, tp, L)
            return decode(M, z, True), {}
        with torch.no_grad():
            final[nfe] = run_protocol(M, clips, gen, 3, label=f"{VAR} {nfe}-NFE (best step {bk['step']})")
    print("  references (earlier runs, 512-clip): reflow student 1-step FID 0.75; teacher 1-step 17.17, "
          "7.36 at s=2.5 as quoted in the plan.")
else:
    print("  no best checkpoint was written (training did not reach an evaluation).")

stable = nonfinite == 0
print(f"\nSTABILITY: {nonfinite} non-finite steps -> {'stable' if stable else 'UNSTABLE'}")
if VAR == "mf" and not stable:
    print("  MF's self-referential target is what iMF exists to fix; an unstable MF here is an independent")
    print("  confirmation of that motivation and worth one sentence in the paper.")
json.dump(dict(variant=VAR, steps=st, lr=LR, bs=BS, omega=OMEGA, ratio=RATIO, jvp_mode=JVP_MODE, jvp_rel_err=rel,
               nonfinite=nonfinite, diverged=DIVERGED, h_scale=net.H_SCALE, best_quick_fid1=best, log=log, final={str(k): v for k, v in final.items()}),
          open(os.path.join(WORK_DIR, f"meanflow_{VTAG}_train.json"), "w"), indent=2, default=float)
print(f"-> {os.path.join(WORK_DIR, f'meanflow_{VTAG}_train.json')}\nMeanFlow [{VAR}] done. Next: onestep_placement.py picks up {best_p}")
