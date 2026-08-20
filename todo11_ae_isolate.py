#!/usr/bin/env python
# TODO 11 — ISOLATE THE SOURCE OF AUTOENCODER CONSTRAINT NON-PRESERVATION.
#
# The paper establishes that A = D o E is not constraint-preserving, but not WHICH part of the
# representation causes it. A literal "encoder-only BLE" is undefined (the latent is not a motion),
# so we decompose differently, with four probes:
#
#   P1  BASELINE           reproduce Table 2 rows on this clip set (sanity / comparability).
#   P2  DECODER-IMAGE      the decisive one. Freeze the decoder and OPTIMIZE the latent z directly
#                          to minimize bone-length error of D(z), starting from z = E(P(X)).
#                          - If BLE -> ~0, the decoder's image DOES contain (near-)valid motion and
#                            the failure is REACHABILITY (the encoder / the ODE never lands there).
#                          - If BLE plateaus well above 0, the decoder's image genuinely lacks valid
#                            motion -> REPRESENTATIONAL, as currently claimed.
#                          We also track drift from the target motion, so a "cheating" solution
#                          (valid bones, wrong motion) is visible rather than counted as success.
#   P3  TEMPORAL-COMPRESSION CONTROL
#                          The encoder downsamples x4 in time. Apply a pure 4-frame temporal box
#                          filter to the REAL joint trajectory (no autoencoder at all) and measure
#                          BLE. If this alone reproduces ~0.006, temporal compression explains most
#                          of the effect and the claim should be narrowed accordingly.
#   P4  PER-BONE BREAKDOWN which bones lose validity (distal vs proximal), for the discussion.
#
# Inference-only apart from P2, which optimizes a latent tensor (no model weights are updated).
#   sbatch run_todo11.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[todo11] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[todo11] models loaded.")

DEVICE = M.DEVICE; rvq = M.rvq; MAXLEN = M.MAX_MOTION_LEN
EDGES = M.EDGES; rest_len = M.rest_len; lengths_to_mask = M.lengths_to_mask
_gj = M._gj; project_joints = M.project_joints; _joints_to_norm = M._joints_to_norm; pad_norm = M.pad_norm
EI = torch.tensor([e[0] for e in EDGES], device=DEVICE)
EJ = torch.tensor([e[1] for e in EDGES], device=DEVICE)

N     = int(os.environ.get("T11_N", "256"))       # clips for P1/P3/P4
N_OPT = int(os.environ.get("T11_NOPT", "64"))     # clips for the latent optimization
STEPS = int(os.environ.get("T11_STEPS", "400"))   # optimization steps
LR    = float(os.environ.get("T11_LR", "0.02"))

def bone_dev(J):                      # (B,T,E) signed deviation from rest length
    return (J[:, :, EI, :] - J[:, :, EJ, :]).norm(dim=-1) - rest_len

def ble_per_clip(J, L):
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    return (bone_dev(J).abs() * fm).sum(dim=(1, 2)) / (fm.sum(dim=(1, 2)) * len(EDGES) + 1e-8)

def ble_per_bone(J, L):               # (E,) mean |deviation| per bone
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    return (bone_dev(J).abs() * fm).sum(dim=(0, 1)) / (fm.sum() + 1e-8)

def mpjpe(Ja, Jb, L):
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    return float(((Ja - Jb).norm(dim=-1) * fm).sum() / (fm.sum() * Ja.shape[2] + 1e-8))

def load_batch(ids):
    mn = torch.stack([torch.tensor(pad_norm(M.test_entries[i]["motion"])[0], device=DEVICE) for i in ids])
    L = torch.tensor([min(len(M.test_entries[i]["motion"]), MAXLEN) for i in ids], device=DEVICE)
    return mn, L

import random as _r; _r.seed(0)
idx = _r.sample(range(len(M.test_entries)), min(N, len(M.test_entries)))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "todo11_ae_isolate"), job_type="diagnostic")

out = {}

# ----------------------------------------------------------------- P1 + P3 + P4
real_b, rec_b, proj_b, rt_b, smooth_b = [], [], [], [], []
perbone_rt = torch.zeros(len(EDGES), device=DEVICE); nb = 0
with torch.no_grad():
    for s in range(0, len(idx), 64):
        mn, L = load_batch(idx[s:s + 64])
        Jr = _gj(mn); real_b.append(ble_per_clip(Jr, L).cpu())
        rec = rvq.decoder(rvq.encoder(mn)); rec_b.append(ble_per_clip(_gj(rec), L).cpu())
        mnp = _joints_to_norm(project_joints(Jr, L), mn)
        proj_b.append(ble_per_clip(_gj(mnp), L).cpu())
        recp = rvq.decoder(rvq.encoder(mnp)); Jrt = _gj(recp)
        rt_b.append(ble_per_clip(Jrt, L).cpu())
        perbone_rt += ble_per_bone(Jrt, L) * len(mn); nb += len(mn)
        # P3: x4 temporal box filter on the PROJECTED (BLE=0) joint trajectory, no autoencoder
        Jp = project_joints(Jr, L)
        Js = torch.nn.functional.avg_pool1d(
            Jp.permute(0, 2, 3, 1).reshape(Jp.shape[0], -1, MAXLEN), kernel_size=4, stride=1, padding=0
        )
        Js = torch.nn.functional.pad(Js, (0, 3), mode="replicate").reshape(Jp.shape[0], 22, 3, MAXLEN).permute(0, 3, 1, 2)
        smooth_b.append(ble_per_clip(Js, L).cpu())

cat = lambda x: torch.cat(x).numpy()
rows = [("real motion X", cat(real_b)),
        ("autoencoder round trip A(X)", cat(rec_b)),
        ("projected motion P(X)", cat(proj_b)),
        ("projected + round trip A(P(X))", cat(rt_b)),
        ("P(X) + 4-frame temporal box filter (no AE)", cat(smooth_b))]
print(f"\n{'='*90}\nP1/P3 — BLE by transformation (N={len(idx)})\n{'='*90}")
print(f" {'transformation':<44}{'mean':>11}{'p95':>11}{'max':>11}")
print("-" * 90)
for name, a in rows:
    print(f" {name:<44}{a.mean():>11.5f}{np.percentile(a,95):>11.5f}{a.max():>11.5f}")
    out[name] = dict(mean=float(a.mean()), p95=float(np.percentile(a, 95)), max=float(a.max()))
print("=" * 90)

sm, rt = cat(smooth_b).mean(), cat(rt_b).mean()
print(f"\nP3 verdict: temporal box filter alone gives BLE {sm:.5f} vs {rt:.5f} for the full round trip "
      f"({100*sm/max(rt,1e-9):.0f}% of the effect).")
print("  -> high share: temporal compression is the dominant mechanism; narrow the claim accordingly.")
print("  -> low share:  the loss is not merely temporal smoothing; the learned map itself is responsible.")

# P4
pb = (perbone_rt / max(nb, 1)).cpu().numpy()
order = np.argsort(-pb)
print(f"\nP4 — per-bone BLE after A(P(X)), worst 6 of {len(EDGES)}:")
for i in order[:6]:
    print(f"    bone {EDGES[i]}  |dev| = {pb[i]:.5f}")
print(f"    best bone {EDGES[order[-1]]} |dev| = {pb[order[-1]]:.5f}")
out["per_bone"] = {f"{EDGES[i][0]}-{EDGES[i][1]}": float(pb[i]) for i in range(len(EDGES))}

# ----------------------------------------------------------------- P2 decoder-image probe
print(f"\n{'='*90}\nP2 — DECODER-IMAGE PROBE: optimize z to minimize bone error of D(z)\n{'='*90}")
oid = idx[:N_OPT]
mn0, L0 = load_batch(oid)
with torch.no_grad():
    Jtgt = project_joints(_gj(mn0), L0)                    # BLE=0 target
    mnp0 = _joints_to_norm(Jtgt, mn0)
    z = rvq.encoder(mnp0).clone()
z = z.detach().requires_grad_(True)
opt = torch.optim.Adam([z], lr=LR)
fmask = lengths_to_mask(L0, MAXLEN).float().unsqueeze(-1)
traj = []
t0 = time.time()
for it in range(STEPS + 1):
    dec = rvq.decoder(z)
    J = M.recover_from_ric(dec * M.std_t + M.mean_t)
    loss = ((bone_dev(J) ** 2) * fmask).sum() / (fmask.sum() * len(EDGES) + 1e-8)
    if it % 25 == 0:
        with torch.no_grad():
            b = float(ble_per_clip(J, L0).mean()); d = mpjpe(J, Jtgt, L0)
        traj.append((it, b, d))
        print(f"   step {it:>4}  BLE={b:.5f}  MPJPE-to-target={d:.4f} m")
        wandb.log({"todo11/opt_step": it, "todo11/opt_BLE": b, "todo11/opt_MPJPE": d})
    if it == STEPS: break
    opt.zero_grad(); loss.backward(); opt.step()
print(f"   ({time.time()-t0:.0f}s)")

b0, bT = traj[0][1], traj[-1][1]
out["decoder_image_probe"] = dict(ble_start=b0, ble_end=bT, mpjpe_end=traj[-1][2],
                                  steps=STEPS, lr=LR, n_clips=N_OPT,
                                  trajectory=[[int(a), float(b), float(c)] for a, b, c in traj])
print(f"\nP2 verdict: BLE {b0:.5f} -> {bT:.5f} after {STEPS} steps of direct latent optimization,")
print(f"            with the decoded motion drifting {traj[-1][2]:.4f} m (MPJPE) from the target.")
if bT < 0.1 * max(b0, 1e-9) and bT < 0.001:
    print("  >>> The decoder's image DOES contain near-valid motion. The ceiling is REACHABILITY,")
    print("      not the decoder's expressive range. REVISE the Section 4.1 wording: the claim")
    print("      should be that the encode-decode map does not PRESERVE validity, not that valid")
    print("      motion is outside the decoder's range. (Also check the MPJPE drift above: if it is")
    print("      large, validity was bought by leaving the target motion.)")
else:
    print("  >>> Even direct optimization of the latent cannot drive BLE to 0 through the frozen")
    print("      decoder. This STRENGTHENS the representational reading: exact validity is not")
    print("      merely unreached by the encoder, it is largely outside the decoder's image.")
print("\nLIMITATION to state in the paper: an encoder-only measurement is not defined, because the")
print("latent is not a motion and carries no bone lengths. P2 is the closest available decomposition:")
print("it bounds what the decoder alone can express, holding the encoder out of the correction path.")

tbl = wandb.Table(columns=["transformation", "BLE_mean", "BLE_p95", "BLE_max"],
                  data=[[n, float(a.mean()), float(np.percentile(a, 95)), float(a.max())] for n, a in rows])
wandb.log({"todo11_table": tbl})
wandb.log({"todo11/opt_curve": wandb.plot.line(
    wandb.Table(data=[[a, b] for a, b, _ in traj], columns=["step", "BLE"]),
    "step", "BLE", title="latent optimization: BLE vs step")})
dst = os.path.join(os.environ.get("WORK_DIR", "."), "todo11_ae_isolate.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.finish()
print("TODO 11 diagnostic done.")
