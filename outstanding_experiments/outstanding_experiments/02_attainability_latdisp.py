#!/usr/bin/env python
# TODO 13, PRIORITY 1 — DECODER ATTAINABILITY, RESOLVED PROPERLY.
#
# The revised TODO 11 asks whether the decoder can approach the constraint-valid set substantially
# more closely than the latents the encoder produces. A single optimization run cannot answer that:
# a plateau might mean the decoder's image lacks valid motion, or that the optimizer stalled, or
# that valid motion exists but not near this initialization. The paper's own three-way distinction
# (guidability / exact attainability / constraint preservation) requires telling these apart, so
# this script runs four probes designed to separate them.
#
#   P1  OPTIMIZER CONTROL, NO DECODER. Optimize the 263-d motion x directly under the same objective
#       and optimizer. There is no decoder in the way, so BLE must fall essentially to zero. If it
#       does not, the optimizer is the bottleneck and NOTHING else in this script can be interpreted.
#       This is the control that makes a plateau in P2 meaningful.
#
#   P2  FREE ATTAINABILITY. Optimize z to minimize bone error of D(z), from three initializations:
#       the encoder latent E(P(X)), that latent plus noise, and a random latent. No penalty on how
#       far the decoded motion drifts. This asks the EXISTENCE question: is there ANY latent whose
#       decoding is (near-)bone-valid? The best value across inits is an upper bound on the
#       attainable BLE, so a high plateau reached from all three inits is real evidence, whereas a
#       low value from any one init settles attainability affirmatively.
#
#   P3  LOCAL ATTAINABILITY. The same optimization with a drift penalty lambda*MPJPE^2 pulling the
#       decoded motion toward the target. This asks whether valid motion exists NEAR the motion we
#       actually wanted, rather than somewhere else in the decoder's range. A large gap between P2
#       and P3 means validity is attainable but only by leaving the intended motion -- which is a
#       reachability statement, not an expressiveness one.
#
#   P4  RECONSTRUCTION TARGET. Optimize z to minimize MPJPE to the bone-valid target, ignoring BLE
#       entirely, then measure the BLE that results. This is the natural question the encoder is
#       trying to answer, and it gives the BLE the decoder produces when it is asked only to
#       reproduce a valid motion as accurately as it can.
#
# HOW TO READ THE OUTCOME, mapped onto the paper's vocabulary:
#   P2 reaches ~0            -> exact attainability holds; the mechanism is round-trip
#                               non-preservation and encoder reachability, NOT decoder range.
#                               Frame Section 4 accordingly (the draft already anticipates this).
#   P2 plateaus well above 0 -> the decoder's image genuinely does not contain bone-valid motion,
#      while P1 reaches ~0      and a stronger statement about the representation is available.
#   P2 low but P3 high       -> valid motion exists in the decoder's range but not near the target;
#                               this is locality, and should be described as such rather than as
#                               either attainability or preservation.
#
#   sbatch run_todo11_attain.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[attain] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[attain] models loaded.")

DEVICE = M.DEVICE; rvq = M.rvq; MAXLEN = M.MAX_MOTION_LEN
EDGES = M.EDGES; rest_len = M.rest_len; lengths_to_mask = M.lengths_to_mask
recover_from_ric = M.recover_from_ric; _gj = M._gj
project_joints = M.project_joints; _joints_to_norm = M._joints_to_norm; pad_norm = M.pad_norm
EI = torch.tensor([e[0] for e in EDGES], device=DEVICE)
EJ = torch.tensor([e[1] for e in EDGES], device=DEVICE)

N     = int(os.environ.get("AT_N", "64"))
ONLY  = os.environ.get("AT_ONLY", "").strip().lower()   # e.g. AT_ONLY=p1 to rerun just the control
STEPS = int(os.environ.get("AT_STEPS", "20000"))
LR    = float(os.environ.get("AT_LR", "0.02"))
LOG   = int(os.environ.get("AT_LOG", "500"))
LAMBDAS = [float(x) for x in os.environ.get("AT_LAMBDAS", "0.1,1.0,10.0").split(",")]

def bone_dev(J):
    return (J[:, :, EI, :] - J[:, :, EJ, :]).norm(dim=-1) - rest_len

def make_metrics(L, Jtgt):
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    nb = fm.sum() * len(EDGES) + 1e-8
    nj = fm.sum() * Jtgt.shape[2] + 1e-8
    def ble(J):   return float((bone_dev(J).abs() * fm).sum() / nb)
    def drift(J): return float(((J - Jtgt).norm(dim=-1) * fm).sum() / nj)
    def loss_terms(J):
        lb = ((bone_dev(J) ** 2) * fm).sum() / nb
        ld = (((J - Jtgt) ** 2).sum(-1) * fm).sum() / nj
        return lb, ld
    return ble, drift, loss_terms

# ---------------------------------------------------------------------------
# LATENT DISPLACEMENT (added for the local-reachability analysis)
#
# The probes optimise the UNSTANDARDIZED autoencoder latent: rvq.encoder returns the
# raw latent and dec_z applies rvq.decoder to it directly, so nothing in this script
# touches z_mean_t / z_std_t. Displacement is therefore measured in raw latent units
# by default. The sigma-normalised variant expresses it in the standardized units the
# flow model actually integrates, which is the quantity comparable to the ODE's own
# scale; it needs the checkpoint statistics, restored below when available.
Z_STD = None
try:
    M.load_net("latent", True)          # restores z_mean_t / z_std_t; inference only
    Z_STD = M.z_std_t
    print(f"[attain] z-stats restored; sigma-normalised displacement enabled "
          f"(mean sigma {float(Z_STD.mean()):.4f})")
except Exception as e:
    print(f"[attain] z-stats unavailable ({type(e).__name__}); reporting raw-unit "
          f"displacement only")

def lat_disp(p, ref):
    """Per-clip displacement of the optimised latent from a reference latent.

    p, ref : (B, T_lat, C) = (B, 49, 256).  Returns means over the batch.
      fro   ||z - z_ref||_F                      raw Frobenius, per clip
      rms   ||z - z_ref||_F / sqrt(D)            per-element RMS, D = 49*256 = 12544
      rel   ||z - z_ref||_F / ||z_ref||_F        dimensionless, relative
      sig   ||(z - z_ref)/sigma_Z||_F / sqrt(D)  per-element RMS in standardized units
    """
    with torch.no_grad():
        d = (p - ref).flatten(1)                      # (B, D)
        D = d.shape[1]
        fro = d.norm(dim=1)                           # (B,)
        rel = fro / (ref.flatten(1).norm(dim=1) + 1e-12)
        out = dict(fro=float(fro.mean()),
                   rms=float((fro / D ** 0.5).mean()),
                   rel=float(rel.mean()))
        if Z_STD is not None:
            ds = ((p - ref) / Z_STD).flatten(1)
            out["sig"] = float((ds.norm(dim=1) / D ** 0.5).mean())
        else:
            out["sig"] = float("nan")
        return out


import random as _r; _r.seed(0)
idx = _r.sample(range(len(M.test_entries)), min(N, len(M.test_entries)))
mn0 = torch.stack([torch.tensor(pad_norm(M.test_entries[i]["motion"])[0], device=DEVICE) for i in idx])
L = torch.tensor([min(len(M.test_entries[i]["motion"]), MAXLEN) for i in idx], device=DEVICE)
with torch.no_grad():
    Jtgt = project_joints(_gj(mn0), L)            # bone-valid target, BLE = 0
    mnp = _joints_to_norm(Jtgt, mn0)
    z_enc = rvq.encoder(mnp).clone()
    mn_rec = rvq.decoder(z_enc).clone()      # the autoencoder reconstruction: BLE ~ 0.0065
    ble_fn, drift_fn, terms = make_metrics(L, Jtgt)
    base_rt = ble_fn(_gj(mn_rec))
print(f"[attain] {len(idx)} clips. Encoder round-trip BLE on the projected target: {base_rt:.5f}")

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "todo11_attainability"), job_type="diagnostic")

def optimize(param, decode, label, lam=0.0, objective="ble", steps=STEPS, ref=None):
    """Adam + cosine decay on `param`; `decode` maps it to joints. Returns the trajectory."""
    p = param.detach().clone().requires_grad_(True)
    z0 = param.detach().clone()          # this probe's own starting point
    opt = torch.optim.Adam([p], lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR * 0.01)
    traj = []; t0 = time.time()
    for it in range(steps + 1):
        J = decode(p)
        lb, ld = terms(J)
        loss = ld if objective == "recon" else (lb + lam * ld)
        if it % LOG == 0 or it == steps:
            with torch.no_grad():
                de = lat_disp(p, ref) if ref is not None else None
                di = lat_disp(p, z0) if ref is not None else None
                traj.append((it, ble_fn(J), drift_fn(J),
                             de["fro"] if de else float("nan"),
                             de["rms"] if de else float("nan"),
                             de["rel"] if de else float("nan"),
                             de["sig"] if de else float("nan"),
                             di["rms"] if di else float("nan")))
        if it == steps: break
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    tail = [t[1] for t in traj[-5:]]
    rel = (max(tail) - min(tail)) / max(min(tail), 1e-12) if len(tail) > 1 else float("nan")
    conv = rel < 0.02
    L = traj[-1]
    dz = "" if ref is None else (f"   |dz|_F {L[3]:.2f}  rms {L[4]:.4f}  "
                                 f"rel {L[5]:.3f}  sigma {L[6]:.3f}")
    print(f"  {label:<44} BLE {traj[0][1]:.5f} -> {L[1]:.5f}   drift {L[2]:.4f} m{dz}   "
          f"{'converged' if conv else 'STILL DESCENDING'}  ({time.time()-t0:.0f}s)")
    wandb.log({f"attain/{label}/BLE": L[1], f"attain/{label}/drift": L[2],
               f"attain/{label}/dz_rms": L[4], f"attain/{label}/dz_sigma": L[6]})
    return dict(label=label, ble0=traj[0][1], ble=L[1], drift=L[2],
                dz_fro=L[3], dz_rms=L[4], dz_rel=L[5], dz_sigma=L[6], dz_rms_from_init=L[7],
                converged=bool(conv), tail_rel=float(rel),
                traj=[[int(t[0])] + [float(v) for v in t[1:]] for t in traj])

dec_z = lambda p: recover_from_ric(rvq.decoder(p) * M.std_t + M.mean_t)
dec_x = lambda p: recover_from_ric(p * M.std_t + M.mean_t)
rows = []

print(f"\n{'='*104}\nP1 — OPTIMIZER CONTROL (optimize the motion directly, no decoder)\n{'='*104}")
# The control must START where the latent probe starts, i.e. from the autoencoder reconstruction
# (BLE ~ 0.0065), not from the already-projected motion (BLE 0, nothing to descend). Same objective,
# same optimizer, same initial BLE -- the only difference is that no decoder sits in the path, so a
# plateau in P2 can be attributed to the decoder rather than to the optimization.
rows.append(optimize(mn_rec, dec_x, "P1 direct motion, no decoder", objective="ble"))
p1 = rows[-1]["ble"]
if p1 > 1e-4:
    print(f"\n  WARNING: the control did not reach ~0 (got {p1:.5f}). The optimizer, not the decoder,")
    print("  is the binding constraint. Raise AT_STEPS or AT_LR before interpreting anything below.")

if ONLY == "p1":
    r = rows[-1]
    print(f"\n  CONTROL RESULT: BLE {r['ble0']:.5f} -> {r['ble']:.5f}")
    print("  Reaching ~0 here, from the same starting BLE as the latent probe, establishes that the")
    print("  optimizer is not the binding constraint and that any latent-side plateau is the decoder.")
    json.dump(dict(control=r), open(os.path.join(os.environ.get("WORK_DIR", "."), "todo11_control.json"), "w"), indent=2)
    sys.exit(0)

print(f"\n{'='*104}\nP2 — FREE ATTAINABILITY (optimize z, bone error only, three initializations)\n{'='*104}")
torch.manual_seed(0)
inits = [("from encoder latent", z_enc),
         ("encoder latent + noise", z_enc + 0.5 * z_enc.std() * torch.randn_like(z_enc)),
         ("random latent", z_enc.mean() + z_enc.std() * torch.randn_like(z_enc))]
for nm, z0 in inits:
    rows.append(optimize(z0, dec_z, f"P2 {nm}", objective="ble", ref=z_enc))

print(f"\n{'='*104}\nP3 — LOCAL ATTAINABILITY (bone error + lambda * drift, from the encoder latent)\n{'='*104}")
for lam in LAMBDAS:
    rows.append(optimize(z_enc, dec_z, f"P3 lambda={lam}", lam=lam, objective="ble", ref=z_enc))

print(f"\n{'='*104}\nP4 — RECONSTRUCTION TARGET (optimize z to match the valid motion, BLE only measured)\n{'='*104}")
rows.append(optimize(z_enc, dec_z, "P4 minimize drift only", objective="recon", ref=z_enc))

print(f"\n{'='*104}")
print(f" {'probe':<44}{'final BLE':>12}{'drift (m)':>12}{'|dz|_F':>10}"
      f"{'dz rms':>10}{'dz rel':>9}{'dz sigma':>10}{'conv':>7}")
print("-" * 112)
# measured, not assumed: the round trip's own displacement from the target
with torch.no_grad():
    rt_drift = drift_fn(_gj(mn_rec))
print(f" {'encoder round trip (no optimization)':<44}{base_rt:>12.5f}{rt_drift:>12.4f}"
      f"{0.0:>10.2f}{0.0:>10.4f}{0.0:>9.3f}{0.0:>10.3f}{'--':>7}")
for r in rows:
    print(f" {r['label']:<44}{r['ble']:>12.5f}{r['drift']:>12.4f}"
          f"{r['dz_fro']:>10.2f}{r['dz_rms']:>10.4f}{r['dz_rel']:>9.3f}"
          f"{r['dz_sigma']:>10.3f}{str(r['converged'])[:5]:>7}")
print("=" * 104)

p2 = [r for r in rows if r["label"].startswith("P2")]
best_p2 = min(p2, key=lambda r: r["ble"]) if p2 else None
p3 = [r for r in rows if r["label"].startswith("P3")]
print("\nVERDICT")
if p1 > 1e-4:
    print("  Inconclusive: the no-decoder control did not converge. Rerun with more steps.")
elif best_p2 and best_p2["ble"] < 5e-4:
    print(f"  EXACT ATTAINABILITY HOLDS. The decoder's image contains motion with BLE {best_p2['ble']:.5f}")
    print(f"  ({best_p2['label']}), far below the {base_rt:.5f} the encoder produces. The paper should")
    print("  therefore frame the mechanism as round-trip non-preservation plus encoder reachability,")
    print("  and must not claim the valid set lies outside the decoder's range. Section 4 of the")
    print("  current draft is already written this way, so this confirms rather than changes it.")
    if p3:
        worst = max(p3, key=lambda r: r["ble"])
        if worst["ble"] > 10 * best_p2["ble"]:
            print(f"  Note the locality caveat: under a drift penalty the best BLE rises to "
                  f"{worst['ble']:.5f}, so validity is attainable but not while staying close to the")
            print("  intended motion. Worth one sentence -- it is the distinction between existence and")
            print("  usefulness of a valid latent.")
elif best_p2:
    print(f"  The decoder plateaus at BLE {best_p2['ble']:.5f} from every initialization while the")
    print(f"  no-decoder control reaches {p1:.5f}. That is evidence the decoder's image does not")
    print("  contain exactly valid motion, and a stronger claim about the representation is available.")
    print("  State the optimizer control explicitly when reporting it, since the claim depends on it.")
print(f"\n  Encoder gap: the encoder yields BLE {base_rt:.5f}; the best optimized latent yields "
      f"{best_p2['ble'] if best_p2 else float('nan'):.5f}.")
print("  That ratio is the quantity separating what the decoder CAN express from what the encoder")
print("  DOES produce, which is exactly what TODO 11 asks for.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "todo11_attainability_latdisp.json")
json.dump(dict(encoder_roundtrip=base_rt, encoder_roundtrip_drift=rt_drift,
               latent_shape=list(z_enc.shape), latent_is_standardized=False,
               steps=STEPS, lr=LR, n=len(idx), probes=rows),
          open(dst, "w"), indent=2)
print(f"\nraw results -> {dst}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for r in rows:
        if r["label"].startswith(("P1", "P2")):
            ax.plot([t[0] for t in r["traj"]], [t[1] for t in r["traj"]], lw=1.1, label=r["label"])
    ax.axhline(base_rt, color="0.4", ls="--", lw=0.9)
    ax.text(STEPS * 0.02, base_rt * 1.08, "encoder round trip", fontsize=6, color="0.4")
    ax.set_yscale("log"); ax.set_xlabel("optimization step"); ax.set_ylabel("mean BLE")
    ax.legend(fontsize=5.5, frameon=False)
    p = os.path.join(os.environ.get("WORK_DIR", "."), "fig_attainability.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"figure -> {p}")
    wandb.log({"attainability_fig": wandb.Image(p.replace(".pdf", ".png"))})
except Exception as e:
    print("figure failed:", e)
wandb.finish()
print("Attainability probe done.")
