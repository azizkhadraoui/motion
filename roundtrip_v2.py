#!/usr/bin/env python
# ITERATED ROUND-TRIP v2 — adds the control that Figure 1 currently lacks.
#
# As reported in Section 6.7, the two existing curves are nearly identical: re-projecting to BLE = 0
# before every round trip accumulates to ~0.024, plain iterated autoencoding to ~0.026. That
# similarity is the whole point of the figure, but as drawn it cannot distinguish two explanations:
#
#   (i)  each round trip specifically destroys the constraint correction (the paper's reading), or
#   (ii) repeated autoencoding degrades the motion generally, and bone-length error rises as a side
#        effect of that drift, with the projection largely irrelevant to the trend.
#
# Curve (c) below settles it: iterate the autoencoder on REAL, NEVER-PROJECTED motion. If (c) also
# climbs to ~0.024, the accumulation is generic reconstruction drift and Section 6.7 must be reworded
# -- the constraint-specific claim would then rest on the SINGLE round trip of Table 2, which is
# clean and sufficient, rather than on the accumulation. If (c) stays much lower, the accumulation is
# genuinely tied to the projection-round-trip cycle and the current wording stands.
#
# We also track MPJPE drift from the original motion, so the figure can show what is actually being
# lost, and report the per-round INCREMENT for curve (a), which is the quantity that corresponds to
# the in-ODE dose-response.
#
#   sbatch run_roundtrip_v2.sh

import os, sys, json
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[roundtrip2] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[roundtrip2] models loaded.")

DEVICE = M.DEVICE; rvq = M.rvq; MAXLEN = M.MAX_MOTION_LEN
EDGES = M.EDGES; rest_len = M.rest_len; lengths_to_mask = M.lengths_to_mask
_gj = M._gj; project_joints = M.project_joints; _joints_to_norm = M._joints_to_norm; pad_norm = M.pad_norm
EI = torch.tensor([e[0] for e in EDGES], device=DEVICE)
EJ = torch.tensor([e[1] for e in EDGES], device=DEVICE)

K = int(os.environ.get("RT_ROUNDS", "8")); N = int(os.environ.get("RT_N", "512"))

def ble_of_norm(mn, L):
    J = _gj(mn)
    bone = (J[:, :, EI, :] - J[:, :, EJ, :]).norm(dim=-1)
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    return ((bone - rest_len).abs() * fm).sum(dim=(1, 2)) / (fm.sum(dim=(1, 2)) * len(EDGES) + 1e-6)

def mpjpe(mn, ref_J, L):
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    return ((_gj(mn) - ref_J).norm(dim=-1) * fm).sum(dim=(1, 2)) / (fm.sum(dim=(1, 2)) * ref_J.shape[2] + 1e-6)

def project_norm(mn, L):
    return _joints_to_norm(project_joints(_gj(mn), L), mn)

import random as _r; _r.seed(0)
idx = _r.sample(range(len(M.test_entries)), min(N, len(M.test_entries)))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "iterated_roundtrip_v2"), job_type="diagnostic")

curves = {c: [[] for _ in range(K + 1)] for c in ("reproj", "plain", "control")}
drifts = {c: [[] for _ in range(K + 1)] for c in ("reproj", "plain", "control")}

with torch.no_grad():
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        mn0 = torch.stack([torch.tensor(pad_norm(M.test_entries[i]["motion"])[0], device=DEVICE) for i in b])
        L = torch.tensor([min(len(M.test_entries[i]["motion"]), MAXLEN) for i in b], device=DEVICE)
        ref_J = _gj(mn0)                                  # drift reference: the original real motion

        # (a) project, then repeat [encode -> decode -> project]; BLE measured after each decode
        cur = project_norm(mn0, L)
        curves["reproj"][0].append(ble_of_norm(cur, L).cpu()); drifts["reproj"][0].append(mpjpe(cur, ref_J, L).cpu())
        for k in range(1, K + 1):
            dec = rvq.decoder(rvq.encoder(cur))
            curves["reproj"][k].append(ble_of_norm(dec, L).cpu()); drifts["reproj"][k].append(mpjpe(dec, ref_J, L).cpu())
            cur = project_norm(dec, L)

        # (b) project once, then plain iterated autoencoding
        cur = project_norm(mn0, L)
        curves["plain"][0].append(ble_of_norm(cur, L).cpu()); drifts["plain"][0].append(mpjpe(cur, ref_J, L).cpu())
        for k in range(1, K + 1):
            cur = rvq.decoder(rvq.encoder(cur))
            curves["plain"][k].append(ble_of_norm(cur, L).cpu()); drifts["plain"][k].append(mpjpe(cur, ref_J, L).cpu())

        # (c) CONTROL: real motion, never projected, plain iterated autoencoding
        cur = mn0
        curves["control"][0].append(ble_of_norm(cur, L).cpu()); drifts["control"][0].append(mpjpe(cur, ref_J, L).cpu())
        for k in range(1, K + 1):
            cur = rvq.decoder(rvq.encoder(cur))
            curves["control"][k].append(ble_of_norm(cur, L).cpu()); drifts["control"][k].append(mpjpe(cur, ref_J, L).cpu())

mean = lambda rows: [float(torch.cat(r).mean()) for r in rows]
B = {c: mean(curves[c]) for c in curves}
D = {c: mean(drifts[c]) for c in drifts}

print(f"\n{'='*94}\nITERATED ROUND TRIP v2 (N={len(idx)} clips, {K} rounds)\n{'='*94}")
print(f" {'round':>6}{'(a) re-project':>18}{'(b) plain, projected':>22}{'(c) CONTROL real':>20}{'drift (a) m':>14}")
print("-" * 94)
for k in range(K + 1):
    print(f" {k:>6}{B['reproj'][k]:>18.5f}{B['plain'][k]:>22.5f}{B['control'][k]:>20.5f}{D['reproj'][k]:>14.4f}")
print("=" * 94)

ratio = B["control"][K] / max(B["reproj"][K], 1e-9)
print(f"\nControl reaches {100*ratio:.0f}% of the re-projected curve's final BLE.")
if ratio > 0.8:
    print(">>> The accumulation is essentially generic autoencoder drift: iterating A degrades motion")
    print("    and bone-length error rises with it, whether or not the constraint is ever imposed.")
    print("    REWORD Section 6.7: the figure shows that repeated correction cannot arrest the drift,")
    print("    NOT that the correction is what is being destroyed. The constraint-specific evidence is")
    print("    the SINGLE round trip in Table 2 (0 -> 0.00655), which is clean and already sufficient.")
else:
    print(">>> The control stays well below the projected curves: the accumulation is specific to the")
    print("    project/round-trip cycle rather than generic drift. The current Section 6.7 wording holds.")
print(f"\nMotion drift after {K} rounds: {D['reproj'][K]:.4f} m (a), {D['plain'][K]:.4f} m (b), {D['control'][K]:.4f} m (c).")
print("Report this alongside the BLE curve so a reader can see what the round trips cost overall.")

# ---- figure ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.0))
    r = range(K + 1)
    axs[0].plot(r, B["reproj"], "o-", color="#1f4e79", ms=3, lw=1.2, label="re-project to BLE=0 each round")
    axs[0].plot(r, B["plain"], "s--", color="#c0504d", ms=3, lw=1.2, label="projected once, then plain AE")
    axs[0].plot(r, B["control"], "^:", color="#7f7f7f", ms=3, lw=1.2, label="control: real motion, never projected")
    axs[0].axhline(0.00598, color="#2e8b57", lw=0.8, ls=":")
    axs[0].text(0.1, 0.0063, "single round trip on real motion", fontsize=6, color="#2e8b57")
    axs[0].set_xlabel("number of autoencoder round trips"); axs[0].set_ylabel("mean BLE")
    axs[0].legend(frameon=False, fontsize=6, loc="upper left")
    axs[1].plot(r, D["reproj"], "o-", color="#1f4e79", ms=3, lw=1.2)
    axs[1].plot(r, D["plain"], "s--", color="#c0504d", ms=3, lw=1.2)
    axs[1].plot(r, D["control"], "^:", color="#7f7f7f", ms=3, lw=1.2)
    axs[1].set_xlabel("number of autoencoder round trips"); axs[1].set_ylabel("MPJPE from original (m)")
    axs[1].set_title("motion drift", fontsize=8)
    p = os.path.join(os.environ.get("WORK_DIR", "."), "fig_roundtrip_v2.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {p}")
    wandb.log({"roundtrip_v2_fig": wandb.Image(p.replace(".pdf", ".png"))})
except Exception as e:
    print("figure failed:", e)

dst = os.path.join(os.environ.get("WORK_DIR", "."), "roundtrip_v2.json")
json.dump(dict(rounds=K, n=len(idx), ble=B, drift=D), open(dst, "w"), indent=2)
print(f"raw results -> {dst}")
tbl = wandb.Table(columns=["round", "BLE_reproject", "BLE_plain", "BLE_control", "drift_reproject"],
                  data=[[k, B["reproj"][k], B["plain"][k], B["control"][k], D["reproj"][k]] for k in range(K + 1)])
wandb.log({"roundtrip_v2_table": tbl})
wandb.finish()
print("Round-trip v2 done.")
