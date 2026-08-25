#!/usr/bin/env python
# TRANSPORT GEOMETRY: DOES THE LINEAR FLOW-MATCHING PATH SURVIVE DECODING?
#
# Conditional flow matching with the OT/linear interpolant trains on
#       Y_t = (1-t) Y_0 + t Y_1,
# a straight line in the generation space, and the target velocity Y_1 - Y_0 is constant along it.
# That construction is what makes the transport "optimal" for the conditional coupling, and it is
# why few-step sampling works at all.
#
# For CDFM the generation space IS motion space, so the path a sample travels is straight in the
# space where physical validity is defined. For LFM it is straight in Z, and what actually reaches
# the metric is D_phi of that path. Because the decoder is nonlinear, a straight line in Z decodes
# to a CURVE in motion space. Three consequences worth measuring, none of which need the flow model
# or any training -- the frozen autoencoder alone determines them:
#
#   P1  METRIC DISTORTION. Minibatch-OT couplings, path optimality, and the interpolant itself all
#       assume the geometry of the generation space is meaningful. We test whether the encoder is
#       approximately an isometry: for pairs of real motions, correlate latent distance with motion
#       distance and with joint-space distance. Weak correlation means distances used to define
#       transport in Z do not correspond to distances in the space we care about.
#
#   P2  PATH DISTORTION, data to data. Interpolate linearly in Z between two encoded real motions,
#       decode along the way, and compare against the straight motion-space interpolation of the
#       same endpoints. We report the arc-length-to-chord ratio of the decoded path: 1.0 means the
#       decoded path is straight, larger means the decoder bends it.
#
#   P3  PATH DISTORTION, noise to data -- the path the sampler actually traverses. Take z_0 ~ N(0,I),
#       z_1 = E(X), interpolate linearly in Z, decode along t, and measure the same ratio. This is
#       the FM path itself. For CDFM the same measurement is 1.0 by construction, which gives a free
#       reference point: any excess for LFM is distortion introduced by the representation.
#       We also track bone-length error along the path, since a curved path visits states whose
#       decodings are further from the data manifold, which is the same mechanism that makes
#       mid-trajectory constraint correction unreliable.
#
# WHY THIS IS INTERESTING BEYOND OUR SETTING. Latent flow matching is usually justified by the
# dimensionality reduction and the tractability of the linear interpolant. If the interpolant's
# straightness is destroyed by the decoder, then the property that motivates the OT path holds in a
# space that is not the space of interest -- a general observation about latent flow models, not a
# statement about bone lengths.
#
#   sbatch run_transport.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[transport] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[transport] models loaded.")

DEVICE = M.DEVICE; rvq = M.rvq; MAXLEN = M.MAX_MOTION_LEN
EDGES = M.EDGES; rest_len = M.rest_len; lengths_to_mask = M.lengths_to_mask
_gj = M._gj; pad_norm = M.pad_norm; ble_pc_joints = M.ble_pc_joints
EI = torch.tensor([e[0] for e in EDGES], device=DEVICE)
EJ = torch.tensor([e[1] for e in EDGES], device=DEVICE)

N_PAIR = int(os.environ.get("TG_PAIRS", "256"))
STEPS  = int(os.environ.get("TG_STEPS", "33"))     # points along each path, including endpoints

def masked_rms(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    if A.dim() == 4:            # joints (B,T,J,3)
        d = ((A - B) ** 2).sum(-1).mean(-1)
    else:                       # motion vectors (B,T,C)
        d = ((A - B) ** 2).mean(-1)
    return ((d * fm).sum(1) / fm.sum(1).clamp_min(1)).sqrt()

import random as _r; _r.seed(0)
ids = _r.sample(range(len(M.test_entries)), min(2 * N_PAIR, len(M.test_entries)))
A_id, B_id = ids[:N_PAIR], ids[N_PAIR:2 * N_PAIR]

def load(idl):
    mn = torch.stack([torch.tensor(pad_norm(M.test_entries[i]["motion"])[0], device=DEVICE) for i in idl])
    L = torch.tensor([min(len(M.test_entries[i]["motion"]), MAXLEN) for i in idl], device=DEVICE)
    return mn, L

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "transport_geometry"), job_type="analysis")

out = {}

# ------------------------------------------------------------------ P1 metric distortion
print(f"\n{'='*96}\nP1 — IS THE ENCODER APPROXIMATELY AN ISOMETRY? ({N_PAIR} motion pairs)\n{'='*96}")
d_lat, d_mot, d_jnt = [], [], []
with torch.no_grad():
    for s in range(0, N_PAIR, 32):
        a, La = load(A_id[s:s + 32]); b, Lb = load(B_id[s:s + 32])
        L = torch.minimum(La, Lb)
        za = (rvq.encoder(a) - M.z_mean_t) / M.z_std_t
        zb = (rvq.encoder(b) - M.z_mean_t) / M.z_std_t
        d_lat.append((za - zb).flatten(1).norm(dim=1).cpu())
        d_mot.append(masked_rms(a, b, L).cpu())
        d_jnt.append(masked_rms(_gj(a), _gj(b), L).cpu())
d_lat = torch.cat(d_lat).numpy(); d_mot = torch.cat(d_mot).numpy(); d_jnt = torch.cat(d_jnt).numpy()

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])
def pearson(x, y): return float(np.corrcoef(x, y)[0, 1])

p1 = dict(spearman_lat_mot=spearman(d_lat, d_mot), pearson_lat_mot=pearson(d_lat, d_mot),
          spearman_lat_jnt=spearman(d_lat, d_jnt), pearson_lat_jnt=pearson(d_lat, d_jnt),
          cv_ratio=float(np.std(d_lat / np.maximum(d_mot, 1e-8)) / np.mean(d_lat / np.maximum(d_mot, 1e-8))))
print(f"  latent vs 263-d motion distance:  Spearman {p1['spearman_lat_mot']:.3f}   Pearson {p1['pearson_lat_mot']:.3f}")
print(f"  latent vs joint-space distance:   Spearman {p1['spearman_lat_jnt']:.3f}   Pearson {p1['pearson_lat_jnt']:.3f}")
print(f"  scale ratio d_lat/d_mot, coeff. of variation: {p1['cv_ratio']:.3f}")
print("  A high correlation with a low coefficient of variation would mean the encoder nearly")
print("  preserves the metric, so transport defined in Z corresponds to transport in motion space.")
print("  Departures quantify how much the latent geometry rearranges distances.")
out["P1_metric"] = p1

# ------------------------------------------------------------------ P2 / P3 path distortion
def path_stats(z0, z1, L, decode, label):
    """Arc length / chord in motion space for the decoded linear path, plus BLE along the way."""
    ts = torch.linspace(0, 1, STEPS, device=DEVICE)
    prev = None; arc = 0.0; bles = []
    first = last = None
    with torch.no_grad():
        for k, t in enumerate(ts):
            z = (1 - t) * z0 + t * z1
            J = decode(z)
            if k == 0: first = J
            if k == STEPS - 1: last = J
            if prev is not None: arc = arc + masked_rms(J, prev, L)
            prev = J
            bles.append(float(ble_pc_joints(J, L).mean()))
    chord = masked_rms(last, first, L)
    ratio = (arc / chord.clamp_min(1e-8))
    r = dict(label=label, ratio_mean=float(ratio.mean()), ratio_p95=float(np.percentile(ratio.cpu().numpy(), 95)),
             ble_along=bles, ble_max_along=float(max(bles)), ble_end=float(bles[-1]))
    print(f"  {label:<44} arc/chord {r['ratio_mean']:.4f} (p95 {r['ratio_p95']:.4f})   "
          f"max BLE along path {r['ble_max_along']:.5f}")
    return r

dec_lat = lambda z: _gj(rvq.decoder(z * M.z_std_t + M.z_mean_t))
dec_dir = lambda x: _gj(x)

print(f"\n{'='*96}\nP2 — DATA-TO-DATA PATHS: straight in Z, how curved after decoding?\n{'='*96}")
rows2 = []
with torch.no_grad():
    for s in range(0, min(N_PAIR, 128), 32):
        a, La = load(A_id[s:s + 32]); b, Lb = load(B_id[s:s + 32]); L = torch.minimum(La, Lb)
        za = (rvq.encoder(a) - M.z_mean_t) / M.z_std_t
        zb = (rvq.encoder(b) - M.z_mean_t) / M.z_std_t
        rows2.append(path_stats(za, zb, L, dec_lat, f"latent interpolation [batch {s//32}]"))
        rows2.append(path_stats(a, b, L, dec_dir, f"motion interpolation [batch {s//32}]"))
lat2 = [r for r in rows2 if r["label"].startswith("latent")]
mot2 = [r for r in rows2 if r["label"].startswith("motion")]
out["P2_data_to_data"] = dict(latent_ratio=float(np.mean([r["ratio_mean"] for r in lat2])),
                              motion_ratio=float(np.mean([r["ratio_mean"] for r in mot2])),
                              latent_ble_max=float(np.mean([r["ble_max_along"] for r in lat2])),
                              motion_ble_max=float(np.mean([r["ble_max_along"] for r in mot2])))

print(f"\n{'='*96}\nP3 — NOISE-TO-DATA PATHS: the interpolant the sampler actually follows\n{'='*96}")
rows3 = []
torch.manual_seed(0)
with torch.no_grad():
    for s in range(0, min(N_PAIR, 128), 32):
        a, L = load(A_id[s:s + 32])
        z1 = (rvq.encoder(a) - M.z_mean_t) / M.z_std_t
        z0 = torch.randn_like(z1)
        rows3.append(path_stats(z0, z1, L, dec_lat, f"LFM noise->data [batch {s//32}]"))
        x0 = torch.randn_like(a)
        rows3.append(path_stats(x0, a, L, dec_dir, f"CDFM noise->data [batch {s//32}]"))
lat3 = [r for r in rows3 if r["label"].startswith("LFM")]
dir3 = [r for r in rows3 if r["label"].startswith("CDFM")]
out["P3_noise_to_data"] = dict(lfm_ratio=float(np.mean([r["ratio_mean"] for r in lat3])),
                               cdfm_ratio=float(np.mean([r["ratio_mean"] for r in dir3])),
                               lfm_ble_max=float(np.mean([r["ble_max_along"] for r in lat3])),
                               cdfm_ble_max=float(np.mean([r["ble_max_along"] for r in dir3])),
                               lfm_ble_curve=[float(np.mean([r["ble_along"][k] for r in lat3])) for k in range(STEPS)],
                               cdfm_ble_curve=[float(np.mean([r["ble_along"][k] for r in dir3])) for k in range(STEPS)])

print(f"\n{'='*96}\nSUMMARY\n{'='*96}")
p2 = out["P2_data_to_data"]; p3 = out["P3_noise_to_data"]
print(f" data->data   latent-interpolated path arc/chord {p2['latent_ratio']:.4f}, "
      f"motion-interpolated {p2['motion_ratio']:.4f}")
print(f" noise->data  LFM path arc/chord {p3['lfm_ratio']:.4f}, CDFM {p3['cdfm_ratio']:.4f}")
print("=" * 96)
print("\nREADING")
excess = p3["lfm_ratio"] / max(p3["cdfm_ratio"], 1e-8)
print(f"  The CDFM figure is 1.0 up to discretization by construction, since its interpolant is")
print(f"  straight in the space being measured; it is the reference. LFM's ratio is {excess:.3f}x that,")
print("  which is the amount by which the frozen decoder bends the flow-matching path in the space")
print("  where physical validity is defined.")
if p3["lfm_ratio"] > 1.05 * p3["cdfm_ratio"]:
    print("  The linear interpolant's straightness therefore does not transfer through the decoder.")
    print("  Two implications worth stating: few-step sampling arguments that rest on path")
    print("  straightness apply to the latent path, not the decoded one; and mid-trajectory states")
    print("  visited by the sampler are further from the motion manifold than the endpoints, which")
    print("  is the same mechanism that makes corrections applied mid-trajectory unreliable.")
else:
    print("  The decoder preserves path straightness closely. That is itself worth reporting: it")
    print("  would mean the latent interpolant is a reasonable proxy for motion-space transport.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "transport_geometry.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.1))
    tt = np.linspace(0, 1, STEPS)
    axs[0].plot(tt, np.array(p3["lfm_ble_curve"]) * 1e3, "o-", ms=2.5, lw=1.1, color="#1f4e79", label="LFM (latent path)")
    axs[0].plot(tt, np.array(p3["cdfm_ble_curve"]) * 1e3, "s--", ms=2.5, lw=1.1, color="#c0504d", label="CDFM (motion path)")
    axs[0].set_xlabel("$t$ along the interpolant"); axs[0].set_ylabel(r"BLE ($\times10^{-3}$)")
    axs[0].set_yscale("log"); axs[0].legend(frameon=False, fontsize=6)
    axs[0].set_title("constraint violation along the path", fontsize=8, pad=3)
    axs[1].bar([0, 1], [p3["cdfm_ratio"], p3["lfm_ratio"]], 0.55,
               color=["#c0504d", "#1f4e79"], edgecolor="white")
    axs[1].axhline(1.0, color="0.4", ls=":", lw=0.9)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["CDFM", "LFM"])
    axs[1].set_ylabel("arc length / chord"); axs[1].set_title("path curvature in motion space", fontsize=8, pad=3)
    p = os.path.join(os.environ.get("WORK_DIR", "."), "fig_transport_geometry.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"figure -> {p}")
    wandb.log({"transport_fig": wandb.Image(p.replace(".pdf", ".png"))})
except Exception as e:
    print("figure failed:", e)
wandb.finish()
print("Transport geometry analysis done.")
