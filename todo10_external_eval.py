#!/usr/bin/env python
# TODO 10 (NEW) — EXTERNAL VALIDATION OF TERMINAL ANALYTIC CORRECTION.
#
# The question is narrow and does not need any external model to be RUN here: does the terminal
# bone-length correction P_X drive BLE to zero on motions produced by generators other than ours?
# This script is therefore split from the expensive half deliberately. It ingests motions that
# somebody else's model produced, in whatever form they came out, and does the evaluation:
#
#     load .npy motions  ->  convert to Cartesian joints  ->  BLE, FSR before
#                        ->  apply P_X                    ->  BLE, FSR after
#
# WHAT YOU MUST SUPPLY. A directory of .npy files, one motion per file, in one of:
#     (a) 263-d HumanML3D vectors, RAW (unnormalized)          shape (T, 263)
#     (b) 263-d HumanML3D vectors, already normalized by our Mean/Std   shape (T, 263)
#     (c) Cartesian joint positions                            shape (T, 22, 3)
# Set EXT_FORMAT=raw263 | norm263 | joints22. Most released text-to-motion repos (MDM, MoMask,
# T2M-GPT, MLD) can dump either (a) or (c) with a few lines in their sampling script, which is the
# only work that cannot be done from inside this project.
#
# TWO THINGS THIS SCRIPT DELIBERATELY DOES NOT DO, per the revised TODO 10:
#   1. It does not compute FID against our evaluator or compare generation quality with our models.
#      The external models use different architectures, training data and protocols, so a quality
#      comparison would not be meaningful and is not what the experiment is for.
#   2. It does not claim anything about the external models' latent representations. Constraint
#      non-preservation was measured for OUR frozen autoencoder; asserting it elsewhere would require
#      running the same round-trip diagnostic on their encoder, which this script does not do.
#
# REST-LENGTH SUBTLETY, which matters for interpreting the result. Our BLE is measured against the
# reference bone lengths estimated from HumanML3D. A different generator may produce a systematically
# different skeleton scale, in which case its raw BLE against OUR reference is inflated for reasons
# that have nothing to do with physical validity. The script therefore reports BLE against both
# (i) our reference lengths and (ii) per-clip median bone lengths estimated from the clip itself,
# which is scale-free and measures only rigidity over time. Report (ii) as the fair baseline number
# and (i) for comparability with our tables.
#
#   EXT_DIR=/path/to/motions EXT_FORMAT=raw263 EXT_NAME=MoMask sbatch run_todo10_external.sh

import os, sys, glob, json
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[todo10] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[todo10] models loaded.")

DEVICE = M.DEVICE; MAXLEN = M.MAX_MOTION_LEN; NFEATS = M.NFEATS; N_JOINTS = M.N_JOINTS
EDGES = M.EDGES; rest_len = M.rest_len; lengths_to_mask = M.lengths_to_mask
recover_from_ric = M.recover_from_ric; project_joints = M.project_joints
project_foot = M.project_foot; project_bonelength = M.project_bonelength
fsr_pc = M.fsr_pc; mean_t = M.mean_t; std_t = M.std_t
EI = torch.tensor([e[0] for e in EDGES], device=DEVICE)
EJ = torch.tensor([e[1] for e in EDGES], device=DEVICE)

EXT_DIR = os.environ.get("EXT_DIR", "")
EXT_FORMAT = os.environ.get("EXT_FORMAT", "raw263")
EXT_NAME = os.environ.get("EXT_NAME", os.path.basename(EXT_DIR.rstrip("/")) or "external")
LIMIT = int(os.environ.get("EXT_LIMIT", "512"))

if not EXT_DIR or not os.path.isdir(EXT_DIR):
    print(f"EXT_DIR not set or not a directory: {EXT_DIR!r}")
    print("\nThis script evaluates motions that an external model produced; it does not run those")
    print("models. Dump their outputs as .npy (one motion per file) and point EXT_DIR at them.")
    print("Formats accepted: raw263 (T,263) unnormalized | norm263 (T,263) our normalization |")
    print("joints22 (T,22,3) Cartesian positions.")
    sys.exit(1)

files = sorted(glob.glob(os.path.join(EXT_DIR, "*.npy")))[:LIMIT]
assert files, f"no .npy files under {EXT_DIR}"
print(f"[todo10] {len(files)} motions from {EXT_NAME}, format={EXT_FORMAT}")

def to_joints(a):
    """Return (1,T,22,3) joints on DEVICE, plus the normalized 263 vector when one exists."""
    x = torch.tensor(np.asarray(a, dtype=np.float32), device=DEVICE)
    if EXT_FORMAT == "joints22":
        if x.ndim == 3 and x.shape[1] == N_JOINTS: return x.unsqueeze(0), None
        raise ValueError(f"expected (T,22,3), got {tuple(x.shape)}")
    if x.ndim != 2 or x.shape[1] != NFEATS:
        raise ValueError(f"expected (T,{NFEATS}), got {tuple(x.shape)}")
    mn = ((x - mean_t) / std_t) if EXT_FORMAT == "raw263" else x
    return recover_from_ric((mn * std_t + mean_t).unsqueeze(0)), mn.unsqueeze(0)

def bone_lengths(J):
    return (J[:, :, EI, :] - J[:, :, EJ, :]).norm(dim=-1)          # (B,T,E)

def ble(J, L, ref):
    fm = lengths_to_mask(L, J.shape[1]).float().unsqueeze(-1)
    return float(((bone_lengths(J) - ref).abs() * fm).sum() / (fm.sum() * len(EDGES) + 1e-8))

rows = []
skipped = 0
stats = {k: [] for k in ("ble_ours", "ble_self", "fsr", "ble_ours_p", "ble_self_p", "fsr_p", "scale")}
with torch.no_grad():
    for f in files:
        try:
            J, mn = to_joints(np.load(f))
        except Exception as e:
            skipped += 1; continue
        T = J.shape[1]
        L = torch.tensor([T], device=DEVICE)
        self_ref = bone_lengths(J).median(dim=1).values[0]          # per-clip skeleton, scale-free
        stats["scale"].append(float((self_ref / rest_len).median()))
        stats["ble_ours"].append(ble(J, L, rest_len))
        stats["ble_self"].append(ble(J, L, self_ref))
        stats["fsr"].append(float(fsr_pc(J, L).mean()))
        # terminal correction, exactly as in the paper: foot contact then bone rescale
        Jp = project_bonelength(project_foot(J, L))
        stats["ble_ours_p"].append(ble(Jp, L, rest_len))
        stats["ble_self_p"].append(ble(Jp, L, self_ref))
        stats["fsr_p"].append(float(fsr_pc(Jp, L).mean()))

m = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
n = len(stats["ble_ours"])
print(f"\n{'='*92}\nTERMINAL CORRECTION ON EXTERNAL MOTIONS — {EXT_NAME} (n={n}, skipped={skipped})\n{'='*92}")
print(f" {'metric':<40}{'before':>14}{'after P_X':>14}")
print("-" * 92)
print(f" {'BLE vs our reference lengths':<40}{m['ble_ours']:>14.5f}{m['ble_ours_p']:>14.5f}")
print(f" {'BLE vs per-clip median lengths (scale-free)':<40}{m['ble_self']:>14.5f}{m['ble_self_p']:>14.5f}")
print(f" {'FSR':<40}{m['fsr']:>14.4f}{m['fsr_p']:>14.4f}")
print("=" * 92)
print(f"\n Skeleton scale vs our reference (median bone ratio): {m['scale']:.3f}")
if abs(m["scale"] - 1.0) > 0.05:
    print(" >>> The external skeleton differs in scale from ours by more than 5%. Report the")
    print("     scale-free row as the baseline BLE; the 'vs our reference' row is inflated by")
    print("     skeleton mismatch and should be labelled as such, not read as physical error.")
print("\nREADING")
if m["ble_ours_p"] < 1e-6:
    print(" The terminal correction reaches BLE = 0 against our reference lengths, as it must: it")
    print(" rescales each segment to the prescribed length by construction, independently of which")
    print(" generator produced the motion. The informative number is the BEFORE column -- it shows")
    print(" how much bone-length violation this external generator exhibits in the first place.")
else:
    print(" BLE did not reach zero after correction. Investigate before reporting: the most likely")
    print(" causes are a joint ordering that differs from our kinematic chains, or a frame count")
    print(" beyond our padding length.")
print("\n Do NOT extend this to a claim about the external model's latent representation. That would")
print(" require running the encode-decode round-trip diagnostic on their autoencoder.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), f"todo10_external_{EXT_NAME}.json")
json.dump(dict(name=EXT_NAME, n=n, skipped=skipped, format=EXT_FORMAT, means=m), open(dst, "w"), indent=2)
print(f"\nraw results -> {dst}")
