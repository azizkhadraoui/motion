#!/usr/bin/env python
# A6 — WEIGHT AVERAGING / "MODEL SOUP" over the checkpoints you already have.
#
# HONEST FRAMING FIRST. Wortsman et al.'s model soups average models fine-tuned INDEPENDENTLY from a
# shared init with different hyperparameters. You do not have that: you have one converged latent run
# (EMA + raw weights) and one penalty-trained run. So this is really a WEIGHT-SPACE INTERPOLATION
# study over 2-3 correlated points, not a soup. Expect a small gain or none. It is included because
# it costs one hour of eval and zero inference overhead, not because it is likely to be the win.
#
# Interpolations swept (alpha from 0 to 1):
#   S1  latent EMA  <->  latent raw weights     (from *_best.pt "state" and *_latest.pt "net")
#         The EMA and the raw endpoint are two nearby points on the same trajectory; interpolating
#         sometimes beats both. Cheapest possible test.
#   S2  latent EMA  <->  latent_pen EMA         (two DIFFERENTLY-TRAINED models, the real soup)
#         This is the interesting one: latent_pen was trained with the bone/foot penalty and is
#         dominated on its own (FID 0.292), but a partial interpolation toward it might inherit some
#         of its lower BLE without the FID cost. If any alpha beats 0.147, that is a genuine result
#         and it connects to the paper: it would show constraint-awareness helps in weight space even
#         though it fails in the ODE.
#
# GUARD: averaging weights across checkpoints trained under DIFFERENT latent standardizations would
# be meaningless. The script compares z_mean/z_std between checkpoints and refuses any pair that
# disagrees, rather than silently producing nonsense.
#
#   sbatch run_boost_soup.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[soup] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[soup] models loaded.")

DEVICE = M.DEVICE; CK = M.CK
FMNet = M.FMNet; eval_variant = M.eval_variant; _agg = M._agg
RVQ_CODE_DIM = M.RVQ_CODE_DIM; T_LAT = M.T_LAT
LHID, LLAYERS, LHEADS = M.LHID, M.LLAYERS, M.LHEADS

ALPHAS = [float(x) for x in os.environ.get("SOUP_ALPHAS", "0,0.125,0.25,0.375,0.5,0.75,1.0").split(",")]

def load_ck(name):
    p = os.path.join(CK, name)
    if not os.path.exists(p): return None
    return torch.load(p, map_location="cpu", weights_only=False)

def weights_of(ck, prefer="state"):
    for k in (prefer, "state", "ema", "net"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            return {kk: v.clone() for kk, v in ck[k].items()}
    return None

def zstats(ck):
    return (np.asarray(ck.get("z_mean")) if ck.get("z_mean") is not None else None,
            np.asarray(ck.get("z_std")) if ck.get("z_std") is not None else None)

def compatible(a, b):
    (am, asd), (bm, bsd) = zstats(a), zstats(b)
    if am is None or bm is None: return True, "one checkpoint has no z-stats; assuming shared"
    if am.shape != bm.shape: return False, "z-stat shapes differ"
    dm = float(np.abs(am - bm).max()); ds = float(np.abs(asd - bsd).max())
    return (dm < 1e-4 and ds < 1e-4), f"max|dz_mean|={dm:.2e} max|dz_std|={ds:.2e}"

def interp(wa, wb, a):
    out = {}
    for k in wa:
        if k not in wb: return None
        if wa[k].dtype.is_floating_point: out[k] = (1 - a) * wa[k].float() + a * wb[k].float()
        else: out[k] = wa[k].clone()
    return out

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "boost_soup"), job_type="ablation")

lb = load_ck("latent_best.pt"); ll = load_ck("latent_latest.pt"); lp = load_ck("latent_pen_best.pt")
assert lb is not None, "latent_best.pt not found"
zm, zs = zstats(lb)
M.z_mean_t = torch.tensor(zm, device=DEVICE).float(); M.z_std_t = torch.tensor(zs, device=DEVICE).float()

PAIRS = []
if ll is not None:
    ok, why = compatible(lb, ll)
    PAIRS.append(("S1 latent EMA -> latent raw", weights_of(lb, "state"), weights_of(ll, "net"), ok, why))
if lp is not None:
    ok, why = compatible(lb, lp)
    PAIRS.append(("S2 latent EMA -> latent_pen EMA", weights_of(lb, "state"), weights_of(lp, "state"), ok, why))
if not PAIRS:
    print("No second checkpoint found (need latent_latest.pt or latent_pen_best.pt). Nothing to soup.")
    sys.exit(0)

rows = []
for name, wa, wb, ok, why in PAIRS:
    print(f"\n{'='*92}\n{name}\n  compatibility: {'OK' if ok else 'REFUSED'} ({why})\n{'='*92}")
    if not ok:
        print("  Skipping: the two checkpoints use different latent standardizations, so averaging")
        print("  their weights has no meaning. This is a real hazard, not a formality.")
        continue
    if wa is None or wb is None:
        print("  Skipping: could not extract comparable state dicts."); continue
    for a in ALPHAS:
        w = interp(wa, wb, a)
        if w is None:
            print(f"  alpha={a}: key mismatch between checkpoints, skipped"); continue
        net = FMNet(RVQ_CODE_DIM, T_LAT, LHID, LLAYERS, LHEADS).to(DEVICE)
        net.load_state_dict({k: v.to(DEVICE) for k, v in w.items()}); net.eval()
        t0 = time.time()
        r = eval_variant(net, True, "none")
        rp = eval_variant(net, True, "posthoc")
        bm, _, _ = _agg(r["ble"])
        row = dict(pair=name, alpha=a, FID=round(float(r["fid"]), 4), R3=round(float(r["R3"]), 4),
                   BLE=round(bm, 5), FID_proj=round(float(rp["fid"]), 4), sec=round(time.time() - t0))
        rows.append(row)
        print(f"  alpha={a:<6} FID={row['FID']:<9} R@3={row['R3']:<7} BLE={row['BLE']:<9} "
              f"FID(+proj)={row['FID_proj']:<9} ({row['sec']}s)")
        wandb.log({f"soup/{name}/FID": row["FID"], f"soup/{name}/FID_proj": row["FID_proj"], "soup/alpha": a})
        del net; torch.cuda.empty_cache()

if rows:
    print(f"\n{'='*92}")
    print(f" {'pair':<34}{'alpha':>8}{'FID':>10}{'R@3':>8}{'FID+proj':>11}")
    print("-" * 92)
    for r in rows:
        print(f" {r['pair']:<34}{r['alpha']:>8}{r['FID']:>10}{r['R3']:>8}{r['FID_proj']:>11}")
    print("=" * 92)
    best = min(rows, key=lambda r: r["FID"])
    ends = [r for r in rows if r["alpha"] in (0.0, 1.0)]
    best_end = min(ends, key=lambda r: r["FID"]) if ends else None
    print(f"\nBest interpolant: {best['pair']} at alpha={best['alpha']}, FID {best['FID']}")
    if best_end and best_end["FID"] - best["FID"] > 0.006:
        print(">>> An interior alpha beats both endpoints by more than the seed-noise band. That is the")
        print("    interesting outcome -- confirm it across seeds before claiming it.")
    else:
        print(">>> No interior alpha clearly beats the endpoints. Expected for two correlated")
        print("    checkpoints; record as a null result and move on.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "boost_soup.json")
json.dump(rows, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.log({"soup_table": wandb.Table(columns=["pair", "alpha", "FID", "R@3", "BLE", "FID_proj"],
                                     data=[[r["pair"], r["alpha"], r["FID"], r["R3"], r["BLE"], r["FID_proj"]] for r in rows])})
wandb.finish()
print("A6 sweep done.")
