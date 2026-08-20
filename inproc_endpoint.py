#!/usr/bin/env python
# IN-PROCESS PROJECTION ON THE ESTIMATED ENDPOINT — a reviewer control for the central negative claim.
#
# The paper's in-ODE projection decodes and corrects the CURRENT ITERATE Z_t. A reviewer can object
# that this is not the strongest form of the method: the standard construction corrects the estimated
# clean sample, as diffusion methods correct xhat_0, and as this paper's own training-time penalty
# (Section 6.3, Eq. 6.2) already does with
#       Yhat_1 = Y_t + (1-t) v_theta(Y_t, t; c).
# Correcting a half-noisy iterate is a weaker operation, so "in-ODE enforcement fails" would be
# vulnerable to "you enforced it in the weakest possible place."
#
# This script runs the endpoint variant:
#       Yhat_1  <- Y_t + (1-t) v
#       Yhat_1' <- E(P(D(Yhat_1)))          (for LFM; P alone for CDFM)
#       Y_t     <- Yhat_1' - (1-t) v        (map the corrected endpoint back to the current time)
# and evaluates it under the same schedules as the existing ablation, alongside the iterate-based
# implementation as the reference.
#
# Expected outcome, given the Table 2 mechanism: the endpoint variant should also degrade, because
# the correction still passes through E o D on every intervention. If it does, the paper gains one
# sentence that closes the objection permanently. If it does NOT degrade, that is a genuinely
# important finding and the negative claim must be narrowed to iterate-based projection.
#
#   sbatch run_inproc_endpoint.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[inproc-ep] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[inproc-ep] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
_cfg = M._cfg; _timesteps = M._timesteps; rvq = M.rvq
project_joints = M.project_joints; _gj = M._gj; _joints_to_norm = M._joints_to_norm
eval_variant = M.eval_variant; load_net = M.load_net; _agg = M._agg

def decode(y, is_lat): return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_lat else y
def encode(mn, is_lat): return rvq.encoder(mn) if is_lat else mn

@torch.no_grad()
def sample_ep(net, is_latent, tseq, tmask, tpool, length, n=ODE, guidance=GUID, mode="none",
              seed=None, guide_w=None, cfg_rescale=0.0, schedule="linear", solver="euler",
              gw_schedule="const"):
    """Drop-in for M.sample. mode 'inproc' corrects the ESTIMATED ENDPOINT instead of the iterate."""
    if seed is not None: torch.manual_seed(seed)
    B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    win = float(getattr(M, "PROJ_WINDOW", 0.5)); stride = int(getattr(M, "PROJ_STRIDE", 1))
    start = int(round(n * (1.0 - win)))
    ts = _timesteps(n, schedule)
    for i in range(n):
        tval = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        t = torch.full((B,), tval, device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, length, ns, nm, npl, guidance, cfg_rescale)
        z = z + dt * v
        if mode == "inproc" and i >= start and ((i - start) % stride == 0):
            tnext = float(ts[i + 1])
            e = z + (1.0 - tnext) * v                       # estimated clean endpoint at t_{i+1}
            mn = decode(e, is_latent)
            mn2 = _joints_to_norm(project_joints(_gj(mn), length), mn)
            e2 = encode(mn2, is_latent)
            z = e2 - (1.0 - tnext) * v                      # map the corrected endpoint back to t_{i+1}
    mn = decode(z, is_latent)
    if mode in ("posthoc", "inproc"):
        mn = _joints_to_norm(project_joints(_gj(mn), length), mn)
    return mn

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "inproc_endpoint"), job_type="ablation")

# schedules: the best-known in-process config, the original config, and one in between
SCHEDULES = [(0.10, 4), (0.10, 1), (0.50, 1)]
rows = []
orig = M.sample
for tag, is_lat in [("latent", True), ("direct", False)]:
    if not M._have(tag): continue
    net = load_net(tag, is_lat)
    for kind in ("iterate", "endpoint"):
        M.sample = orig if kind == "iterate" else sample_ep
        for win, stride in SCHEDULES:
            M.PROJ_WINDOW = win; M.PROJ_STRIDE = stride
            t0 = time.time()
            r = eval_variant(net, is_lat, "inproc")
            fm, _, _ = _agg(r["fsr"]); bm, _, _ = _agg(r["ble"])
            lab = f"{tag} {kind} w={int(win*100)}% k={stride}"
            row = dict(base=tag, kind=kind, window=win, stride=stride, FID=round(float(r["fid"]), 4),
                       R3=round(float(r["R3"]), 4), BLE=round(bm, 5), FSR=round(fm, 4), sec=round(time.time() - t0))
            print(f"  {lab:<34} FID={row['FID']:<9} R@3={row['R3']:<7} BLE={row['BLE']:<8} ({row['sec']}s)")
            wandb.log({f"inproc_ep/{lab}/FID": row["FID"], f"inproc_ep/{lab}/BLE": row["BLE"]})
            rows.append(row)
M.sample = orig

print(f"\n{'='*92}")
print(f" {'base':<8}{'correction target':>20}{'window':>9}{'stride':>8}{'FID':>10}{'R@3':>8}{'BLE':>10}")
print("-" * 92)
for r in rows:
    print(f" {r['base']:<8}{r['kind']:>20}{int(r['window']*100):>8}%{r['stride']:>8}{r['FID']:>10}{r['R3']:>8}{r['BLE']:>10}")
print("=" * 92)

lat = [r for r in rows if r["base"] == "latent"]
if lat:
    it = min((r["FID"] for r in lat if r["kind"] == "iterate"), default=float("nan"))
    ep = min((r["FID"] for r in lat if r["kind"] == "endpoint"), default=float("nan"))
    print(f"\nLFM best FID: iterate-corrected {it}, endpoint-corrected {ep}.")
    if ep > 3.0:
        print(">>> Endpoint correction also collapses. Add one sentence to Section 4.2: the negative")
        print("    result is not an artifact of correcting a noisy iterate; correcting the estimated")
        print("    clean endpoint fails the same way, because the correction still passes through EoD.")
    else:
        print(">>> Endpoint correction does NOT collapse. This is a material finding: narrow the claim")
        print("    in Sections 3.1/4.2 to iterate-based in-ODE projection and report this variant.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "inproc_endpoint.json")
json.dump(rows, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.log({"inproc_endpoint_table": wandb.Table(
    columns=["base", "kind", "window", "stride", "FID", "R@3", "BLE", "FSR"],
    data=[[r["base"], r["kind"], r["window"], r["stride"], r["FID"], r["R3"], r["BLE"], r["FSR"]] for r in rows])})
wandb.finish()
print("Endpoint in-process control done.")
