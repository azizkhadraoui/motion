#!/usr/bin/env python
"""
inproc_restd.py — resolves the two open questions on the in-ODE projection path.

Q1. WHERE IS BLE MEASURED?
    Answered from the code, not from a run: sample() applies a FINAL projection to
    mode="inproc" as well as to mode="posthoc" (lines 454-455 of the main script),
    and eval_variant computes BLE on the returned motion. So the BLE=0.0 column of
    the published in-process ablation is produced by that final projection, not by
    the in-trajectory ones. This script measures BLE both BEFORE and AFTER the final
    projection so the two are separated for the first time.

Q2. IS THE COLLAPSE THE MISSING RE-STANDARDIZATION?
    The in-ODE branch decodes with z*z_std+z_mean and re-encodes without inverting
    it, so the state returns to the ODE on the raw latent scale but is consumed as
    standardized. This script runs both arms and reports them side by side.

DESIGN
    2 x 2 arms:  restd in {off, on}  x  final projection in {on, off}
    plus two references (unconstrained, post-hoc only) that do not depend on either flag.
    A scale probe records ||z|| per element before and after every re-encode; for a
    correctly standardized latent this should sit near 1.0 and stay there.

    Everything else -- clip subset, seeds, guidance, step count, batching -- is copied
    from the main script's eval_variant so the numbers are comparable to the published
    ablation cell for cell.

USAGE
    WORK_DIR=$WORK/runs python inproc_restd.py
    env knobs:  ID_BASE=latent|direct   ID_N=512   ID_WINDOW=0.10   ID_STRIDES=4,2,1
"""
import os, sys, json, math, time
os.environ.setdefault("VARIANT", "eval")
os.environ["USE_WANDB"] = "0"
os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util

MAIN = os.environ.get("MAIN_SCRIPT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN)
M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[inproc] importing {MAIN} ...", flush=True)
spec.loader.exec_module(M)

DEVICE = M.DEVICE
ODE = M.ODE_STEPS
GUID = M.GUIDANCE
T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN
rvq = M.rvq
_cfg, _timesteps, embed_text = M._cfg, M._timesteps, M.embed_text
memb, fid_calc, rprec = M.memb, M.fid_calc, M.rprec
_gj, lengths_to_mask, pad_norm = M._gj, M.lengths_to_mask, M.pad_norm
project_joints, _joints_to_norm = M.project_joints, M._joints_to_norm
ble_pc_joints, fsr_pc = M.ble_pc_joints, M.fsr_pc

BASE     = os.environ.get("ID_BASE", "latent")
N        = int(os.environ.get("ID_N", "512"))
WINDOW   = float(os.environ.get("ID_WINDOW", "0.10"))
STRIDES  = [int(s) for s in os.environ.get("ID_STRIDES", "4,2,1").split(",")]
WORK_DIR = os.environ.get("WORK_DIR", ".")
IS_LAT   = (BASE == "latent")

net = M.load_net(BASE, IS_LAT)          # also restores z_mean_t / z_std_t
Z_MEAN, Z_STD = M.z_mean_t, M.z_std_t


# ---------------------------------------------------------------------------
# sample() replicated from the main script, with two flags and a scale probe.
# Only the two marked lines differ from the original.
# ---------------------------------------------------------------------------
def sample_diag(net, is_latent, tseq, tmask, tpool, length, *, n=ODE_STEPS,
                guidance=GUID, mode="none", seed=None, window=0.5, stride=1,
                restd=False, final_proj=True, probe=None):
    if seed is not None:
        torch.manual_seed(seed)
    B, cd, Tl = tpool.shape[0], net.cd, net.Tlen
    z = torch.randn(B, Tl, cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    start = int(round(n * (1.0 - window)))
    ts = _timesteps(n, "linear")

    for i in range(n):
        tval = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        t = torch.full((B,), tval, device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, length, ns, nm, npl, guidance, 0.0)
        z = z + dt * v
        if mode == "inproc" and i >= start and ((i - start) % stride == 0):
            if probe is not None:
                probe.setdefault("pre", []).append(
                    float(z.pow(2).mean().sqrt().item()))
            mn = (rvq.decoder(z * Z_STD + Z_MEAN) if is_latent else z)
            J = project_joints(_gj(mn), length)
            mn2 = _joints_to_norm(J, mn)
            if is_latent:
                zr = rvq.encoder(mn2)
                # ---- the one line under test -------------------------------
                z = (zr - Z_MEAN) / Z_STD if restd else zr
                # ------------------------------------------------------------
            else:
                z = mn2
            if probe is not None:
                probe.setdefault("post", []).append(
                    float(z.pow(2).mean().sqrt().item()))

    mn = (rvq.decoder(z * Z_STD + Z_MEAN) if is_latent else z)
    mn_pre = mn.clone()                       # BEFORE the final projection
    if final_proj and mode in ("posthoc", "inproc"):
        J = project_joints(_gj(mn), length)
        mn = _joints_to_norm(J, mn)
    return mn, mn_pre


# ---------------------------------------------------------------------------
def evaluate(label, *, mode, window=0.5, stride=1, restd=False, final_proj=True,
             probe=None):
    """Mirrors eval_variant: same clip subset, same per-batch seeds."""
    rng = np.random.default_rng(0)
    sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N].tolist()))
    caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
    lens = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
    tseq, tmask, tpool = embed_text(caps)

    ble_post, ble_pre, fsr, mf, real_mf = [], [], [], [], []
    t0 = time.time()
    for s in range(0, N, 32):
        e = min(s + 32, N)
        ts_ = torch.tensor(tseq[s:e], device=DEVICE)
        tm = torch.tensor(tmask[s:e], device=DEVICE)
        tp = torch.tensor(tpool[s:e], device=DEVICE)
        x, x_pre = sample_diag(net, IS_LAT, ts_, tm, tp, lens[s:e], mode=mode,
                               seed=s, window=window, stride=stride,
                               restd=restd, final_proj=final_proj, probe=probe)
        gm = lengths_to_mask(lens[s:e], MAXLEN)
        ble_post.append(ble_pc_joints(_gj(x), lens[s:e]))
        ble_pre.append(ble_pc_joints(_gj(x_pre), lens[s:e]))
        fsr.append(fsr_pc(_gj(x), lens[s:e]))
        mf.append(memb(x * gm[..., None], lens[s:e]))
        rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0]
                                    for i in sel[s:e]]), device=DEVICE)
        real_mf.append(memb(rm * gm[..., None], lens[s:e]))

    G = np.concatenate(mf, 0); R = np.concatenate(real_mf, 0)
    out = dict(label=label,
               fid=float(fid_calc(G, R)), r3=float(rprec(G, R)[3]),
               ble_post=float(np.concatenate(ble_post).mean()),
               ble_pre=float(np.concatenate(ble_pre).mean()),
               fsr=float(np.concatenate(fsr).mean()),
               secs=round(time.time() - t0))
    print(f"  {label:<44} FID={out['fid']:<9.4f} R@3={out['r3']:.4f}  "
          f"BLE_pre={out['ble_pre']:.5f}  BLE_post={out['ble_post']:.5f}  "
          f"FSR={out['fsr']:.4f}  ({out['secs']}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
print("\n" + "=" * 104)
print(f"IN-ODE RE-STANDARDIZATION TEST — base={BASE}, {N} clips, window={WINDOW:.0%}")
print("=" * 104, flush=True)

rows = [evaluate("unconstrained", mode="none"),
        evaluate("post-hoc (final only)", mode="posthoc")]

probes = {}
for k in STRIDES:
    for restd in ([False, True] if IS_LAT else [False]):
        for fp in [True, False]:
            tag = ("restd ON " if restd else "restd off") + \
                  ("  +final proj" if fp else "  NO final proj")
            pr = {} if fp else None
            r = evaluate(f"inproc k={k}  {tag}", mode="inproc", window=WINDOW,
                         stride=k, restd=restd, final_proj=fp, probe=pr)
            r.update(stride=k, restd=restd, final_proj=fp)
            rows.append(r)
            if pr:
                probes[f"k={k} restd={'on' if restd else 'off'}"] = dict(
                    pre_rms=float(np.mean(pr.get("pre", [np.nan]))),
                    post_rms=float(np.mean(pr.get("post", [np.nan]))))

# ---- scale probe ----------------------------------------------------------
if IS_LAT and probes:
    print("\n" + "=" * 104)
    print("LATENT SCALE PROBE — root-mean-square of z per element at each projection")
    print("  A correctly standardized latent sits near 1.0. 'post' is the state handed")
    print("  back to the ODE. If post >> pre with the fix off and post ~ pre with it on,")
    print("  the collapse is the scale corruption and not a structural limit.")
    print("-" * 104)
    print(f"  {'arm':<28}{'before re-encode':>20}{'after re-encode':>20}{'ratio':>12}")
    for k, v in probes.items():
        ratio = v["post_rms"] / v["pre_rms"] if v["pre_rms"] else float("nan")
        print(f"  {k:<28}{v['pre_rms']:>20.4f}{v['post_rms']:>20.4f}{ratio:>12.2f}")

# ---- summary --------------------------------------------------------------
print("\n" + "=" * 104)
print(f" {'configuration':<46}{'FID':>10}{'R@3':>9}{'BLE_pre':>11}{'BLE_post':>11}{'FSR':>9}")
print("-" * 104)
for r in rows:
    print(f" {r['label']:<46}{r['fid']:>10.4f}{r['r3']:>9.4f}"
          f"{r['ble_pre']:>11.5f}{r['ble_post']:>11.5f}{r['fsr']:>9.4f}")
print("=" * 104)

print("""
READING

  Q1 -- where BLE is measured.
    Compare BLE_pre against BLE_post on any 'inproc ... +final proj' row. BLE_post is
    what the published ablation reports. If BLE_post is 0 while BLE_pre is not, the
    zero comes from the final projection, and the ablation's BLE column says nothing
    about in-trajectory enforcement. The 'NO final proj' rows are the honest measure
    of what the in-ODE projections achieve on their own.

  Q2 -- the re-standardization.
    Compare 'restd off' against 'restd ON' at matched k. Two outcomes:
      (a) restd ON lands near the post-hoc FID -> the published collapse was the bug.
          The structural claim then has NO hard-projection evidence and must rest on
          the soft-guidance sweep, which is clean and unaffected.
      (b) restd ON still collapses -> the ceiling is real and the claim is stronger
          than it is now, because the obvious implementation objection is closed.
    Either way the paper needs this table; (b) is the better outcome and is why the
    run is worth doing before the framing is fixed.
""")

dst = os.path.join(WORK_DIR, f"inproc_restd_{BASE}.json")
json.dump(dict(base=BASE, n=N, window=WINDOW, strides=STRIDES,
               rows=rows, scale_probe=probes),
          open(dst, "w"), indent=2)
print(f"raw results -> {dst}")
