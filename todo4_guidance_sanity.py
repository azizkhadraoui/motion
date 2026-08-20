#!/usr/bin/env python
# TODO 4 — DIRECT-SPACE SOFT-GUIDANCE SANITY CHECK (+ an implementation issue it exposes).
#
# Part A (what the TODO asks). At every ODE step, the sampler replaces the raw physical gradient by
# a unit-normalized direction scaled to the velocity magnitude. Normalization can only change the
# STEP LENGTH, not the sign, so the step must be locally descending -- but only for the quantity the
# gradient was taken of. We measure it directly: for each step, compare the raw unnormalized penalty
# at the current state against the penalty after the guidance displacement.
#
# Part B (the reason Part A may still not explain the BLE trend). The implemented guidance evaluates
# the penalty on the CURRENT ITERATE Y_t, i.e. pen(D(Y_t)), not on the estimated clean endpoint
#       Yhat_1 = Y_t + (1-t) v_theta(Y_t, t; c),
# which is the quantity the training-time penalty of Section 6.3 uses and the quantity the reported
# BLE is actually measured on. Descending pen(Y_t) need not descend pen(Yhat_1): for most of the
# trajectory Y_t is a partly-noisy state whose bone geometry is not the geometry of the final sample.
# We therefore log BOTH penalties per step, and additionally run an endpoint-guided sampler end to
# end so the paper can say which of the two the reported BLE trend belongs to.
#
# Outcomes and what to write:
#   raw penalty descends AND endpoint penalty descends -> guidance behaves as intended; the BLE
#       increase at large w is a genuine empirical limitation of soft guidance (TODO 4's first branch).
#   raw penalty descends but endpoint penalty does NOT -> the BLE trend is an artifact of guiding the
#       iterate rather than the endpoint. Do not attribute it to the physical effect of guidance;
#       report the endpoint-guided numbers instead, or state the limitation explicitly.
# Either way the robust claim is preserved: CDFM shows no catastrophic FID degradation over the sweep.
#
#   sbatch run_todo4.sh

import os, sys, math, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[todo4] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[todo4] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
diff_penalty = M.diff_penalty; _cfg = M._cfg; _timesteps = M._timesteps; rvq = M.rvq
eval_variant = M.eval_variant; load_net = M.load_net; _agg = M._agg

NB = int(os.environ.get("T4_BATCH", "32"))
WEIGHTS = [float(x) for x in os.environ.get("T4_WEIGHTS", "0.25,1.0,2.0").split(",")]
RUN_EVAL = os.environ.get("T4_EVAL", "1") == "1"

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "todo4_guidance_sanity"), job_type="diagnostic")

def decode(y, is_latent):
    return rvq.decoder(y * M.z_std_t + M.z_mean_t) if is_latent else y

def batch_cond(n):
    rng = np.random.default_rng(0)
    sel = sorted(rng.permutation(len(M.test_entries))[:n].tolist())
    caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
    lens = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
    ts, tm, tp = M.embed_text(caps)
    return (torch.tensor(ts, device=DEVICE), torch.tensor(tm, device=DEVICE),
            torch.tensor(tp, device=DEVICE), lens)

# --------------------------------------------------------------------- Part A/B: per-step probe
def probe(net, is_latent, gw, tseq, tmask, tpool, lens, n=ODE):
    B = tpool.shape[0]
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    torch.manual_seed(1234)
    y = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ts = _timesteps(n, "linear")
    rec = []
    for i in range(n):
        tval = float(ts[i]); dt = float(ts[i + 1] - ts[i])
        t = torch.full((B,), tval, device=DEVICE)
        with torch.no_grad():
            v = _cfg(net, y, t, tseq, tmask, tpool, lens, ns, nm, npl, GUID, 0.0)
        # gradient of the penalty of the ITERATE, exactly as the sampler implements it
        with torch.enable_grad():
            yc = y.detach().requires_grad_(True)
            pen_i = diff_penalty(decode(yc, is_latent), lens)
            g = torch.autograd.grad(pen_i, yc)[0]
        gh = g / g.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
        vmag = v.flatten(1).norm(dim=1).view(-1, 1, 1)
        step = -gw * gh * vmag * dt                    # the displacement guidance contributes
        with torch.no_grad():
            p_i0 = float(diff_penalty(decode(y, is_latent), lens))
            p_i1 = float(diff_penalty(decode(y + step, is_latent), lens))
            e0 = y + (1 - tval) * v                    # endpoint estimate before guidance
            v2 = _cfg(net, y + step, t, tseq, tmask, tpool, lens, ns, nm, npl, GUID, 0.0)
            e1 = (y + step) + (1 - tval) * v2          # endpoint estimate after guidance
            p_e0 = float(diff_penalty(decode(e0, is_latent), lens))
            p_e1 = float(diff_penalty(decode(e1, is_latent), lens))
            graw = float(g.flatten(1).norm(dim=1).mean())
        rec.append(dict(step=i, t=tval, pen_iter_before=p_i0, pen_iter_after=p_i1,
                        pen_end_before=p_e0, pen_end_after=p_e1, raw_grad_norm=graw))
        with torch.no_grad():
            y = y + dt * v + step
    return rec

# --------------------------------------------------------------------- endpoint-guided sampler
def make_endpoint_sampler(orig):
    """Same signature as M.sample; in mode 'guided' the penalty is taken of the endpoint estimate."""
    def sample_ep(net, is_latent, tseq, tmask, tpool, length, n=ODE, guidance=GUID, mode="none",
                  seed=None, guide_w=None, cfg_rescale=0.0, schedule="linear", solver="euler",
                  gw_schedule="const"):
        if mode != "guided":
            return orig(net, is_latent, tseq, tmask, tpool, length, n=n, guidance=guidance, mode=mode,
                        seed=seed, guide_w=guide_w, cfg_rescale=cfg_rescale, schedule=schedule,
                        solver=solver, gw_schedule=gw_schedule)
        if seed is not None: torch.manual_seed(seed)
        B = tpool.shape[0]
        z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
        ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
        nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
        npl = net.null_pool.unsqueeze(0).expand(B, -1)
        gw = float(guide_w) if guide_w is not None else float(getattr(M, "GUIDE_W", 0.0))
        ts = _timesteps(n, schedule)
        for i in range(n):
            tval = float(ts[i]); dt = float(ts[i + 1] - ts[i])
            t = torch.full((B,), tval, device=DEVICE)
            with torch.no_grad():
                v = _cfg(net, z, t, tseq, tmask, tpool, length, ns, nm, npl, guidance, cfg_rescale)
            if gw > 0.0:
                with torch.enable_grad():
                    zc = z.detach().requires_grad_(True)
                    vc = _cfg(net, zc, t, tseq, tmask, tpool, length, ns, nm, npl, guidance, cfg_rescale)
                    ehat = zc + (1.0 - tval) * vc          # estimated clean endpoint
                    pen = diff_penalty(decode(ehat, is_latent), length)
                    g = torch.autograd.grad(pen, zc)[0]
                g = g / g.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
                v = v - gw * g * v.flatten(1).norm(dim=1).view(-1, 1, 1)
            z = z + dt * v
        return decode(z, is_latent)
    return sample_ep

results = {}
print(f"\n{'='*96}\nPART A/B — per-step guidance probe (batch={NB})\n{'='*96}")
tseq, tmask, tpool, lens = batch_cond(NB)
for tag, is_lat in [("direct", False), ("latent", True)]:
    net = load_net(tag, is_lat)
    for gw in WEIGHTS:
        rec = probe(net, is_lat, gw, tseq, tmask, tpool, lens)
        di = np.array([r["pen_iter_after"] - r["pen_iter_before"] for r in rec])
        de = np.array([r["pen_end_after"] - r["pen_end_before"] for r in rec])
        gn = np.array([r["raw_grad_norm"] for r in rec])
        fi, fe = float((di < 0).mean()), float((de < 0).mean())
        print(f"\n  base={tag}  w={gw}")
        print(f"    raw penalty gradient norm      mean {gn.mean():.3e}  (this is what normalization rescales)")
        print(f"    ITERATE penalty  pen(D(Y_t)):  decreased on {100*fi:5.1f}% of steps, mean delta {di.mean():+.3e}")
        print(f"    ENDPOINT penalty pen(D(Yhat1)): decreased on {100*fe:5.1f}% of steps, mean delta {de.mean():+.3e}")
        results[f"{tag}_w{gw}"] = dict(frac_iter_down=fi, frac_end_down=fe,
                                       mean_d_iter=float(di.mean()), mean_d_end=float(de.mean()),
                                       raw_grad_norm=float(gn.mean()),
                                       per_step=[{k: float(v) for k, v in r.items()} for r in rec])
        wandb.log({f"todo4/{tag}/w{gw}/frac_iter_down": fi, f"todo4/{tag}/w{gw}/frac_end_down": fe})
        if fi > 0.9 and fe < 0.6:
            print("    >>> descends the ITERATE penalty but not the ENDPOINT penalty: the reported BLE")
            print("        trend cannot be attributed to the physical effect of guidance (TODO 4, branch 2).")
        elif fi > 0.9 and fe > 0.9:
            print("    >>> descends both: guidance behaves as intended (TODO 4, branch 1).")

# --------------------------------------------------------------------- endpoint-guided end-to-end eval
if RUN_EVAL:
    print(f"\n{'='*96}\nENDPOINT-GUIDED SAMPLER — full evaluation\n{'='*96}")
    orig = M.sample
    M.sample = make_endpoint_sampler(orig)          # eval_variant resolves `sample` as a module global
    try:
        for tag, is_lat in [("direct", False), ("latent", True)]:
            net = load_net(tag, is_lat)
            base = eval_variant(net, is_lat, "none")
            bm, _, _ = _agg(base["ble"])
            print(f"\n  {tag}: w=0 (control)      FID={base['fid']:.4f}  R@3={base['R3']:.4f}  BLE={bm:.5f}")
            for gw in WEIGHTS:
                r = eval_variant(net, is_lat, "guided", guide_w=gw)
                bm, _, bx = _agg(r["ble"])
                print(f"  {tag}: w={gw} endpoint-guided FID={r['fid']:.4f}  R@3={r['R3']:.4f}  BLE={bm:.5f}  BLEmax={bx:.5f}")
                results[f"endpoint_eval_{tag}_w{gw}"] = dict(fid=float(r["fid"]), R3=float(r["R3"]), ble=float(bm))
                wandb.log({f"todo4/endpoint/{tag}/FID": float(r["fid"]), f"todo4/endpoint/{tag}/BLE": float(bm),
                           f"todo4/endpoint/{tag}/w": gw})
    finally:
        M.sample = orig
    print("\nCompare these BLE values against Table 5. If endpoint guidance removes the CDFM BLE")
    print("increase, Table 5's BLE column is measuring the implementation choice, not soft guidance.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "todo4_guidance_sanity.json")
json.dump(results, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.finish()
print("TODO 4 sanity check done.")
