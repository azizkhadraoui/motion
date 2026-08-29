#!/usr/bin/env python
# DOES STRAIGHTENING THE INTEGRATION-SPACE PATH IMPROVE FEW-STEP BEHAVIOUR?
#
# THE PREDICTION UNDER TEST. Our measurements say degradation is predicted by the curvature of the
# path the solver integrates, and not by the curvature of the decoded path. If that identifies the
# right geometry, then a method that reduces integration-space curvature should improve few-step
# quality, and a method that only affected the decoded geometry should not. Reflow is exactly such a
# method: it retrains the velocity field on the model's own noise-sample pairs, which are joined by
# straight lines by construction, so it targets the integration-space path directly.
#
# This turns a sentence in the discussion into a measurement. The paper currently says the prediction
# "is testable"; the referee's obvious question is why we did not test it.
#
# WHAT IS MEASURED, for teacher and reflow student on the same clips and matched noise:
#   1. integration-space arc/chord and maximum turning angle  -- did reflow actually straighten it?
#   2. decoded arc/chord                                       -- did the decoded geometry change too?
#   3. FID at n = 1,2,4,8,16,50                                -- did few-step quality improve?
#   4. the correlation structure within the student            -- does the same predictor still hold?
#
# HOW TO READ IT, stated before seeing the numbers:
#   straighter in Z AND better at low n   -> the prediction is confirmed; the identified geometry is
#                                            the actionable one, and the paper closes its own loop.
#   straighter in Z but NOT better at low n -> the prediction fails. Report it. Curvature would then
#                                            be diagnostic but not a lever, which is a weaker but
#                                            honest claim and must replace the discussion sentence.
#   not straighter in Z                    -> reflow did not do its job here (too few pairs, too
#                                            short a schedule), and the test is inconclusive rather
#                                            than negative. Say so; do not report it as evidence.
#
# CRITICAL: the student was trained on pairs generated WITH classifier-free guidance, so it has the
# guided field baked in and must be sampled at guidance 1.0. Sampling it at 2.5 double-applies
# guidance and would make a working student look broken.
#
#   sbatch run_reflow_test.sh

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[reflowtest] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[reflowtest] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm; load_net = M.load_net

N_EVAL = int(os.environ.get("EVAL_N", "512"))
NFES = [int(x) for x in os.environ.get("RT_NFE", "1,2,4,8,16,50").split(",")]
REPS = int(os.environ.get("RT_REPS", "3"))
N_REF = 50

TCRIT = {2:4.303,3:3.182,4:2.776,5:2.571,9:2.262,19:2.093}
def ci95(a):
    a = np.asarray(a, float); n = len(a)
    if n < 2: return float(a.mean()), float("nan")
    return float(a.mean()), float(TCRIT.get(n - 1, 2.093) * a.std(ddof=1) / math.sqrt(n))
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def decode(z): return rvq.decoder(z * M.z_std_t + M.z_mean_t)
def masked_rms(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    d = ((A - B) ** 2).sum(-1).mean(-1) if A.dim() == 4 else ((A - B) ** 2).mean(-1)
    return ((d * fm).sum(1) / fm.sum(1).clamp_min(1)).sqrt()
def mpjpe(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    return ((A - B).norm(dim=-1).mean(-1) * fm).sum(1) / fm.sum(1).clamp_min(1)

@torch.no_grad()
def run(net, guidance, tseq, tmask, tpool, L, n, seed, trace=False):
    torch.manual_seed(seed); B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    ts = _timesteps(n, "linear"); zs, js = [], []
    for i in range(n):
        t = torch.full((B,), float(ts[i]), device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, L, ns, nm, npl, guidance, 0.0)
        if trace: zs.append(z.clone()); js.append(_gj(decode(z)))
        z = z + float(ts[i + 1] - ts[i]) * v
    if trace: zs.append(z.clone()); js.append(_gj(decode(z)))
    return z, zs, js

def arc_over_chord(states, L, joints):
    arc = 0.0
    for a, b in zip(states[:-1], states[1:]):
        arc = arc + (masked_rms(b, a, L) if joints else (b - a).flatten(1).norm(dim=1))
    ch = masked_rms(states[-1], states[0], L) if joints else (states[-1] - states[0]).flatten(1).norm(dim=1)
    return arc / ch.clamp_min(1e-8)

def max_turn(states):
    d = [(b - a).flatten(1) for a, b in zip(states[:-1], states[1:])]
    worst = torch.zeros(d[0].shape[0], device=d[0].device)
    for a, b in zip(d[:-1], d[1:]):
        c = (a * b).sum(1) / (a.norm(dim=1).clamp_min(1e-8) * b.norm(dim=1).clamp_min(1e-8))
        worst = torch.maximum(worst, torch.arccos(c.clamp(-1 + 1e-6, 1 - 1e-6)))
    return worst

if not M._have("reflow"):
    print("\nNo reflow checkpoint found under the model directory. This test needs the student from")
    print("the reflow run (reflow_best.pt). Nothing to do; skipping rather than reporting a null.")
    sys.exit(0)

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "reflow_prediction_test"), job_type="analysis")

rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

# teacher at the trained guidance scale; student at 1.0 because guidance is baked into its pairs
MODELS = [("teacher", load_net("latent", True), GUID),
          ("reflow",  load_net("reflow", True), 1.0)]

out = {}
for name, net, g in MODELS:
    print(f"\n{'='*100}\n{name.upper()}  (guidance {g})\n{'='*100}")
    per = {"arcZ": [], "arcQ": [], "turn": [], "fid": {n: [] for n in NFES},
           "corr_arcZ": {n: [] for n in NFES}, "corr_arcQ": {n: [] for n in NFES}}
    t0 = time.time()
    for rep in range(REPS):
        aZ, aQ, tu = [], [], []
        deg = {n: [] for n in NFES}; mf = {n: [] for n in NFES}; real = []
        for s in range(0, N_EVAL, 32):
            e = min(s + 32, N_EVAL)
            ts_ = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
            tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
            gm = lengths_to_mask(L, MAXLEN); seed = s + rep * 100000
            zref, zs, js = run(net, g, ts_, tm, tp, L, N_REF, seed, trace=True)
            Jref = _gj(decode(zref))
            aZ.append(arc_over_chord(zs, L, False).cpu()); aQ.append(arc_over_chord(js, L, True).cpu())
            tu.append(max_turn(zs).cpu())
            for n in NFES:
                zn, _, _ = run(net, g, ts_, tm, tp, L, n, seed)
                xn = decode(zn)
                mf[n].append(memb(xn * gm[..., None], L))
                deg[n].append(mpjpe(_gj(xn), Jref, L).cpu())
            rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
            real.append(memb(rm * gm[..., None], L))
        aZ = torch.cat(aZ).numpy(); aQ = torch.cat(aQ).numpy(); tu = torch.cat(tu).numpy()
        R = np.concatenate(real, 0)
        per["arcZ"].append(float(aZ.mean())); per["arcQ"].append(float(aQ.mean())); per["turn"].append(float(tu.mean()))
        for n in NFES:
            per["fid"][n].append(float(fid_calc(np.concatenate(mf[n], 0), R)))
            d = torch.cat(deg[n]).numpy()
            per["corr_arcZ"][n].append(spearman(aZ, d)); per["corr_arcQ"][n].append(spearman(aQ, d))
        print(f"  rep {rep+1}/{REPS}  ({(time.time()-t0)/60:.1f}m)")
    out[name] = dict(
        arcZ=ci95(per["arcZ"]), arcQ=ci95(per["arcQ"]), turn=ci95(per["turn"]),
        fid={str(n): ci95(per["fid"][n]) for n in NFES},
        corr_arcZ={str(n): ci95(per["corr_arcZ"][n]) for n in NFES},
        corr_arcQ={str(n): ci95(per["corr_arcQ"][n]) for n in NFES})
    print(f"  integration arc/chord {out[name]['arcZ'][0]:.4f} +/- {out[name]['arcZ'][1]:.4f}")
    print(f"  max turning angle     {out[name]['turn'][0]:.4f} +/- {out[name]['turn'][1]:.4f}")
    print(f"  decoded arc/chord     {out[name]['arcQ'][0]:.4f} +/- {out[name]['arcQ'][1]:.4f}")
    print("  FID: " + "  ".join(f"n={n}:{out[name]['fid'][str(n)][0]:.4f}" for n in NFES))

T, S = out["teacher"], out["reflow"]
print(f"\n{'='*100}\nREADING\n{'='*100}")
straighter = S["arcZ"][0] < T["arcZ"][0] - max(T["arcZ"][1], 1e-6)
low = [n for n in NFES if n <= 4]
better = all(S["fid"][str(n)][0] < T["fid"][str(n)][0] for n in low)
print(f"  integration-space arc/chord: teacher {T['arcZ'][0]:.4f} -> student {S['arcZ'][0]:.4f}"
      f"   {'STRAIGHTER' if straighter else 'not straighter'}")
print(f"  decoded arc/chord:           teacher {T['arcQ'][0]:.4f} -> student {S['arcQ'][0]:.4f}")
for n in NFES:
    print(f"    n={n:<3} teacher {T['fid'][str(n)][0]:.4f}   student {S['fid'][str(n)][0]:.4f}")
if not straighter:
    print("\n  >>> Reflow did not straighten the integration-space path in this run. The test is")
    print("      INCONCLUSIVE, not negative: with 20k pairs and a short schedule the student may")
    print("      simply not have rectified the field. Report it as such, or omit it, but do not")
    print("      present it as evidence against the prediction.")
elif better:
    print("\n  >>> PREDICTION CONFIRMED. Reducing integration-space curvature improves few-step")
    print("      quality. The geometry our correlations identify is the actionable one, and the")
    print("      discussion sentence becomes a result rather than a conjecture.")
else:
    print("\n  >>> PREDICTION FAILS. The path was straightened but few-step quality did not improve.")
    print("      Curvature is then diagnostic but not a lever. Report this plainly and revise the")
    print("      discussion: identifying the predictive geometry does not imply that acting on it")
    print("      helps.")
print("\n  Reminder: the student is sampled at guidance 1.0 because its training pairs were generated")
print("  with CFG applied. A student sampled at 2.5 would double-apply guidance and look broken.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "reflow_prediction_test.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.finish()
print("Reflow prediction test done.")
