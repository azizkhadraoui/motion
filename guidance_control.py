#!/usr/bin/env python
# GUIDANCE CONTROL FOR THE REFLOW COMPARISON.
#
# THE CONFOUND. The reflow test compares the teacher at classifier-free guidance 2.5 against the
# student at guidance 1.0. The student must be sampled at 1.0 because its training pairs were
# generated with guidance already applied, so its velocity field has the guided field baked in;
# sampling it at 2.5 would apply guidance twice. That is correct, but it means the comparison mixes
# two changes at once -- the straightening of the trajectory, and the guidance scale -- and the
# reported improvement (FID 7.36 -> 0.75 at one step) cannot be attributed to straightening alone
# without a control. A referee will raise this immediately.
#
# THE CONTROL. Evaluate the TEACHER at guidance 1.0 on the same clips, seeds and step grid. This
# isolates the guidance scale: teacher@2.5 versus teacher@1.0 is the pure guidance effect, and
# teacher@1.0 versus student@1.0 is the pure straightening effect at matched guidance. We also record
# the teacher's trajectory curvature at 1.0, since guidance changes the velocity field and therefore
# the geometry -- if the teacher is already much straighter at 1.0, that is itself part of the
# explanation and needs to be visible.
#
# HOW TO READ IT, stated before the numbers are in:
#   teacher@1.0 still far worse than student@1.0 at low n
#       -> straightening carries the effect; the reflow result stands as a straightening result.
#   teacher@1.0 close to student@1.0 at low n
#       -> most of the apparent gain was the guidance scale, not the rectification. The honest
#          conclusion is then that reflow's contribution is smaller than the headline suggests, and
#          the paper must say so.
#   teacher@1.0 markedly straighter than teacher@2.5
#       -> guidance itself curves the trajectory. That is a finding in its own right and worth a
#          sentence: strong classifier-free guidance is one source of the curvature we measure.
#
# Inference only, both checkpoints frozen.
#
#   sbatch run_guidance_control.sh

import os, sys, json, time, math
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[gctrl] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[gctrl] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
MAXLEN = M.MAX_MOTION_LEN; rvq = M.rvq
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm; load_net = M.load_net

N_EVAL = int(os.environ.get("EVAL_N", "512"))
NFES = [int(x) for x in os.environ.get("GC_NFE", "1,2,4,8,16,50").split(",")]
REPS = int(os.environ.get("GC_REPS", "3"))
GUIDES = [float(x) for x in os.environ.get("GC_GUIDES", "1.0,1.5,2.5").split(",")]
N_REF = 50

TCRIT = {2:4.303,3:3.182,4:2.776,5:2.571,9:2.262,19:2.093}
def ci95(a):
    a = np.asarray(a, float); n = len(a)
    if n < 2: return float(a.mean()), float("nan")
    return float(a.mean()), float(TCRIT.get(n - 1, 2.093) * a.std(ddof=1) / math.sqrt(n))

def decode(z): return rvq.decoder(z * M.z_std_t + M.z_mean_t)
def masked_rms(A, B, L):
    fm = lengths_to_mask(L, MAXLEN).float()
    d = ((A - B) ** 2).sum(-1).mean(-1) if A.dim() == 4 else ((A - B) ** 2).mean(-1)
    return ((d * fm).sum(1) / fm.sum(1).clamp_min(1)).sqrt()

@torch.no_grad()
def run(net, guidance, tseq, tmask, tpool, L, n, seed, trace=False):
    torch.manual_seed(seed); B = tpool.shape[0]
    z = torch.randn(B, net.Tlen, net.cd, device=DEVICE)
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    ts = _timesteps(n, "linear"); zs = []
    for i in range(n):
        t = torch.full((B,), float(ts[i]), device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, L, ns, nm, npl, guidance, 0.0)
        if trace: zs.append(z.clone())
        z = z + float(ts[i + 1] - ts[i]) * v
    if trace: zs.append(z.clone())
    return z, zs

def arc_over_chord_z(zs):
    arc = 0.0
    for a, b in zip(zs[:-1], zs[1:]): arc = arc + (b - a).flatten(1).norm(dim=1)
    return arc / (zs[-1] - zs[0]).flatten(1).norm(dim=1).clamp_min(1e-8)

def max_turn(zs):
    d = [(b - a).flatten(1) for a, b in zip(zs[:-1], zs[1:])]
    worst = torch.zeros(d[0].shape[0], device=d[0].device)
    for a, b in zip(d[:-1], d[1:]):
        c = (a * b).sum(1) / (a.norm(dim=1).clamp_min(1e-8) * b.norm(dim=1).clamp_min(1e-8))
        worst = torch.maximum(worst, torch.arccos(c.clamp(-1 + 1e-6, 1 - 1e-6)))
    return worst

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "guidance_control"), job_type="analysis")

rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

ARMS = [("teacher", load_net("latent", True), g) for g in GUIDES]
if M._have("reflow"):
    ARMS.append(("reflow", load_net("reflow", True), 1.0))
else:
    print("[gctrl] no reflow checkpoint; running the teacher guidance sweep only.")

out = {}
for name, net, g in ARMS:
    key = f"{name}@{g}"
    print(f"\n{'='*96}\n{key}\n{'='*96}")
    per = {"arcZ": [], "turn": [], "fid": {n: [] for n in NFES}, "r3": {n: [] for n in NFES}}
    t0 = time.time()
    for rep in range(REPS):
        aZ, tu = [], []
        mf = {n: [] for n in NFES}; real = []
        for s in range(0, N_EVAL, 32):
            e = min(s + 32, N_EVAL)
            ts_ = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
            tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
            gm = lengths_to_mask(L, MAXLEN); seed = s + rep * 100000
            _, zs = run(net, g, ts_, tm, tp, L, N_REF, seed, trace=True)
            aZ.append(arc_over_chord_z(zs).cpu()); tu.append(max_turn(zs).cpu())
            for n in NFES:
                zn, _ = run(net, g, ts_, tm, tp, L, n, seed)
                mf[n].append(memb(decode(zn) * gm[..., None], L))
            rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
            real.append(memb(rm * gm[..., None], L))
        R = np.concatenate(real, 0)
        per["arcZ"].append(float(torch.cat(aZ).mean())); per["turn"].append(float(torch.cat(tu).mean()))
        for n in NFES:
            G = np.concatenate(mf[n], 0)
            per["fid"][n].append(float(fid_calc(G, R))); per["r3"][n].append(float(rprec(G, R)[3]))
        print(f"  rep {rep+1}/{REPS}  ({(time.time()-t0)/60:.1f}m)")
    out[key] = dict(arcZ=ci95(per["arcZ"]), turn=ci95(per["turn"]),
                    fid={str(n): ci95(per["fid"][n]) for n in NFES},
                    r3={str(n): ci95(per["r3"][n]) for n in NFES})
    print(f"  arc/chord {out[key]['arcZ'][0]:.4f} +/- {out[key]['arcZ'][1]:.4f}   "
          f"max turn {out[key]['turn'][0]:.4f} +/- {out[key]['turn'][1]:.4f}")
    print("  FID: " + "  ".join(f"n={n}:{out[key]['fid'][str(n)][0]:.4f}" for n in NFES))

print(f"\n{'='*96}\nSUMMARY\n{'='*96}")
print(f" {'arm':<16}{'arc/chord':>12}{'max turn':>11}" + "".join(f"{'n='+str(n):>10}" for n in NFES))
print("-" * 96)
for k, o in out.items():
    print(f" {k:<16}{o['arcZ'][0]:>12.4f}{o['turn'][0]:>11.4f}"
          + "".join(f"{o['fid'][str(n)][0]:>10.4f}" for n in NFES))
print("=" * 96)

t25, t10 = out.get(f"teacher@{GUIDES[-1]}"), out.get("teacher@1.0")
stu = out.get("reflow@1.0")
print("\nREADING")
if t25 and t10:
    print(f"  Guidance effect on geometry: arc/chord {t25['arcZ'][0]:.4f} at g={GUIDES[-1]} "
          f"-> {t10['arcZ'][0]:.4f} at g=1.0; max turn {t25['turn'][0]:.4f} -> {t10['turn'][0]:.4f}.")
    if t10["turn"][0] < 0.85 * t25["turn"][0]:
        print("  >>> Guidance itself curves the trajectory substantially. That is a finding worth a")
        print("      sentence in its own right: strong classifier-free guidance is one source of the")
        print("      curvature we measure, and part of the few-step penalty is the price of guidance.")
if t10 and stu:
    lo = [n for n in NFES if n <= 4]
    print(f"\n  Decomposition at matched guidance (g=1.0):")
    for n in NFES:
        a = t25['fid'][str(n)][0] if t25 else float('nan')
        b = t10['fid'][str(n)][0]; c = stu['fid'][str(n)][0]
        print(f"    n={n:<3} teacher@{GUIDES[-1]} {a:.4f} | teacher@1.0 {b:.4f} | reflow@1.0 {c:.4f}"
              f"   guidance {a-b:+.4f}, straightening {b-c:+.4f}")
    gain_str = np.mean([t10['fid'][str(n)][0] - stu['fid'][str(n)][0] for n in lo])
    gain_gd  = np.mean([(t25['fid'][str(n)][0] if t25 else np.nan) - t10['fid'][str(n)][0] for n in lo])
    print(f"\n  Averaged over n <= 4: guidance accounts for {gain_gd:+.4f} FID, straightening for "
          f"{gain_str:+.4f}.")
    if gain_str > 2 * abs(gain_gd):
        print("  >>> Straightening carries the effect. The reflow result stands as a straightening")
        print("      result and can be reported as such, with this control shown alongside.")
    elif gain_str < 0.5 * abs(gain_gd):
        print("  >>> Most of the apparent improvement was the guidance scale, not the rectification.")
        print("      The headline must be revised: reflow's contribution is smaller than the raw")
        print("      teacher-vs-student comparison suggests, and we should say so plainly.")
    else:
        print("  >>> Both contribute materially. Report the decomposition rather than a single")
        print("      number, and attribute the improvement to the pair of changes.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), "guidance_control.json")
json.dump(out, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.finish()
print("Guidance control done.")
