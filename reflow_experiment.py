#!/usr/bin/env python
# REFLOW / TRAJECTORY RECTIFICATION on the latent LFM base — overnight job, three phases.
#
# Reflow (Liu, Gong & Liu, arXiv:2209.03003) retrains the velocity field on the model's OWN
# noise->sample pairs. Because each pair is joined by a straight line, the retrained field is
# straighter and can be integrated in very few steps. "Improving Training of Rectified Flows"
# (arXiv:2405.20320) reports that a single reflow round is enough, which is what we do here.
#
#   PHASE 1  generate N pairs (Z0, Z1) with the frozen TEACHER at guidance 2.5, 50 Euler steps.
#            The pairs are latent-space endpoints, not decoded motions, since the student is a
#            latent flow model. Shards are cached on disk so a restarted job skips this phase.
#   PHASE 2  warm-start a student from the teacher and train the flow-matching loss on those pairs.
#            Resumable; checkpoints every SAVE_EVERY steps.
#   PHASE 3  full assessment: NFE in {1,2,4,8,16,50} x {teacher, student} x {no projection,
#            post-decode projection}, reporting FID, R@3, BLE, FSR, plus a straightness measure.
#
# WHY PHASE 3 IS SHAPED THIS WAY. The result this project can use is not "reflow is faster" -- that
# is known. It is whether POST-DECODE PROJECTION SURVIVES SAMPLING ACCELERATION: if the student at
# 1-4 NFE still reaches BLE = 0 at no FID cost, the paper earns one sentence saying the correction is
# sampler-agnostic and free, which composes with the main claim. The sweep is built to answer exactly
# that, so the night produces something usable either way.
#
# CRITICAL -- CFG IS BAKED IN. Pairs come from the teacher WITH classifier-free guidance applied, so
# the student learns the already-guided field and MUST be sampled at guidance 1.0. Sampling it at 2.5
# applies guidance twice; FID explodes and it looks as though reflow failed. This script forces
# guidance 1.0 for the student everywhere and prints a reminder.
#
# SAFETY: writes ONLY reflow_*.pt. It never touches latent_*.pt / direct_*.pt.
#
#   sbatch run_reflow.sh
#   REFLOW_PAIRS=20000 REFLOW_STEPS=40000 sbatch run_reflow.sh

import os, sys, time, json, math, random
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, torch.nn.functional as F, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[reflow] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[reflow] models loaded.")

DEVICE = M.DEVICE; ODE = M.ODE_STEPS; GUID = M.GUIDANCE; T5_MAXLEN = M.T5_MAXLEN
UNIT = M.UNIT_LEN; MAXLEN = M.MAX_MOTION_LEN; CK = M.CK
_cfg = M._cfg; _timesteps = M._timesteps; embed_text = M.embed_text
eval_variant = M.eval_variant; load_net = M.load_net; _agg = M._agg
safe_save = M.safe_save; lr_sched = M.lr_sched; EMAh = M.EMAh; quick_fid = M.quick_fid

N_PAIRS    = int(os.environ.get("REFLOW_PAIRS", "20000"))
STEPS      = int(os.environ.get("REFLOW_STEPS", "40000"))
BS         = int(os.environ.get("REFLOW_BS", "64"))
LR         = float(os.environ.get("REFLOW_LR", "5e-5"))     # lower than 2e-4: we are refining, not training
GEN_BS     = int(os.environ.get("REFLOW_GEN_BS", "128"))
SAVE_EVERY = int(os.environ.get("REFLOW_SAVE_EVERY", "2000"))
EVAL_EVERY = int(os.environ.get("REFLOW_EVAL_EVERY", "5000"))
NFES       = [int(x) for x in os.environ.get("REFLOW_NFE", "1,2,4,8,16,50").split(",")]
WORK_DIR   = os.environ.get("WORK_DIR", ".")
PAIR_DIR   = os.path.join(WORK_DIR, "reflow_pairs")
TAG        = "reflow"                                        # never "latent"
assert TAG not in ("latent", "direct", "latent_pen", "direct_pen"), "refusing to clobber a trained base"
os.makedirs(PAIR_DIR, exist_ok=True)

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", "reflow_rectification"), job_type="training")

teacher = load_net("latent", True)          # also restores z_mean_t / z_std_t into the module
for p in teacher.parameters(): p.requires_grad_(False)
teacher.eval()
print(f"[reflow] teacher loaded: Tlen={teacher.Tlen} cd={teacher.cd}")

def clip_len(mot):
    return min((len(mot) // UNIT) * UNIT, MAXLEN)

# ===================================================================== PHASE 1: pair generation
@torch.no_grad()
def teacher_pairs(tseq, tmask, tpool, lens, n=ODE, guidance=GUID):
    """Integrate the teacher and return (Z0, Z1) latent endpoints of the SAME trajectory."""
    B = tpool.shape[0]
    ns = teacher.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = teacher.null_pool.unsqueeze(0).expand(B, -1)
    z0 = torch.randn(B, teacher.Tlen, teacher.cd, device=DEVICE)
    z = z0.clone()
    ts = _timesteps(n, "linear")
    for i in range(n):
        t = torch.full((B,), float(ts[i]), device=DEVICE)
        v = _cfg(teacher, z, t, tseq, tmask, tpool, lens, ns, nm, npl, guidance, 0.0)
        z = z + float(ts[i + 1] - ts[i]) * v
    return z0, z

shards = sorted([f for f in os.listdir(PAIR_DIR) if f.startswith("shard_") and f.endswith(".pt")])
have = sum(torch.load(os.path.join(PAIR_DIR, f), map_location="cpu")["z0"].shape[0] for f in shards) if shards else 0
print(f"\n{'='*92}\nPHASE 1 — pair generation ({have}/{N_PAIRS} already cached)\n{'='*92}")

if have < N_PAIRS:
    random.seed(1234)
    pool = list(range(len(M.train_entries)))
    t0 = time.time(); made = have; sh = len(shards)
    while made < N_PAIRS:
        ids = [random.choice(pool) for _ in range(GEN_BS)]
        caps = [M.train_entries[i]["texts"][0] for i in ids]
        lens = torch.tensor([clip_len(M.train_entries[i]["motion"]) for i in ids], device=DEVICE)
        ts_, tm_, tp_ = embed_text(caps)
        z0, z1 = teacher_pairs(torch.tensor(ts_, device=DEVICE), torch.tensor(tm_, device=DEVICE),
                               torch.tensor(tp_, device=DEVICE), lens)
        torch.save(dict(z0=z0.half().cpu(), z1=z1.half().cpu(),
                        ids=torch.tensor(ids), lens=lens.cpu()),
                   os.path.join(PAIR_DIR, f"shard_{sh:05d}.pt"))
        sh += 1; made += GEN_BS
        if sh % 10 == 0:
            el = time.time() - t0
            print(f"  {made}/{N_PAIRS} pairs  ({el/60:.1f}m, eta {el/max(made-have,1)*(N_PAIRS-made)/60:.1f}m)")
    print(f"  pair generation done in {(time.time()-t0)/60:.1f}m")
    shards = sorted([f for f in os.listdir(PAIR_DIR) if f.startswith("shard_") and f.endswith(".pt")])

print("[reflow] loading pairs into memory ...")
Z0, Z1, IDS, LENS = [], [], [], []
for f in shards:
    d = torch.load(os.path.join(PAIR_DIR, f), map_location="cpu")
    Z0.append(d["z0"]); Z1.append(d["z1"]); IDS.append(d["ids"]); LENS.append(d["lens"])
Z0 = torch.cat(Z0); Z1 = torch.cat(Z1); IDS = torch.cat(IDS); LENS = torch.cat(LENS)
NP_ = Z0.shape[0]
print(f"[reflow] {NP_} pairs, Z0 {tuple(Z0.shape)} {Z0.dtype}")

# text embeddings for the captions used (cache once; re-embedding every step is wasteful)
uniq = sorted(set(IDS.tolist()))
remap = {g: k for k, g in enumerate(uniq)}
print(f"[reflow] embedding {len(uniq)} unique captions ...")
cs, cm, cp = embed_text([M.train_entries[i]["texts"][0] for i in uniq])
CS = torch.tensor(cs); CM = torch.tensor(cm); CP = torch.tensor(cp)
CIDX = torch.tensor([remap[i] for i in IDS.tolist()])

# straightness of the teacher pairs, for reference
with torch.no_grad():
    d = (Z1[:2048].float() - Z0[:2048].float())
    print(f"[reflow] mean pair displacement norm {d.flatten(1).norm(dim=1).mean():.3f}")

def make_g1_sampler(orig):
    """Force guidance 1.0: the student's field already contains classifier-free guidance."""
    def s(net, is_latent, tseq, tmask, tpool, length, n=ODE, guidance=GUID, mode="none", seed=None,
          guide_w=None, cfg_rescale=0.0, schedule="linear", solver="euler", gw_schedule="const"):
        return orig(net, is_latent, tseq, tmask, tpool, length, n=n, guidance=1.0, mode=mode, seed=seed,
                    guide_w=guide_w, cfg_rescale=cfg_rescale, schedule=schedule, solver=solver,
                    gw_schedule=gw_schedule)
    return s

# ===================================================================== PHASE 2: student training
best_p = os.path.join(CK, f"{TAG}_best.pt"); latest_p = os.path.join(CK, f"{TAG}_latest.pt")
student = load_net("latent", True)                       # warm start from the teacher
student.train()
opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
ema = EMAh(student, M.EMA); st = 0; best = float("inf")
if os.path.exists(latest_p):
    r = torch.load(latest_p, map_location=DEVICE, weights_only=False)
    student.load_state_dict(r["net"]); ema.shadow = {k: v.to(DEVICE) for k, v in r["ema"].items()}
    try: opt.load_state_dict(r["opt"])
    except Exception: pass
    st = int(r.get("step", 0)); best = float(r.get("best", float("inf")))
    print(f"[reflow] resuming student at step {st}")

zm = M.z_mean_t.detach().cpu().numpy(); zs = M.z_std_t.detach().cpu().numpy()
def save_latest():
    safe_save(dict(net=student.state_dict(), ema={k: v.clone() for k, v in ema.shadow.items()},
                   opt=opt.state_dict(), step=st, best=best, z_mean=zm, z_std=zs), latest_p)
def save_best(metric):
    global best; best = metric
    safe_save(dict(state={k: v.clone() for k, v in ema.shadow.items()}, step=st, metric=metric,
                   z_mean=zm, z_std=zs), best_p)

print(f"\n{'='*92}\nPHASE 2 — student training {st} -> {STEPS} (lr={LR}, bs={BS})\n{'='*92}")
t0 = time.time()
while st < STEPS:
    idx = torch.randint(0, NP_, (BS,))
    z0 = Z0[idx].to(DEVICE).float(); z1 = Z1[idx].to(DEVICE).float()
    ci = CIDX[idx]
    tseq = CS[ci].to(DEVICE); tmask = CM[ci].to(DEVICE); tpool = CP[ci].to(DEVICE)
    L = LENS[idx].to(DEVICE)
    for pg in opt.param_groups: pg["lr"] = lr_sched(st, 500, LR, LR * 0.1, STEPS)
    t = torch.rand(BS, device=DEVICE)
    zt = (1 - t).view(-1, 1, 1) * z0 + t.view(-1, 1, 1) * z1
    # NOTE: no CFG dropout. The pairs already encode the guided field, so the student is trained
    # conditionally only and is sampled at guidance 1.0.
    v = student(zt, t, tseq, tmask, tpool, L)
    loss = F.mse_loss(v, z1 - z0)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), M.GRAD_CLIP)
    opt.step(); ema.update(student); st += 1
    if st % 200 == 0:
        print(f"  [reflow] {st:>6} loss={loss.item():.4f}  {(time.time()-t0)/60:.1f}m")
        wandb.log({"reflow/loss": loss.item(), "reflow/lr": opt.param_groups[0]["lr"], "reflow/step": st})
    if st % SAVE_EVERY == 0: save_latest()
    if st % EVAL_EVERY == 0 or st == STEPS:
        bk = {k: v.detach().clone() for k, v in student.state_dict().items()}
        student.load_state_dict({k: v.to(DEVICE) for k, v in ema.shadow.items()}); student.eval()
        orig = M.sample
        M.sample = make_g1_sampler(orig)          # student is sampled at guidance 1.0
        try:
            fid = quick_fid(student, True)
        finally:
            M.sample = orig
        star = ""
        if fid < best: save_best(fid); star = "  <-BEST"
        print(f"    [reflow eval {st}] quick FID={fid:.4f}{star}  (student sampled at guidance 1.0)")
        wandb.log({"reflow/quick_FID": fid, "reflow/step": st})
        student.load_state_dict(bk); student.train(); save_latest()
save_latest()
if not os.path.exists(best_p): save_best(float("inf"))
print(f"[reflow] training done ({(time.time()-t0)/60:.1f}m)")

# ===================================================================== PHASE 3: assessment
student.load_state_dict({k: v.to(DEVICE) for k, v in ema.shadow.items()}); student.eval()

@torch.no_grad()
def straightness(net, guidance, n=16, nb=256):
    """E_t || v(Z_t,t) - (Z1-Z0) ||^2 along the model's own trajectory. 0 = perfectly straight."""
    ids = [random.choice(range(len(M.train_entries))) for _ in range(nb)]
    caps = [M.train_entries[i]["texts"][0] for i in ids]
    lens = torch.tensor([clip_len(M.train_entries[i]["motion"]) for i in ids], device=DEVICE)
    a, b, c = embed_text(caps)
    tseq = torch.tensor(a, device=DEVICE); tmask = torch.tensor(b, device=DEVICE); tpool = torch.tensor(c, device=DEVICE)
    B = nb
    ns = net.null_seq.unsqueeze(0).expand(B, -1, -1)
    nm = torch.ones(B, T5_MAXLEN, dtype=torch.bool, device=DEVICE)
    npl = net.null_pool.unsqueeze(0).expand(B, -1)
    torch.manual_seed(7)
    z0 = torch.randn(B, net.Tlen, net.cd, device=DEVICE); z = z0.clone()
    ts = _timesteps(n, "linear"); vs = []
    for i in range(n):
        t = torch.full((B,), float(ts[i]), device=DEVICE)
        v = _cfg(net, z, t, tseq, tmask, tpool, lens, ns, nm, npl, guidance, 0.0)
        vs.append(v); z = z + float(ts[i + 1] - ts[i]) * v
    u = (z - z0).unsqueeze(0)
    return float(torch.stack(vs).sub(u).pow(2).mean())

print(f"\n{'='*100}\nPHASE 3 — assessment. NFE sweep, teacher (guidance {GUID}) vs reflow student (guidance 1.0)\n{'='*100}")
s_t = straightness(teacher, GUID); s_s = straightness(student, 1.0)
print(f"  straightness (lower = straighter):  teacher {s_t:.5f}   student {s_s:.5f}   "
      f"({'IMPROVED' if s_s < s_t else 'NOT improved'})")

rows = []
orig_sample = M.sample
for name, net, g in [("teacher", teacher, GUID), ("reflow", student, 1.0)]:
    M.sample = orig_sample if g == GUID else make_g1_sampler(orig_sample)
    try:
        for nfe in NFES:
            for mode, mlab in [("none", "unconstrained"), ("posthoc", "post-decode proj")]:
                t0 = time.time()
                r = eval_variant(net, True, mode, n_steps=nfe)
                fm, fp, _ = _agg(r["fsr"]); bm, _, bx = _agg(r["ble"])
                row = dict(model=name, nfe=nfe, mode=mlab, FID=round(float(r["fid"]), 4),
                           R3=round(float(r["R3"]), 4), BLE=round(bm, 5), BLE_max=round(bx, 5),
                           FSR=round(fm, 4), sec=round(time.time() - t0))
                rows.append(row)
                print(f"  {name:<8} NFE={nfe:<3} {mlab:<18} FID={row['FID']:<9} R@3={row['R3']:<7} "
                      f"BLE={row['BLE']:<9} FSR={row['FSR']:<8} ({row['sec']}s)")
                wandb.log({f"reflow_eval/{name}/{mlab}/FID": row["FID"],
                           f"reflow_eval/{name}/{mlab}/BLE": row["BLE"], "reflow_eval/nfe": nfe})
    finally:
        M.sample = orig_sample

print(f"\n{'='*100}")
print(f" {'model':<9}{'NFE':>5}{'treatment':>20}{'FID':>10}{'R@3':>8}{'BLE':>10}{'FSR':>9}")
print("-" * 100)
for r in rows:
    print(f" {r['model']:<9}{r['nfe']:>5}{r['mode']:>20}{r['FID']:>10}{r['R3']:>8}{r['BLE']:>10}{r['FSR']:>9}")
print("=" * 100)

def get(m, n, md):
    x = [r for r in rows if r["model"] == m and r["nfe"] == n and r["mode"] == md]
    return x[0] if x else None
t50 = get("teacher", 50, "unconstrained")
print("\nREADING THE RESULT")
if t50:
    for n in NFES:
        a, b = get("teacher", n, "unconstrained"), get("reflow", n, "unconstrained")
        if a and b:
            print(f"  NFE={n:<3} teacher {a['FID']:<9} reflow {b['FID']:<9} "
                  f"({'reflow better' if b['FID'] < a['FID'] else 'teacher better'})")
    low = [n for n in NFES if n <= 4]
    ok = [get("reflow", n, "post-decode proj") for n in low]
    ok = [r for r in ok if r]
    if ok and all(r["BLE"] == 0.0 for r in ok):
        print("\n  Post-decode projection reaches BLE = 0 at every low-NFE setting, as it must: it acts")
        print("  on the decoded output and is indifferent to how the sample was produced. The usable")
        print("  sentence for the paper is that the correction is sampler-agnostic and free even under")
        print(f"  {min(low)}-step sampling, PROVIDED the FID column above supports it.")
print("\n  REMINDER: the student is sampled at guidance 1.0 because the pairs were generated with CFG")
print("  already applied. If its FID looks catastrophic, check that first before concluding anything.")
print("\n  Also honest: this is one reflow round on 20k self-generated pairs, warm-started, with a short")
print("  schedule. A negative result here is weak evidence about reflow in general and should not be")
print("  written up as one.")

dst = os.path.join(WORK_DIR, "reflow_assessment.json")
json.dump(dict(pairs=NP_, steps=STEPS, lr=LR, straightness=dict(teacher=s_t, student=s_s), rows=rows),
          open(dst, "w"), indent=2)
print(f"\nraw results -> {dst}")
wandb.log({"reflow_table": wandb.Table(
    columns=["model", "NFE", "treatment", "FID", "R@3", "BLE", "BLE_max", "FSR"],
    data=[[r["model"], r["nfe"], r["mode"], r["FID"], r["R3"], r["BLE"], r["BLE_max"], r["FSR"]] for r in rows])})
wandb.finish()
print("Reflow experiment done.")
