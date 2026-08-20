#!/usr/bin/env python
# A2 — BEST-OF-N NOISE SELECTION, with the control that decides whether any gain is real.
#
# Draw N initial noises per prompt, decode all N, keep the one minimising a cheap proxy. This is
# inference-time scaling: no training, cost linear in N.
#
# TWO THINGS THAT MAKE THIS EASY TO GET WRONG, both handled here:
#
# 1. THE PROXY MUST NOT USE GROUND TRUTH. The obvious proxy -- pick the candidate whose evaluator
#    feature is closest to the real motion of that clip -- would select on the very quantity FID
#    measures, using test labels. It would produce a spectacular, meaningless FID. We use only
#    proxies computable at inference: pre-projection BLE, and foot-skate rate. Your own composition
#    experiment already showed BLE is a weak quality proxy (v1 blending had low BLE and was visibly
#    incoherent), so treat a BLE-selected win with suspicion until the diversity column agrees.
#
# 2. FID IS DIVERSITY-SENSITIVE. Selecting the "best" of N systematically narrows the output
#    distribution, and FID punishes that. A best-of-N run can improve per-sample plausibility while
#    making FID WORSE. We therefore report Diversity (mean pairwise distance between generated
#    evaluator features, the standard HumanML3D definition) for every N, plus a RANDOM-PICK control
#    that draws N candidates and keeps a random one. The random control has the same noise statistics
#    as best-of-N but no selection, so:
#         best-of-N better than random control  ->  the selection is doing the work
#         best-of-N same as random control      ->  you are measuring seed luck, not selection
#
#   sbatch run_boost_bestofn.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[bestofn] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[bestofn] models loaded.")

DEVICE = M.DEVICE; MAXLEN = M.MAX_MOTION_LEN; EVAL_N = M.EVAL_N
sample = M.sample; load_net = M.load_net; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; ble_pc_joints = M.ble_pc_joints; fsr_pc = M.fsr_pc
lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
project_joints = M.project_joints; _joints_to_norm = M._joints_to_norm

BASE = os.environ.get("BOOST_BASE", "latent"); IS_LAT = (BASE in ("latent", "latent_pen"))
NS = [int(x) for x in os.environ.get("BOOST_N", "1,2,4,8").split(",")]
PROXIES = os.environ.get("BOOST_PROXY", "ble,fsr,random").split(",")
N_EVAL = int(os.environ.get("EVAL_N", str(EVAL_N)))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", f"boost_bestofn_{BASE}"), job_type="ablation")

net = load_net(BASE, IS_LAT)
rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

def diversity(feat, n_pairs=300, seed=0):
    """Standard HumanML3D Diversity: mean L2 between randomly paired generated features."""
    r = np.random.default_rng(seed); n = len(feat)
    if n < 2 * n_pairs: n_pairs = n // 2
    a = r.permutation(n)[:n_pairs]; b = r.permutation(n)[:n_pairs]
    return float(np.linalg.norm(feat[a] - feat[b], axis=1).mean())

@torch.no_grad()
def run(N, proxy, project):
    fsr_all, ble_all, mf, real_mf = [], [], [], []
    t0 = time.time()
    for s in range(0, N_EVAL, 32):
        e = min(s + 32, N_EVAL)
        ts = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
        tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
        gm = lengths_to_mask(L, MAXLEN)
        cands, scores = [], []
        for k in range(N):
            x = sample(net, IS_LAT, ts, tm, tp, L, seed=s + k * 7919)   # distinct noise per candidate
            J = _gj(x)
            cands.append(x)
            if proxy == "ble":     scores.append(ble_pc_joints(J, L))
            elif proxy == "fsr":   scores.append(fsr_pc(J, L))
            else:                  scores.append(np.random.default_rng(s + k).random(len(L)))
        S = np.stack(scores)                                  # (N, B)
        pick = S.argmin(axis=0) if proxy != "random" else np.random.default_rng(s).integers(0, N, len(L))
        x = torch.stack([cands[int(pick[j])][j] for j in range(len(L))])
        if project:
            x = _joints_to_norm(project_joints(_gj(x), L), x)
        J = _gj(x)
        fsr_all.append(fsr_pc(J, L)); ble_all.append(ble_pc_joints(J, L))
        mf.append(memb(x * gm[..., None], L))
        rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
        real_mf.append(memb(rm * gm[..., None], L))
    G = np.concatenate(mf, 0); R = np.concatenate(real_mf, 0)
    return dict(FID=round(float(fid_calc(G, R)), 4), R3=round(float(rprec(G, R)[3]), 4),
                BLE=round(float(np.concatenate(ble_all).mean()), 5),
                FSR=round(float(np.concatenate(fsr_all).mean()), 4),
                DIV=round(diversity(G), 3), DIV_real=round(diversity(R), 3),
                sec=round(time.time() - t0))

rows = []
print(f"\n{'='*104}\nA2 — BEST-OF-N  (base={BASE}, {N_EVAL} clips)\n{'='*104}")
print(f" {'proxy':<10}{'N':>4}{'proj':>7}{'FID':>10}{'R@3':>8}{'BLE':>10}{'FSR':>9}{'Diversity':>12}{'sec':>7}")
print("-" * 104)
for proxy in PROXIES:
    for N in NS:
        if N == 1 and proxy != PROXIES[0]: continue           # N=1 is the same run for every proxy
        for project in (False, True):
            r = run(N, proxy, project)
            r.update(proxy=proxy, N=N, project=project); rows.append(r)
            print(f" {proxy:<10}{N:>4}{str(project):>7}{r['FID']:>10}{r['R3']:>8}{r['BLE']:>10}"
                  f"{r['FSR']:>9}{r['DIV']:>12}{r['sec']:>7}")
            wandb.log({f"bestofn/{proxy}/N{N}/proj{int(project)}/FID": r["FID"],
                       f"bestofn/{proxy}/N{N}/proj{int(project)}/DIV": r["DIV"]})
print("=" * 104)
if rows: print(f"\n Real-motion Diversity reference: {rows[0]['DIV_real']}")

def get(p, n, pr):
    x = [r for r in rows if r["proxy"] == p and r["N"] == n and r["project"] == pr]
    return x[0] if x else None
base = get(PROXIES[0], 1, False)
print("\nREADING THE RESULT")
for proxy in PROXIES:
    for N in NS:
        if N == 1: continue
        a, c = get(proxy, N, False), get("random", N, False)
        if not a: continue
        line = f"  {proxy} N={N}: FID {a['FID']} (baseline {base['FID']}), Diversity {a['DIV']} (baseline {base['DIV']})"
        if c and proxy != "random":
            line += f", random-pick control {c['FID']}"
        print(line)
print("\n  Decide with BOTH columns. A lower FID together with a Diversity close to baseline is a real")
print("  gain. A lower FID with collapsed Diversity is mode-narrowing and must not be reported as an")
print("  improvement. And if the proxy runs match the random-pick control, nothing was selected for --")
print("  you are seeing seed variance, which your 20-seed study already showed spans 0.140-0.186.")
print("\n  Cost note: best-of-N multiplies inference by N. If you report it, report the compute too.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), f"boost_bestofn_{BASE}.json")
json.dump(rows, open(dst, "w"), indent=2); print(f"\nraw results -> {dst}")
wandb.log({"bestofn_table": wandb.Table(
    columns=["proxy", "N", "projected", "FID", "R@3", "BLE", "FSR", "Diversity"],
    data=[[r["proxy"], r["N"], r["project"], r["FID"], r["R3"], r["BLE"], r["FSR"], r["DIV"]] for r in rows])})
wandb.finish()
print("A2 sweep done.")
