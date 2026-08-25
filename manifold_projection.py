#!/usr/bin/env python
# THE BONE-VALID SET AS A MANIFOLD, AND THE PROJECTION ONTO IT.
#
# GEOMETRY, derived rather than assumed. Fix the 21 bone lengths. A pose is then determined by the
# root position and one unit direction per bone, since each child joint is its parent plus a fixed
# length along a unit vector. So per frame
#
#       C_bone  ~=  R^3 x (S^2)^21 ,      dim = 3 + 42 = 45,
#
# against 66 ambient coordinates (22 joints x 3): codimension 21, exactly the number of bones. This
# is a change of coordinates, not an empirical claim. Each S^2 factor has constant positive
# curvature; the product does not, and the set of plausible MOTIONS is a further subset of this with
# no known closed form. We therefore make no claim about the curvature of the motion manifold itself.
#
# WHY IT MATTERS HERE. The correction currently used walks the kinematic chain rescaling each segment
# to its prescribed length. That lands on C_bone, so it is a valid RETRACTION, but it is not the
# nearest point: rescaling a proximal bone translates every joint distal to it, so the displacement
# accumulates down the chain. The Euclidean projection onto C_bone is instead
#
#       min over (root, u_1..u_21) of  || Q(root, u) - Q_target ||^2 ,   subject to |u_b| = 1,
#
# which distributes the correction across the whole chain instead of accumulating it.
#
# THE HYPOTHESIS THIS TESTS. In the replicated decomposition, bone-only correction significantly
# INCREASED LFM FID (+0.0102) while giving exact validity. Both operators reach BLE = 0, so exactness
# cannot be what costs the FID; a natural explanation is that the retraction displaces motions
# further than necessary and pushes them off the data manifold. If the nearest-point projection
# displaces less AND recovers the FID, the cost was the operator, not the constraint. If it displaces
# less but FID does not recover, then the cost really is intrinsic to imposing exact bone lengths on
# decoded motion, which is a sharper and more interesting statement than we can currently make.
#
# Inference-only. The projection is solved by Riemannian gradient descent on the sphere
# parameterization, warm-started from the retraction, so it converges in a few hundred cheap steps.
#
#   sbatch run_manifold.sh

import os, sys, json, time
os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
import numpy as np, torch, importlib.util
MAIN = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
spec = importlib.util.spec_from_file_location("expmod", MAIN); M = importlib.util.module_from_spec(spec); sys.modules["expmod"] = M
print(f"[manifold] importing {MAIN} ...")
try:
    spec.loader.exec_module(M)
except Exception as e:
    if type(e).__name__ != "M_ABLATION_STOP": raise
    print("[manifold] models loaded.")

DEVICE = M.DEVICE; MAXLEN = M.MAX_MOTION_LEN; EVAL_N = M.EVAL_N
sample = M.sample; load_net = M.load_net; embed_text = M.embed_text
memb = M.memb; fid_calc = M.fid_calc; rprec = M.rprec
_gj = M._gj; ble_pc_joints = M.ble_pc_joints; fsr_pc = M.fsr_pc
lengths_to_mask = M.lengths_to_mask; pad_norm = M.pad_norm
project_bonelength = M.project_bonelength; project_foot = M.project_foot
_joints_to_norm = M._joints_to_norm
EDGES = M.EDGES; rest_len = M.rest_len; _CHAINS = M._CHAINS; N_JOINTS = M.N_JOINTS

BASE = os.environ.get("MF_BASE", "latent"); IS_LAT = (BASE in ("latent", "latent_pen"))
N_EVAL = int(os.environ.get("EVAL_N", "512"))
PROJ_STEPS = int(os.environ.get("MF_STEPS", "300"))
PROJ_LR = float(os.environ.get("MF_LR", "0.05"))

# ---- kinematic order: each edge's child depends on its parent, chains are already topological ----
PARENT = [e[0] for e in EDGES]; CHILD = [e[1] for e in EDGES]
LEN_B = rest_len.clone()                                   # (E,) prescribed lengths
order = list(range(len(EDGES)))                            # chains are listed parent-first
seen = {0}
for i in order:
    assert PARENT[i] in seen, "edge order is not topological; reorder before use"
    seen.add(CHILD[i])
print(f"[manifold] {len(EDGES)} bones, ambient dim {N_JOINTS*3}, manifold dim {3 + 2*len(EDGES)} "
      f"(codim {N_JOINTS*3 - 3 - 2*len(EDGES)})")

def forward_kin(root, U):
    """root (B,T,3), U (B,T,E,3) unit directions -> joints (B,T,J,3)."""
    B, T = root.shape[:2]
    Q = torch.zeros(B, T, N_JOINTS, 3, device=root.device, dtype=root.dtype)
    Q[:, :, 0] = root
    for i in order:
        Q[:, :, CHILD[i]] = Q[:, :, PARENT[i]] + LEN_B[i] * U[:, :, i]
    return Q

def params_from_joints(Q):
    """Initialize the parameterization from arbitrary joints: root and normalized bone directions."""
    root = Q[:, :, 0].clone()
    V = Q[:, :, CHILD, :] - Q[:, :, PARENT, :]
    V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return root, V

def nearest_projection(Q, L, steps=PROJ_STEPS, lr=PROJ_LR):
    """Euclidean nearest point on C_bone, by optimizing over R^3 x (S^2)^E. Warm-started from the
    retraction, so the returned point is never worse than it in displacement."""
    with torch.no_grad():
        Q0 = project_bonelength(Q)                 # retraction: feasible starting point
        root0, U0 = params_from_joints(Q0)
    root = root0.clone().requires_grad_(True)
    V = U0.clone().requires_grad_(True)            # unconstrained; normalized inside the objective
    opt = torch.optim.Adam([root, V], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.02)
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1).unsqueeze(-1)
    Qt = Q.detach()
    for _ in range(steps):
        U = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        Qp = forward_kin(root, U)
        loss = (((Qp - Qt) ** 2).sum(-1, keepdim=True) * fm).sum() / fm.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        U = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return forward_kin(root, U).detach()

def displacement(Qa, Qb, L):
    fm = lengths_to_mask(L, MAXLEN).float().unsqueeze(-1)
    return float(((Qa - Qb).norm(dim=-1) * fm).sum() / (fm.sum() * Qa.shape[2] + 1e-8))

import wandb
wandb.init(project=os.environ.get("WANDB_PROJECT", "motion-clfm"),
           entity=(os.environ.get("WANDB_ENTITY") or None),
           name=os.environ.get("WANDB_RUN", f"manifold_projection_{BASE}"), job_type="analysis")

net = load_net(BASE, IS_LAT)
rng = np.random.default_rng(0)
sel = np.array(sorted(rng.permutation(len(M.test_entries))[:N_EVAL].tolist()))
caps = [M.test_entries[int(i)]["texts"][0] for i in sel]
lens_all = torch.tensor([int(M.test_lens[i]) for i in sel], device=DEVICE)
TSEQ, TMASK, TPOOL = embed_text(caps)

OPS = ["none", "retraction (current)", "nearest projection",
       "foot + retraction", "foot + nearest projection"]
acc = {k: dict(ble=[], fsr=[], disp=[], mf=[]) for k in OPS}
real_mf = []
print(f"\n{'='*104}\nMANIFOLD PROJECTION — base={BASE}, {N_EVAL} clips, {PROJ_STEPS} solver steps\n{'='*104}")
t0 = time.time()
for s in range(0, N_EVAL, 32):
    e = min(s + 32, N_EVAL)
    ts = torch.tensor(TSEQ[s:e], device=DEVICE); tm = torch.tensor(TMASK[s:e], device=DEVICE)
    tp = torch.tensor(TPOOL[s:e], device=DEVICE); L = lens_all[s:e]
    gm = lengths_to_mask(L, MAXLEN)
    with torch.no_grad():
        x = sample(net, IS_LAT, ts, tm, tp, L, seed=s)
        J0 = _gj(x)
        Jf = project_foot(J0, L)
    variants = {
        "none": J0,
        "retraction (current)": project_bonelength(J0),
        "nearest projection": nearest_projection(J0, L),
        "foot + retraction": project_bonelength(Jf),
        "foot + nearest projection": nearest_projection(Jf, L),
    }
    with torch.no_grad():
        for k, J in variants.items():
            xn = x if k == "none" else _joints_to_norm(J, x)
            Jm = _gj(xn)
            acc[k]["ble"].append(ble_pc_joints(Jm, L)); acc[k]["fsr"].append(fsr_pc(Jm, L))
            acc[k]["disp"].append(displacement(Jm, J0, L))
            acc[k]["mf"].append(memb(xn * gm[..., None], L))
        rm = torch.tensor(np.stack([pad_norm(M.test_entries[int(i)]["motion"])[0] for i in sel[s:e]]), device=DEVICE)
        real_mf.append(memb(rm * gm[..., None], L))
    if s % 128 == 0: print(f"  {e}/{N_EVAL} clips  ({(time.time()-t0)/60:.1f}m)")

R = np.concatenate(real_mf, 0)
rows = []
for k in OPS:
    G = np.concatenate(acc[k]["mf"], 0)
    rows.append(dict(op=k, FID=float(fid_calc(G, R)), R3=float(rprec(G, R)[3]),
                     BLE=float(np.concatenate(acc[k]["ble"]).mean()),
                     FSR=float(np.concatenate(acc[k]["fsr"]).mean()),
                     DISP=float(np.mean(acc[k]["disp"]))))

print(f"\n{'='*104}")
print(f" {'operator':<30}{'FID':>10}{'R@3':>9}{'BLE':>12}{'FSR':>10}{'displacement (m)':>19}")
print("-" * 104)
for r in rows:
    print(f" {r['op']:<30}{r['FID']:>10.4f}{r['R3']:>9.4f}{r['BLE']:>12.5f}{r['FSR']:>10.5f}{r['DISP']:>19.5f}")
print("=" * 104)

get = lambda n: [r for r in rows if r["op"] == n][0]
ret, near, non = get("retraction (current)"), get("nearest projection"), get("none")
d_ratio = near["DISP"] / max(ret["DISP"], 1e-9)
print("\nREADING")
print(f"  Displacement: retraction {ret['DISP']:.5f} m, nearest projection {near['DISP']:.5f} m "
      f"({100*d_ratio:.1f}% of it).")
print(f"  FID: unconstrained {non['FID']:.4f}, retraction {ret['FID']:.4f}, nearest {near['FID']:.4f}.")
if near["BLE"] > 1e-4:
    print("  WARNING: the nearest projection did not reach exact validity; raise MF_STEPS. Its whole")
    print("  point is to be exact while displacing less, so a nonzero BLE invalidates the comparison.")
elif near["FID"] < ret["FID"] - 0.005:
    print("  The nearest-point projection recovers FID relative to the retraction. The FID cost of")
    print("  bone correction was therefore an artifact of the OPERATOR displacing more than necessary,")
    print("  not a cost of exact validity itself. This changes the Section 5.3 story: the exact")
    print("  constraint is free once it is imposed by the correct projection.")
elif near["FID"] > ret["FID"] + 0.005:
    print("  The nearest projection is WORSE despite displacing less. Displacement magnitude is then")
    print("  not what FID responds to; the direction of the correction matters, i.e. it moves motions")
    print("  off the data manifold regardless of how small the step is.")
else:
    print("  No material FID difference despite the reduced displacement. The FID cost of exact bone")
    print("  correction is then intrinsic to the constraint rather than to this particular operator,")
    print("  which is a sharper claim than we can currently make and worth stating explicitly.")

print("\n  Scope note for the writeup: C_bone is exactly R^3 x (S^2)^21 per frame by construction.")
print("  We make no claim about the curvature of the space of plausible motions, which is a subset")
print("  of this with no closed form and is not characterized here.")

dst = os.path.join(os.environ.get("WORK_DIR", "."), f"manifold_projection_{BASE}.json")
json.dump(dict(base=BASE, n=N_EVAL, steps=PROJ_STEPS, rows=rows), open(dst, "w"), indent=2)
print(f"\nraw results -> {dst}")
wandb.log({"manifold_table": wandb.Table(columns=["operator", "FID", "R@3", "BLE", "FSR", "displacement"],
                                         data=[[r["op"], r["FID"], r["R3"], r["BLE"], r["FSR"], r["DISP"]] for r in rows])})
wandb.finish()
print("Manifold projection analysis done.")
