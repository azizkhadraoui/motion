#!/usr/bin/env python
# Shared harness for the ICLR constraint-placement experiments.
#
# Everything here follows the conventions of the earlier scripts, so numbers stay comparable:
#   * fixed clip set   : np.random.default_rng(0), first N of a permutation of the test indices, sorted
#   * seeding          : batch starting at clip s, replication r  ->  seed s + 100000*r
#   * two-level average: statistic per clip -> one number per replication -> t-interval ACROSS reps
#   * the latent re-encode is RE-STANDARDISED: z = (E(x) - z_mean) / z_std. The main script's in-ODE
#     branch omits this (see outstanding_experiments/01_inproc_restd.py); every new script uses the
#     corrected round trip so no result depends on that bug.
#
# Nothing in this file trains or writes to a checkpoint.

import os, sys, math, time, importlib.util
import numpy as np, torch

TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 19: 2.093}


def load_main(tag="iclr"):
    """Import the main module through the ablation sentinel (models, data, evaluator built)."""
    os.environ.setdefault("VARIANT", "eval"); os.environ["USE_WANDB"] = "0"; os.environ["ABLATION_IMPORT"] = "1"
    main = os.environ.get("MAIN_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm_clfm_cdfm_experiment.py"))
    spec = importlib.util.spec_from_file_location("expmod", main); M = importlib.util.module_from_spec(spec)
    sys.modules["expmod"] = M
    print(f"[{tag}] importing {main} ...", flush=True)
    try:
        spec.loader.exec_module(M)
    except Exception as e:
        if type(e).__name__ != "M_ABLATION_STOP": raise
    print(f"[{tag}] models loaded.", flush=True)
    return M


def ci95(a):
    a = np.asarray([x for x in a if x == x], float); n = len(a)
    if n == 0: return float("nan"), float("nan")
    if n < 2: return float(a.mean()), float("nan")
    return float(a.mean()), float(TCRIT.get(n - 1, 2.0) * a.std(ddof=1) / math.sqrt(n))


def fmt(m, h, p=4):
    return f"{m:.{p}f}" if h != h else f"{m:.{p}f}±{h:.{p}f}"


# ------------------------------------------------------------------ clip set
class ClipSet:
    """The fixed evaluation clips plus everything that does not depend on the sampler."""
    def __init__(self, M, n, bs=32, start=0):
        """start > 0 takes clips [start, start+n) of the same permutation: start = 512 gives a SCREENING
        set disjoint from the standard 512-clip report set, for hyperparameter selection."""
        self.M = M; self.n = n; self.bs = bs; D = M.DEVICE
        rng = np.random.default_rng(0)
        self.sel = np.array(sorted(rng.permutation(len(M.test_entries))[start:start + n].tolist()))
        caps = [M.test_entries[int(i)]["texts"][0] for i in self.sel]
        self.lens = torch.tensor([int(M.test_lens[i]) for i in self.sel], device=D)
        self.tseq, self.tmask, self.tpool = M.embed_text(caps)
        self._real = None

    def batches(self):
        D = self.M.DEVICE
        for s in range(0, self.n, self.bs):
            e = min(s + self.bs, self.n)
            yield (s, e, torch.tensor(self.tseq[s:e], device=D), torch.tensor(self.tmask[s:e], device=D),
                   torch.tensor(self.tpool[s:e], device=D), self.lens[s:e])

    def real_norm(self, s, e):
        """Normalised, padded ground-truth motion for clips s:e (B,196,263)."""
        M = self.M
        return torch.tensor(np.stack([M.pad_norm(M.test_entries[int(i)]["motion"])[0] for i in self.sel[s:e]]),
                            device=M.DEVICE)

    @torch.no_grad()
    def real_feats(self):
        if self._real is None:
            M = self.M; out = []
            for s, e, _, _, _, L in self.batches():
                gm = M.lengths_to_mask(L, M.MAX_MOTION_LEN)
                out.append(M.memb(self.real_norm(s, e) * gm[..., None], L))
            self._real = np.concatenate(out, 0)
        return self._real


def seed_of(s, rep): return int(s + 100000 * rep)


# ------------------------------------------------------------------ latent round trip (corrected)
def decode(M, z, is_lat):
    return M.rvq.decoder(z * M.z_std_t + M.z_mean_t) if is_lat else z


def encode(M, mn, is_lat):
    return (M.rvq.encoder(mn) - M.z_mean_t) / M.z_std_t if is_lat else mn


def null_cond(net, B, M):
    return (net.null_seq.unsqueeze(0).expand(B, -1, -1),
            torch.ones(B, M.T5_MAXLEN, dtype=torch.bool, device=M.DEVICE),
            net.null_pool.unsqueeze(0).expand(B, -1))


# ------------------------------------------------------------------ distribution metrics
def sliced_wasserstein(G, R, n_dir=1000, seed=0):
    """Sliced W2 between equal-size feature sets (random unit directions, fixed seed)."""
    rng = np.random.default_rng(seed); d = G.shape[1]
    W = rng.standard_normal((d, n_dir)); W /= np.linalg.norm(W, axis=0, keepdims=True)
    n = min(len(G), len(R))
    pg = np.sort((G[:n] @ W), 0); pr = np.sort((R[:n] @ W), 0)
    return float(np.sqrt(((pg - pr) ** 2).mean()))


def mmd_rbf(G, R, bw=None):
    """Unbiased MMD^2 with an RBF kernel; bandwidth = median pairwise distance of the REAL set,
    so the kernel is fixed across methods."""
    G = torch.tensor(G, dtype=torch.float64); R = torch.tensor(R, dtype=torch.float64)
    if bw is None:
        dr = torch.cdist(R, R); bw = float(dr[dr > 0].median())
    k = lambda A, B: torch.exp(-torch.cdist(A, B) ** 2 / (2 * bw * bw))
    m, n = len(G), len(R)
    kgg = k(G, G); krr = k(R, R); kgr = k(G, R)
    s = ((kgg.sum() - kgg.diag().sum()) / (m * (m - 1)) + (krr.sum() - krr.diag().sum()) / (n * (n - 1))
         - 2 * kgr.mean())
    return float(s), bw


def prdc(R, G, k=5):
    """Precision / recall / density / coverage (Naeem et al. 2020), k-NN balls, Euclidean."""
    R = torch.tensor(R, dtype=torch.float64); G = torch.tensor(G, dtype=torch.float64)
    drr = torch.cdist(R, R); dgg = torch.cdist(G, G); drg = torch.cdist(R, G)
    rr = drr.kthvalue(k + 1, dim=1).values          # +1: the point itself is its own 0-distance neighbour
    rg = dgg.kthvalue(k + 1, dim=1).values
    inside = drg < rr[:, None]                      # (real i, gen j): gen j inside real i's ball
    precision = float(inside.any(0).double().mean())
    recall = float((drg < rg[None, :]).any(1).double().mean())
    density = float(inside.double().sum() / (k * G.shape[0]))
    coverage = float((drg.min(1).values < rr).double().mean())
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


def diversity(G, n_pairs=300, seed=0):
    rng = np.random.default_rng(seed); n = len(G)
    a = rng.integers(0, n, n_pairs); b = rng.integers(0, n, n_pairs)
    return float(np.linalg.norm(G[a] - G[b], axis=1).mean())


# ------------------------------------------------------------------ one protocol run
@torch.no_grad()
def run_protocol(M, clips, gen, reps, label="", full_metrics=False, extra_keys=()):
    """gen(s, e, tseq, tmask, tpool, L, seed) -> (mn, extras) with mn normalised (B,196,263) and
    extras a dict of per-clip numpy arrays. Returns per-replication numbers and their CIs."""
    R = clips.real_feats(); per = {}
    t0 = time.time()
    for rep in range(reps):
        mf, ble, fsr = [], [], []; ex = {k: [] for k in extra_keys}
        for s, e, ts, tm, tp, L in clips.batches():
            mn, extras = gen(s, e, ts, tm, tp, L, seed_of(s, rep))
            gm = M.lengths_to_mask(L, M.MAX_MOTION_LEN); J = M._gj(mn)
            mf.append(M.memb(mn * gm[..., None], L))
            ble.append(M.ble_pc_joints(J, L)); fsr.append(M.fsr_pc(J, L))
            for k in extra_keys: ex[k].append(np.asarray(extras[k]))
        G = np.concatenate(mf, 0)
        row = dict(FID=float(M.fid_calc(G, R)), R3=float(M.rprec(G, R)[3]),
                   BLE=float(np.concatenate(ble).mean()), FSR=float(np.concatenate(fsr).mean()))
        for k in extra_keys: row[k] = float(np.concatenate(ex[k]).mean())
        if full_metrics:
            row["SW"] = sliced_wasserstein(G, R)
            row["MMD"] = mmd_rbf(G, R)[0]
            row.update(prdc(R, G))
            row["DIV"] = diversity(G)
        for k, v in row.items(): per.setdefault(k, []).append(v)
    out = {k: ci95(v) for k, v in per.items()}
    out["_per_rep"] = per; out["_secs"] = round(time.time() - t0)
    if label:
        print(f"  {label:<46} FID={fmt(*out['FID'])}  R@3={fmt(*out['R3'], p=3)}  "
              f"BLE={fmt(*out['BLE'], p=5)}  FSR={fmt(*out['FSR'])}  ({out['_secs']}s)", flush=True)
    return out


# ------------------------------------------------------------------ MeanFlow network (two time arguments)
# Convention of this codebase: z_t = (1 - t) z0 + t z1, t = 0 noise, t = 1 data, integrate 0 -> 1.
# The MeanFlow network predicts the AVERAGE velocity over [t, s] (s >= t):
#     u(z_t, t, s) = 1/(s - t) * integral_t^s v(z_tau, tau) dtau,   so   z_s = z_t + (s - t) u.
# One-step sampling: z1 = z0 + u(z0, 0, 1). At s = t it is the instantaneous velocity.
def mf_net_class(M):
    import torch.nn as nn

    class MFNet(M.FMNet):
        """FMNet plus an embedding of the interval length h = s - t, zero-initialised so that at
        warm start u(z, t, s) == v_teacher(z, t) for every s.

        H_SCALE: the interval is embedded as sinusoidal(H_SCALE * (s - t)). It must match the scale at
        which the teacher embeds t (raw, i.e. 1.0): with 1000, du/dt inherits a ~1000x sensitivity to
        the interval branch once it learns, the self-referential MF target grows with it, and training
        diverged (raw MSE 0.7 -> 2e5 in 4k steps, job 404281)."""
        H_SCALE = float(os.environ.get("MF_HSCALE", "1.0"))

        def __init__(s_, cd, Tlen, hid, layers, heads):
            super().__init__(cd, Tlen, hid, layers, heads)
            s_.h_mlp = nn.Sequential(nn.Linear(M.TIME_DIM, hid), nn.SiLU(), nn.Linear(hid, hid))
            nn.init.zeros_(s_.h_mlp[-1].weight); nn.init.zeros_(s_.h_mlp[-1].bias)

        def forward(s_, z, t, s, tseq, tmask, tpool, length):
            h = s_.in_proj(z) + s_.pos
            c = (s_.time_mlp(M.sinusoidal(t, M.TIME_DIM)) + s_.h_mlp(M.sinusoidal((s - t) * s_.H_SCALE, M.TIME_DIM))
                 + s_.text_proj(tpool) + s_.len_emb(length.clamp(0, M.MAX_MOTION_LEN)))
            for b in s_.blocks: h = b(h, c, tseq, tmask)
            return s_.out_proj(h)
    return MFNet


def new_mfnet(M):
    return mf_net_class(M)(M.RVQ_CODE_DIM, M.T_LAT, M.LHID, M.LLAYERS, M.LHEADS).to(M.DEVICE)


def mf_ckpt_path(M, variant): return os.path.join(M.CK, f"meanflow_{variant}_best.pt")


def load_mf(M, variant):
    """Returns (net, z_mean, z_std) or None if MF-1 has not produced this checkpoint."""
    p = mf_ckpt_path(M, variant)
    if not os.path.exists(p): return None
    ck = torch.load(p, map_location=M.DEVICE, weights_only=False)
    net = new_mfnet(M); net.H_SCALE = float(ck.get("h_scale", 1000.0)); net.load_state_dict(ck["state"]); net.eval()
    zm = torch.tensor(ck["z_mean"], device=M.DEVICE).float(); zs = torch.tensor(ck["z_std"], device=M.DEVICE).float()
    return net, zm, zs
