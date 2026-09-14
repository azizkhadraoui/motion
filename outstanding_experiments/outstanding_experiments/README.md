# Outstanding experiments — run pack

Everything discussed for the ICLR paper whose results have not come back, packaged so it can be run in one sitting. Compiled 9 September 2026.

All GPU items are **inference-only on frozen checkpoints**. Nothing here trains, and nothing modifies a checkpoint or an existing result file.

---

## Install

Copy the whole directory **into the code checkout**, beside `lfm_clfm_cdfm_experiment.py` — the scripts import it by relative path.

```bash
scp -r outstanding_experiments kaziz@panther-login:~/motion/code/
cd ~/motion/code/outstanding_experiments
cp ../slurm/env.sh .        # or export ENV_SH=/path/to/env.sh
ln -s ../lfm_clfm_cdfm_experiment.py .
```

Check it before submitting anything:

```bash
bash submit_all.sh gate      # refuses with a clear message if a file is missing
```

---

## What is here

| File | Item | What it settles | GPU | Time |
|---|---|---|---|---|
| `01_inproc_restd.py` + `01_run.sh` | 1 | **The paper's framing.** In-ODE projection with the re-standardisation fix, both arms at matched stride, BLE before *and* after the final projection, plus a latent-scale probe. | yes | ~45 min |
| `02_attainability_latdisp.py` + `02_run.sh` | 2, 5–7 | Latent displacement in four normalisations; the gap in the locality curve; the degenerate P1 control; the hardcoded round-trip drift. | yes | ~2.5 h |
| `03_penalty_balance.py` + `03_run.sh` | 6 | Whether the bone term steers the guidance at all, or the sweep is foot-skate guidance with a vestigial bone term. | yes | ~30 min |
| `04_guidance_replicated.py` + `04_run.sh` | 7, 12 | Replications and t-intervals on the sweep that now carries the in-trajectory argument. `GR_L1=1` adds the unsquared-bone control. | yes | ~3.5 h |
| `05_paired_ci.py` | 4 | The paired R@3 and FSR intervals that gate the Pareto wording. | no | seconds |
| `06_lookups.sh` | — | Compute figures, clip counts, best-of-N, sampler rows, per-rep values. | no | seconds |
| `patch_restd_fix.py` | 3 | Applies the same one-line fix to `inproc_endpoint.py`. Inspects by default; `--apply` to write, with a backup. | — | — |
| `submit_all.sh` | — | Submission in four modes. | — | — |

---

## How to run them in parallel

The four GPU jobs share no state and touch no common file, so they are independent **as jobs**. They are not independent **as evidence**: job 01 decides how the paper is framed, and if it lands the wrong way some of the rest changes meaning. That is why `gate` is the default rather than `parallel`.

```bash
bash submit_all.sh gate       # 01 alone — recommended
bash submit_all.sh parallel   # all four at once, 4 × V100
bash submit_all.sh after      # 02–04 concurrently, once 01 has landed
bash submit_all.sh chain      # all four sequentially on one GPU
```

**While the GPU jobs run**, on the login node, costing nothing:

```bash
bash 06_lookups.sh > lookups.txt
python 05_paired_ci.py $WORK_DIR/replicated_ci.json   # needs the per-rep file
```

If the queue gives you four GPUs, `parallel` finishes everything in about three and a half hours wall-clock. If it gives you one, `chain` takes about seven. Either way the non-GPU items are done before the first job returns.

---

## Reading job 01, in order

1. **The two reference rows** must reproduce roughly 0.147 unconstrained and 0.142 post-hoc. If they do not, the harness is wrong and nothing below them means anything. Check this first.
2. **`BLE_pre` against `BLE_post`** on any `+final proj` row. If `BLE_post` is 0 while `BLE_pre` is not, the published BLE column came from the final projection, and the in-ODE arm was partly a post-hoc arm.
3. **`restd ON` against `restd off`** at matched `k`. This is the result that decides the framing.

The prediction to record *before* running, as the plan asks: with the fix, window 10% and k = 4 should land near the post-hoc FID of about 0.14 rather than at 21.25.

Either outcome is usable. If it still collapses, the structural claim is vindicated and stronger than it is today, because it survived the obvious alternative explanation. If the collapse disappears, the in-trajectory claim narrows to soft guidance plus locality — which is why items 3, 4 and 7 in the outstanding list matter more in that branch, not less.

---

## Not included, and why

**Item 3, endpoint correction.** I have not seen `inproc_endpoint.py`, so there is no patched copy here. `patch_restd_fix.py` finds the re-encode line, shows it with context, and only rewrites on `--apply`. If the pattern does not match it says so and changes nothing — make the edit by hand rather than trusting a regex against a file I cannot read.

**Items 8–11** (manifold debugging, LDF intermediates, `compose2_softmask`, composition under the replicated protocol) are bounded investigations rather than scripted runs, and each needs a stop-loss decision that is yours to make.

**Items 13–16** (cross-domain diagnostic, SOTA panel, full test set, physics validation) are new work, not reruns.

---

## Outputs

Everything lands in `$WORK_DIR`:

```
inproc_restd_latent.json          inproc_restd_direct.json
todo11_attainability_latdisp.json
penalty_balance_latent.json       penalty_balance_direct.json
guidance_replicated_latent.json   guidance_replicated_direct.json
guidance_replicated_latent_L1.json          (only with GR_L1=1)
```

Send me those seven plus `lookups.txt` and I can close items 1–7 of the outstanding list in one pass.

---

## One correction to an earlier note

I told you the foot term dominates the bone term by roughly three orders of magnitude *in magnitude*. That is right about the loss values but incomplete about the mechanism: the guidance branch normalises the gradient to unit norm before applying it,

```
g = g / ||g|| ;   v = v - gwt * g * ||v||
```

so the penalty's magnitude does not affect the step size at all — only its **direction** matters. `03_penalty_balance.py` therefore measures `cos(g, g_bone)`, the share of the applied direction the bone term is responsible for, rather than comparing loss magnitudes. Same expected conclusion, correct reasoning behind it.
