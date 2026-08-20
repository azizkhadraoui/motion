# Workshop draft — analysis, TODO closure plan, and prepared experiments

Deadline **29 Aug 2026 AoE**. Draft read in full; every TODO box triaged below.

## 1. Where the draft actually stands

The argument is complete and the writing is good. Sections 2–5 are publishable as they are: the
problem formalization, the preservation condition `A(C_X) ⊆ C_X`, the operator-ordering table, and
the mechanism are all clean, and the cautious wording around the ceiling is right.

What is missing is almost entirely **reporting**, not science. Of the twelve TODO boxes, **five need
no new compute at all** — the numbers exist and only have to be tabulated. Three need short
inference-only jobs. One is correctly deferred. Two are pure writing. One (TODO 9) cannot be done as
specified and needs an honest substitution.

The two things that genuinely worry me are in §4 below, and neither is flagged by any TODO box.

## 2. TODO closure matrix

| # | What it wants | Data status | Action | GPU |
|---|---|---|---|---|
| 1 | Hard in-process ablation subsection | **Have it** (9 configs, FID + BLE) | Tabulate; `fig_inproc_ablation.pdf` supplied. R@3/FSR are in the ablation job's W&B table and stdout — paste `ablation_<jobid>.out` if you want those columns | none |
| 2 | Paired FID stats with p-value | **Have the run**, not the numbers | `paired_stats.py` computes Δ, CI, t and p from per-rep values. Otherwise `replicated_ci_v2.py` recomputes them in-script | none |
| 3 | Penalty-baseline numbers under the same protocol | Single-seed only (0.292 / 0.263) | `replicated_ci_v2.py` adds both penalty bases to the replicated protocol | shared |
| 4 | Direct-guidance sanity check | **Missing** | `todo4_guidance_sanity.py` — measures raw-penalty descent per step, and tests endpoint-based guidance | ~4 h |
| 5 | Sampling-strategy ablation | **Have it** (8 configs) | Tabulate; `fig_sampler_ablation.pdf` supplied | none |
| 6 | Compositional-control subsection | **Have it** (v1 fail, v2 success) | Writing only. Keep the terminology "position-space composition" | none |
| 7 | Architecture / reproducibility details | **Have it all** | Writing only; every constant is in the handoff §7 | none |
| 8 | Foot-contact correction + metric | **Have FSR** but never reported | `replicated_ci_v2.py` now collects FSR mean and p95 with CIs. State plainly that FSR is reduced, never zeroed (0.0026–0.0064 floors) | shared |
| 9 | Full 4,384-clip test set + mirror aug | **Not possible as written** | See §3.6 — 11,692 mirror entries are not materialized. Run `EVAL_N=2644` (all available clips) and state the limitation | ~3 h |
| 10 | SOTA checkpoint panel | Not started | **Keep deferred.** Needs external codebases; not workshop-scale work | — |
| 11 | Isolate encoder vs decoder | **Missing** | `todo11_ae_isolate.py` — four probes, including direct latent optimization | ~1 h |
| 12 | Main-paper / appendix consolidation | — | Last step, after everything lands | none |

## 3. Issues the TODO boxes do not cover

**3.1 Mixed evaluation protocols in the abstract and intro.** Line 7 of the abstract and line 51 of
the introduction compare hard-projection FID (21 / 55) against the *replicated* baseline 0.169. Those
21 and 55 figures came from the single-seed protocol whose matched baseline is 0.147. §4.3 already
handles exactly this for guidance (lines 227–228) and does it well — apply the same sentence to the
hard-projection numbers. The comparison is unaffected in substance (both are ~140× and ~330×), but a
reviewer who notices the mismatch will assume worse.

**3.2 HumanML3D is used but never cited.** §6.1 names the dataset with no reference. The References
section has seven entries, all text-to-motion models, and is missing: flow matching itself (Lipman et
al.), rectified flow (Liu, Gong & Liu, arXiv:2209.03003), the dataset (Guo et al. 2022), CFG (Ho &
Salimans), Sentence-T5, physics-based motion (PhysDiff), the FM-guidance literature, and the two
July-2026 adjacent papers from the handoff (Lagrangian Dual Flows, ConFlow) which **must** be
differentiated. This is the single largest remaining gap and it is not represented by any TODO box.

**3.3 R@3 is absent from Table 4.** The harness computes it and the sweep tables report it. A
reviewer's first question about a method that changes the output geometry is whether text alignment
survives. `replicated_ci_v2.py` collects it with CIs.

**3.4 The evaluation-set limitation is nowhere in the paper.** Evaluation uses 1,024 clips drawn from
2,644, not the standard 4,384, and there is no mirror augmentation. That belongs in §6.1 and in
checklist item 2, not only in TODO 9.

**3.5 Cross-reference errors.** Lines 132 and 142 refer to "Section 2.3"; Section 2 has no numbered
subsections. Also Eq. (2.9) writes `Y0 ∈ N(0,I)` where it means `∼`.

**3.6 TODO 9 cannot be executed as specified.** The 11,692 `M`-prefixed mirror entries are not
materialized as files on disk, so the 4,384-clip test set does not exist locally. Two honest options:
(a) evaluate on all 2,644 available clips and state the deviation explicitly, or (b) implement the
mirror transform on the 263-d vector to regenerate them. (b) is error-prone — a sign error in the
root-rotation or RIC channels silently corrupts the metric — and I would not attempt it nine days out.
Take (a): `CI_REPS=5 EVAL_N=2644 sbatch run_ci_v2.sh`.

## 4. Two risks that could change what the paper claims

**4.1 Figure 1 is missing its control, and it may not show what the caption says.**

§6.7 reports that re-projecting each round (≈0.024) is nearly indistinguishable from plain iterated
autoencoding (≈0.026), and concludes that repeated correction cannot overcome the autoencoder's
failure to preserve the constraint. But the near-identity of the two curves admits a second reading:
that repeated autoencoding degrades the motion generally, BLE rises as a side effect of that drift,
and the projection is largely irrelevant to the trend. The experiment as run cannot distinguish these,
because **the unprojected control is missing** — nobody has iterated the autoencoder on plain real
motion.

`roundtrip_v2.py` adds that third curve plus MPJPE drift. If the control also climbs to ≈0.024, §6.7
must be reworded (the constraint-specific evidence then rests on Table 2's single round trip, which is
clean and sufficient on its own). If it stays low, the current wording stands and the figure gets
stronger. Either way this is a 20-minute job and it protects a figure that is currently front and
centre.

**4.2 In-ODE projection corrects the noisy iterate, not the estimated endpoint.**

I read this out of the sampler in the previous turn. `mode=="inproc"` decodes `z` at t ≥ 0.5 and
corrects that; soft guidance likewise differentiates `pen(D(Z_t))`. The paper's own training-time
penalty (Eq. 6.2) correctly uses `Ŷ₁ = Y_t + (1−t)v_θ`, and diffusion analogues correct `x̂₀`. So the
central negative claim — "in-ODE enforcement fails" — is open to: *you enforced it in the weakest
available place*.

The mechanism in §4.2 predicts the endpoint variant fails too, since the correction still passes
through `E∘D`. `inproc_endpoint.py` tests it directly at three schedules. If it collapses, one sentence
closes the objection permanently. If it does not, that is a real finding and Sections 3.1 and 4.2 need
narrowing — better to learn that now than in review.

The same issue is the likely explanation for the CDFM BLE increase that TODO 4 asks about:
descending `pen(Z_t)` need not descend `pen(Ŷ₁)`, and BLE is measured on the final sample.
`todo4_guidance_sanity.py` logs both penalties per step and settles which branch of TODO 4 applies.

**One more, smaller:** if TODO 11's latent-optimization probe drives BLE to ≈0, then valid motion *is*
inside the decoder's image and the failure is reachability rather than expressiveness. Your §4.1
hedge already survives this ("does not imply that no latent code can decode to an exactly valid
motion"), but the term "representational ceiling" would need a defining sentence to match. Worth
knowing before submission rather than being asked.

## 5. Figures delivered (built from data already in hand, no compute needed)

| File | Use |
|---|---|
| `fig_main_replicated.pdf` | Table 4 as a figure: FID with 95% CIs + exact BLE=0. Good abstract-adjacent figure |
| `fig_inproc_ablation.pdf` | **TODO 1.** 9 schedules, log-FID, with both reference lines. Shows the dose–response and that nothing recovers |
| `fig_guidance_sweep.pdf` | **§6.6.** Two panels, LFM fragility vs CDFM robustness, with the AE floor drawn in — ties the sweep to the mechanism |
| `fig_sampler_ablation.pdf` | **TODO 5.** Makes the null result readable at a glance |

NeurIPS column width, serif, vector PDF; PNG copies included for quick viewing. `make_figs.py` is
included so numbers can be updated in one place. `roundtrip_v2.py` emits its own figure on the cluster.

Note: `fig_inproc_ablation.pdf` labels its reference lines 0.147 / 0.142, the matched single-seed
values — consistent with §3.1 above. Do not relabel them 0.169 without re-running the sweep.

## 6. Suggested run order

Everything is inference-only on existing checkpoints; nothing retrains.

```bash
cd $HOME/motion/code && git fetch origin && git reset --hard origin/main
source $HOME/motion/env.sh          # or $CPY will be unset
squeue -u kaziz                      # cancel anything stale BEFORE submitting

sbatch run_roundtrip_v2.sh           # ~20 min — protects Figure 1 (risk 4.1)
sbatch run_todo11.sh                 # ~1 h   — TODO 11
sbatch run_ci_v2.sh                  # ~10-14 h — TODO 2, 3, 8 in one job
sbatch run_inproc_endpoint.sh        # ~4 h   — risk 4.2
sbatch run_todo4.sh                  # ~4 h   — TODO 4
squeue -u kaziz                      # confirm 5 jobs, no duplicates
```

Five jobs, under the 8-GPU limit, all can run concurrently. The CI job is the long pole; start it
early. Then, after it lands:

```bash
CI_REPS=5 EVAL_N=2644 sbatch run_ci_v2.sh    # TODO 9 substitute, ~3 h
```

While they run, the zero-compute work is: TODO 1, 5, 6, 7 tabulation and the References section
(§3.2) — that last one is the biggest single remaining gap and needs no cluster at all.

## 7. What I would not do before the deadline

TODO 10 (SOTA panel) — deferred, as agreed. Regenerating mirror augmentation. Any new constraint
type. Reflow. The story is complete; these five jobs are about defending it, not extending it.
