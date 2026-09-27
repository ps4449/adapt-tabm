# TabLoRA — Experiment Summary

## Motivation

TabM uses BatchEnsemble to build k ensemble heads cheaply: each head shares a weight matrix W
and learns only per-head scaling vectors r and s. The per-head capacity is limited to element-wise
scaling, which is expressive but inflexible in terms of subspace coverage.

TabLoRA replaces this with LoRA-style low-rank adapters. Each head learns an additive low-rank
delta (A_i, B_i) on top of the shared W, giving it a richer low-dimensional subspace to
personalise rather than just scaling activations. The hypothesis is that this leads to more
diverse and complementary ensemble members.

---

## Experiment #1 — 2026-09-04

> Log: `logs/results/tablora_california_20260904_214457.txt`

**Dataset:** California Housing — regression (median house value), metric: negative RMSE (higher = better).

### Configuration

| Parameter | TabM (baseline) | TabLoRA |
|-----------|----------------|---------|
| arch_type | `tabm` | `tabm-lora` |
| k | 32 | 32 |
| rank | — | 4 |
| n_blocks | 3 | 3 |
| d_block | 400 | 400 |
| dropout | 0.2077 | 0.2077 |
| lr | 8.72e-4 | 8.72e-4 |
| weight_decay | 3.78e-2 | 3.78e-2 |
| n_parameters | 438,688 | 631,456 (+44%) |
| Hyperparams tuned | Yes (100 Optuna trials) | No — borrowed from TabM |

### Per-seed results (15 seeds)

| Seed | TabLoRA Val | TabLoRA Test |
|------|-------------|--------------|
| 0  | −0.5024 | −0.5054 |
| 1  | −0.5004 | −0.5019 |
| 2  | −0.5015 | −0.5052 |
| 3  | −0.4982 | −0.4985 |
| 4  | −0.5011 | −0.5013 |
| 5  | −0.5017 | −0.4969 |
| 6  | −0.4975 | −0.4943 |
| 7  | −0.5000 | −0.4960 |
| 8  | −0.4993 | −0.5006 |
| 9  | −0.4973 | −0.4974 |
| 10 | −0.5012 | −0.4971 |
| 11 | −0.4993 | −0.5004 |
| 12 | −0.4997 | −0.4966 |
| 13 | −0.5053 | −0.4949 |
| 14 | −0.4999 | −0.4953 |

### Summary

| Model | Mean Val | Mean Test | Std (Test) | Ensemble-5 Test |
|-------|----------|-----------|------------|-----------------|
| TabM (baseline) | −0.4458 | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA | −0.5003 | −0.4988 | ±0.0034 | −0.4915 |
| diff. | −0.0545 | −0.0574 | 3× higher | −0.0513 |

### Analysis

**Why the gap exists.**
The TabLoRA run reused hyperparameters tuned specifically for BatchEnsemble dynamics. LoRA
adapters have a product structure (A × B) that typically benefits from a higher learning rate
and different regularisation. Running with TabM's lr/wd is effectively mis-specified for the
new architecture.

**Why variance is higher.**
std ±0.0034 is nearly 3x TabM's ±0.0012. The adapter matrices are sensitive to initialisation
under this configuration - a sign that rank or regularisation needs tuning.

**Parameter count.**
At rank=4, TabLoRA uses 192K more parameters than TabM (+44%). This is partly structural: each
LoRA delta costs rank × (in + out) vs BatchEnsemble's in + out. Rank=2 would bring this to
~+25% — worth testing before concluding LoRA is less efficient.

**Ensemble diversity.**
Ensemble-5 (−0.4915) is noticeably better than the single-model mean (−0.4988), confirming
heads are still learning diverse solutions despite the overall gap.

---

## Experiment #2 — 2026-09-06

> Log: `logs/results/tablora_california_20260906_153246.txt`
> Config: `california/0-evaluation/0-rank16.toml`

**Dataset:** California Housing — regression (median house value), metric: negative RMSE (higher = better).

### Configuration

| Parameter | TabM (baseline) | TabLoRA rank 4 | TabLoRA rank 16 |
|-----------|-----------------|----------------|-----------------|
| arch_type | `tabm` | `tabm-lora` | `tabm-lora` |
| k | 32 | 32 | 32 |
| rank | — | 4 | 16 |
| n_blocks | 3 | 3 | 3 |
| d_block | 400 | 400 | 400 |
| dropout | 0.2077 | 0.2077 | 0.2077 |
| lr | 8.72e-4 | 8.72e-4 | 8.72e-4 |
| weight_decay | 3.78e-2 | 3.78e-2 | 3.78e-2 |
| n_parameters | 438,688 | 631,456 | 1,402,528 |
| Hyperparams tuned | Yes (100 Optuna trials) | No — borrowed from TabM | No — borrowed from TabM |

### Per-seed results (15 seeds)

| Seed | TabLoRA Val | TabLoRA Test |
|------|-------------|--------------|
| 0  | −0.4918 | −0.4933 |
| 1  | −0.4871 | −0.4867 |
| 2  | −0.4928 | −0.4926 |
| 3  | −0.4881 | −0.4942 |
| 4  | −0.4883 | −0.4968 |
| 5  | −0.4901 | −0.4881 |
| 6  | −0.4924 | −0.4921 |
| 7  | −0.4931 | −0.4919 |
| 8  | −0.4965 | −0.4936 |
| 9  | −0.4923 | −0.4911 |
| 10 | −0.4932 | −0.4903 |
| 11 | −0.4921 | −0.4975 |
| 12 | −0.4907 | −0.4945 |
| 13 | −0.4941 | −0.4926 |
| 14 | −0.4889 | −0.4935 |

### Summary

| Model | Mean Val | Mean Test | Std (Test) | Ensemble-5 Test |
|-------|----------|-----------|------------|-----------------|
| TabM (baseline) | −0.4458 | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank 4 | −0.5003 | −0.4988 | ±0.0034 | −0.4915 |
| TabLoRA rank 16 | −0.4914 | −0.4926 | ±0.0027 | −0.4871 |
| rank 16 vs rank 4 | +0.0089 | +0.0062 | −0.0007 | +0.0044 |
| rank 16 vs TabM | −0.0456 | −0.0512 | +0.0015 | −0.0469 |

### Analysis

**Higher rank helps.**
Increasing the LoRA rank from 4 to 16 improves the mean test score by 0.0062 and the
five-model ensemble score by 0.0044. Test-score variability also falls from ±0.0034 to
±0.0027.

**The baseline gap remains.**
Rank 16 does not close the gap to the tuned TabM baseline: its mean test score remains 0.0512
lower, and its ensemble score remains 0.0469 lower. Both TabLoRA runs reuse hyperparameters
tuned for TabM, so this comparison isolates the effect of rank but not TabLoRA's tuned potential.

**Parameter cost.**
Rank 16 uses 1,402,528 parameters: 122% more than rank 4 and 220% more than TabM. The accuracy
gain therefore comes with a substantial parameter-efficiency tradeoff.

**Ensemble gain.**
The rank-16 five-model ensemble improves over its single-model mean by 0.0054. This confirms
that independently seeded rank-16 models remain complementary, although the gain is smaller
than the 0.0073 improvement observed at rank 4.

---

## Experiment #3 — 2026-09-06

> Log: `logs/results/tablora_california_20260906_163700.txt`

**Config:** rank=2, adapter_scale=1.0 (default), lr=8.72e-4, wd=3.78e-2 (TabM defaults), no input scaling.

| Model | Mean Test | Std | Ensemble-5 Test |
|-------|-----------|-----|-----------------|
| TabM | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank=2 | −0.5016 | ±0.0040 | −0.4930 |

Worse than rank=4 and rank=16. Rank=2 is too small — the adapter cannot represent enough of the
residual subspace. High variance (±0.0040) confirms instability.

---

## Experiment #4 — 2026-09-10

> Log: `logs/results/tablora_california_20260910_162345.txt`

**Config:** rank=2, adapter_scale=1.0 (default), lr=1.50e-3, wd=5.00e-3 (manually adjusted).

| Model | Mean Test | Std | Ensemble-5 Test |
|-------|-----------|-----|-----------------|
| TabM | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank=2 | −0.5026 | ±0.0025 | −0.4935 |

Higher lr reduced variance slightly vs experiment #3 (±0.0040 → ±0.0025) but did not improve
mean test. Rank=2 remains too limited regardless of lr.

---

## Experiment #5 — 2026-09-10

> Log: `logs/results/tablora_california_20260910_180352.txt`

**Config:** rank=2, adapter_scale=1.0, lr=4.98e-3, wd=2.01e-4 (Optuna-found values for rank=2).

| Model | Mean Test | Std | Ensemble-5 Test |
|-------|-----------|-----|-----------------|
| TabM | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank=2 | −0.4997 | ±0.0036 | −0.4865 |

Best rank=2 result — ensemble −0.4865 improved significantly. The high lr (5e-3) with very low
wd compensates for rank=2's limited capacity by training more aggressively. Still well below
TabM. Confirmed rank=2 is not a viable direction.

---

## Experiment #6 — 2026-09-10

> Log: `logs/results/tablora_california_20260910_184206.txt`

**Config:** rank=4, adapter_scale=0.25, lora_input_scaling=true, lr=1.77e-3, wd=8.52e-3
(Optuna-tuned, 50 trials).

| Model | Mean Test | Std | Ensemble-5 Test |
|-------|-----------|-----|-----------------|
| TabM | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank=4 | −0.4523 | ±0.0017 | −0.4510 |
| Δ | | | −0.0108 |

**Best result with adapter_scale approach.** Gap narrowed to −0.011 from −0.057 at the start.
Three changes drove this: (1) adapter_scale=0.25 reduces adapter over-contribution, (2)
lora_input_scaling adds per-head input diversity, (3) lr/wd tuned specifically for LoRA dynamics.

---

## Experiment #7 — 2026-09-10

> Log: `logs/results/tablora_california_20260910_200309.txt`

**Config:** rank=8, adapter_scale=0.25, lora_input_scaling=true, lr=5.57e-4, wd=3.17e-2
(Optuna-tuned, 50 trials).

| Model | Mean Test | Std | Ensemble-5 Test |
|-------|-----------|-----|-----------------|
| TabM | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank=8 | −0.4566 | ±0.0016 | −0.4557 |
| vs rank=4 (#6) | −0.0043 | — | −0.0047 |

Rank=8 with fixed adapter_scale=0.25 is **worse** than rank=4. With fixed scale, doubling rank
doubles the adapter's aggregate contribution magnitude — the tuned lr (5.57e-4, much lower than
rank=4's 1.77e-3) compensates partially but cannot fully recover. This motivates `lora_alpha`
canonical scaling where effective scale = alpha/rank adjusts automatically with rank.

---

## Experiment #8 — 2026-09-11 (swara-experiments replication)

> Log: `logs/results/tablora_california_20260911_025319.txt`

**Config:** rank=8, lora_alpha=0.5 (effective scale=0.0625), lora_input_scaling=true,
lr=8.72e-4, wd=3.78e-2 (TabM defaults). Replicates swara-experiments branch result.

| Model | Mean Test | Std | Ensemble-5 Test |
|-------|-----------|-----|-----------------|
| TabM | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA | −0.4463 | ±0.0016 | −0.4448 |
| Δ | −0.0049 | — | −0.0046 |

Confirmed replication of swara-experiments' best result (−0.4451). The missing piece from
earlier failed attempts was `lora_input_scaling=true` — without it, lora_alpha=0.5 gave
−0.5002. The ScaleEnsemble at the input is essential for head diversity at this low adapter scale.

---

## Experiment #9 — 2026-09-11 (tuned lr/wd)

> Log: `logs/results/tablora_california_20260911_042935.txt`

**Config:** rank=8, lora_alpha=0.5, lora_input_scaling=true, lr=1.12e-3, wd=9.39e-2
(Optuna-tuned, 50 trials over lr, wd).

| Model | Mean Test | Std | Ensemble-5 Test |
|-------|-----------|-----|-----------------|
| TabM | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA | −0.4448 | ±0.0009 | **−0.4434** |
| Δ | −0.0034 | — | −0.0032 |

**Best result to date.** Gap to TabM narrowed to −0.003 on ensemble. Notably, std (±0.0009)
is now *lower* than TabM's (±0.0012) — TabLoRA is more stable across seeds. Tuning lr/wd
specifically for the lora_alpha regime (higher lr 1.12e-3 vs TabM's 8.72e-4, higher wd 9.39e-2
vs TabM's 3.78e-2) gained another 0.0014 on ensemble over the default values.

---

## Experiment #10 — 2026-09-11 (multi-dataset comparison)

> Log: `logs/results/comparison_20260911_181945.txt` (TabM, computed locally) +
> cluster comparison run (TabLoRA), consolidated via `tools/compare_results.py`.

**Config:** rank=8, lora_alpha=0.5, lora_input_scaling=true, lr=1.1157e-3, wd=9.3899e-2,
n_blocks=3, d_block=400 — same config as Experiment #9, reused as-is across all 5 datasets
(not re-tuned per dataset; only `california` has had Optuna tuning for TabLoRA).

| Dataset | TabM Test | Ens-5 | TabLoRA Test | Ens-5 | Δ Test | Δ Ens |
|---------|-----------|-------|--------------|-------|--------|-------|
| california | −0.4414 | −0.4402 | −0.4453 | −0.4439 | −0.0039 | −0.0037 |
| adult | 0.8575 | 0.8583 | 0.8572 | 0.8575 | −0.0003 | −0.0008 |
| higgs-small | 0.7394 | 0.7409 | 0.7362 | 0.7369 | −0.0032 | −0.0040 |
| otto | 0.8275 | 0.8284 | 0.8171 | 0.8175 | −0.0104 | −0.0109 |
| diamond | −0.1310 | −0.1307 | −0.1319 | −0.1314 | −0.0009 | −0.0007 |

`california`/`diamond` scores are negative RMSE (regression, higher is better); `adult`/`higgs-small`/`otto`
scores are accuracy-style (classification, higher is better).

### Analysis

**TabLoRA trails TabM on all 5 datasets**, with the config borrowed unchanged from California's
tuned Experiment #9 — none of `adult`, `higgs-small`, `otto`, `diamond` have had their own
Optuna sweep for TabLoRA yet.

**`adult` and `diamond` are nearly matched** (Δ Ens of −0.0008 and −0.0007) — the untuned config
already generalises well here.

**`otto` shows the largest gap** (Δ Ens −0.0109), consistent with it being the only multiclass
(9-way) dataset in the set — the shared-config assumption may hold less well for multiclass
outputs than binary/regression.

**Next step:** run per-dataset Optuna tuning (`slurm/tune_tablora.sh`) for `adult`, `higgs-small`,
`otto`, and `diamond` the way it was already done for `california`, before drawing conclusions
about TabLoRA's viability outside California.

---

## Experiment #11 — 2026-09-11 (joint rank+alpha Optuna search)

> Log: `logs/results/tablora_california_20260911_225626.txt`
> Config: `california/0-rank4-tuned-evaluation/0.toml`
> Tuning source: SLURM job 34593 — 100-trial Optuna search over `rank` (categorical
> 2/4/8/16), `lora_alpha`, `dropout`, `lr`, `weight_decay` jointly (see
> `training_summary.md` for the full tuning-job rationale).

**Config:** rank=4, lora_alpha=0.14822 (effective scale ≈ alpha/rank = 0.037),
lora_input_scaling=true, dropout=0.1787, lr=1.56e-3, wd=9.30e-2, n_blocks=3, d_block=400.

This is the first search to let `rank` and `lora_alpha` vary together, instead of fixing
rank and tuning alpha alone (as Experiment #10/job 34490 did). Initiated because the
alpha-only tune (Exp #10) did not beat the manually-chosen rank=8/alpha=0.5 combo — since
`lora_alpha`'s effective contribution is `alpha/rank`, tuning alpha without rank searches
only half of a coupled quantity.

### Summary

| Model | Mean Test | Std (Test) | Ensemble-5 Test | n_parameters |
|-------|-----------|------------|------------------|--------------|
| TabM (baseline) | −0.4414 | ±0.0012 | −0.4402 | 438,688 |
| TabLoRA rank=8 (Exp #9, previous best) | −0.4448 | ±0.0009 | −0.4434 | 888,480 |
| **TabLoRA rank=4 (this run)** | **−0.4439** | ±0.0013 | **−0.4424** | **631,712** |
| vs TabM | −0.0025 | +0.0001 | −0.0022 | +44% |
| vs Exp #9 | +0.0009 | +0.0004 | +0.0010 | −29% |

### Analysis

**Best TabLoRA result to date**, on both mean test and ensemble score, and the smallest gap
to TabM yet reached (−0.0025 mean / −0.0022 ensemble, down from Exp #9's −0.0034/−0.0032).

**Achieved with 29% fewer parameters than Exp #9** (631,712 vs 888,480) — the joint search
found a more parameter-efficient operating point (lower rank, smaller effective adapter
scale), not just a marginally better one. Std (±0.0013) remains close to TabM's (±0.0012),
confirming training stability holds at this smaller rank too.

**Winner's-curse gap confirmed.** The single best-trial test score reported by Optuna during
tuning was −0.4421 (near-parity with TabM); the true 15-seed mean is −0.4439, a 0.0018
regression. The best-of-100-trials number was optimistic, as expected — trials are selected
on validation score using one seed, so some regression toward the true mean is normal. Future
"best trial" numbers should be treated as upper bounds pending full 15-seed confirmation, not
as final results.

**Still short of TabM**, but via a qualitatively different mechanism (rank=4, alpha≈0.037)
than any prior attempt — suggests the search space around rank=8/alpha=0.5 was not actually
optimal, and supports doing this same joint search for the other 4 datasets rather than
reusing the California-tuned config as Experiment #10 did.

---

## Master Summary Table

> All scores are negative RMSE — higher is better.
> Δ = TabLoRA − TabM (negative means TabLoRA is worse).

| # | Log timestamp | Rank | Scale | input_scaling | lr | Mean Test | Δ | Ens-5 | Δ Ens |
|---|---------------|------|-------|---------------|----|-----------|---|-------|-------|
| TabM | — | — | — | — | 8.72e-4 | −0.4414 | — | −0.4402 | — |
| 1 | 20260904_214457 | 4 | 1.0 (fixed) | No | 8.72e-4 | −0.4988 | −0.0574 | −0.4915 | −0.0513 |
| 2 | 20260906_153246 | 16 | 1.0 (fixed) | No | 8.72e-4 | −0.4926 | −0.0512 | −0.4871 | −0.0469 |
| 3 | 20260906_163700 | 2 | 1.0 (fixed) | No | 8.72e-4 | −0.5016 | −0.0602 | −0.4930 | −0.0528 |
| 4 | 20260910_162345 | 2 | 1.0 (fixed) | No | 1.50e-3 | −0.5026 | −0.0612 | −0.4935 | −0.0533 |
| 5 | 20260910_180352 | 2 | 1.0 (fixed) | No | 4.98e-3 | −0.4997 | −0.0583 | −0.4865 | −0.0463 |
| **6** | **20260910_184206** | **4** | **0.25 (fixed)** | **Yes** | **1.77e-3** | **−0.4523** | **−0.0109** | **−0.4510** | **−0.0108** |
| 7 | 20260910_200309 | 8 | 0.25 (fixed) | Yes | 5.57e-4 | −0.4566 | −0.0152 | −0.4557 | −0.0155 |
| 8 | 20260911_025319 | 8 | alpha=0.5 (0.0625) | Yes | 8.72e-4 | −0.4463 | −0.0049 | −0.4448 | −0.0046 |
| **9** | **20260911_042935** | **8** | **alpha=0.5 (0.0625)** | **Yes** | **1.12e-3** | **−0.4448** | **−0.0034** | **−0.4434** | **−0.0032** |
| 10 | 20260911_144735 | 8 | alpha=0.2774 (0.0347, Optuna, rank fixed) | Yes | 8.23e-4 | −0.4453 | −0.0039 | −0.4439 | −0.0037 |
| **11** | **20260911_225626** | **4** | **alpha=0.1482 (0.0371, Optuna, rank+alpha joint)** | **Yes** | **1.56e-3** | **−0.4439** | **−0.0025** | **−0.4424** | **−0.0022** |
