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

## Master Summary Table

> All scores are negative RMSE — higher is better.
> Δ = TabLoRA − TabM (negative means TabLoRA is worse).

| # | Date | Dataset | Rank | Params | TabM Test | TabLoRA Test | Δ | TabM Ens-5 | TabLoRA Ens-5 | Δ Ens |
|---|------|---------|------|--------|-----------|--------------|---|------------|---------------|-------|
| 1 | 2026-09-04 | california | 4 | 631,456 | −0.4414 | −0.4988 | −0.0574 | −0.4402 | −0.4915 | −0.0513 |
| 2 | 2026-09-06 | california | 16 | 1,402,528 | −0.4414 | −0.4926 | −0.0512 | −0.4402 | −0.4871 | −0.0469 |
