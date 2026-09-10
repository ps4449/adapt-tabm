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

## Experiment #3 — 2026-09-10 — Rank 8, input scaling, adapter scale 0.125

> Output: `california/r8-scale0125/scale0125-evaluation`

**Dataset:** California Housing — regression (median house value), metric: negative RMSE (higher = better).

This checkpoint combines the per-head input `ScaleEnsemble` with rank-8 LoRA adapters. The
effective adapter multiplier is `lora_alpha / rank = 1 / 8 = 0.125`.

### Configuration

| Parameter | Value |
|-----------|-------|
| arch_type | `tabm-lora` |
| k | 32 |
| rank | 8 |
| lora_alpha | 1.0 |
| effective adapter scale | 0.125 |
| lora_input_scaling | `true` |
| n_blocks | 3 |
| d_block | 400 |
| dropout | 0.2077 |
| optimizer | AdamW |
| lr | 8.72e-4 |
| weight_decay | 3.78e-2 |
| n_parameters | 888,736 |

### Per-seed results (15 seeds)

| Seed | Train | Val | Test |
|------|-------|-----|------|
| 0  | −0.3648 | −0.4536 | −0.4506 |
| 1  | −0.3468 | −0.4526 | −0.4494 |
| 2  | −0.3462 | −0.4522 | −0.4481 |
| 3  | −0.3556 | −0.4518 | −0.4482 |
| 4  | −0.3280 | −0.4496 | −0.4465 |
| 5  | −0.3464 | −0.4514 | −0.4497 |
| 6  | −0.3413 | −0.4521 | −0.4480 |
| 7  | −0.3455 | −0.4508 | −0.4480 |
| 8  | −0.3358 | −0.4515 | −0.4481 |
| 9  | −0.3201 | −0.4492 | −0.4473 |
| 10 | −0.3365 | −0.4498 | −0.4469 |
| 11 | −0.3321 | −0.4512 | −0.4481 |
| 12 | −0.3311 | −0.4496 | −0.4489 |
| 13 | −0.3262 | −0.4514 | −0.4499 |
| 14 | −0.3565 | −0.4515 | −0.4492 |
| **Mean** | **−0.3409** | **−0.4512** | **−0.4485** |
| **Std** | **0.0121** | **0.0012** | **0.0011** |

### Ensemble results (size 5)

| Ensemble | Seeds | Val | Test |
|----------|-------|-----|------|
| 0 | 0–4 | −0.4508 | −0.4472 |
| 1 | 5–9 | −0.4499 | −0.4470 |
| 2 | 10–14 | −0.4494 | −0.4472 |
| **Mean** | — | — | **−0.4472** |

### Analysis

Scale 0.125 was selected after a three-seed screen against rank-8 scales 0.25, 0.5 and 1.0.
On the full matched 15-seed comparison, it improves every test seed over scale 0.25. Mean test
improves from −0.4505 to −0.4485, and ensemble-5 improves from −0.4493 to −0.4472. This
checkpoint established that the rank-8 LoRA delta needed substantially weaker scaling than the
initial settings.

---

## Experiment #4 — 2026-09-10 — Rank 8, input scaling, adapter scale 0.0625

> Output: `california/r8-scale00625/scale00625-evaluation`

**Dataset:** California Housing — regression (median house value), metric: negative RMSE (higher = better).

This checkpoint halves the previous adapter multiplier while preserving the architecture,
optimizer and training protocol. The effective multiplier is
`lora_alpha / rank = 0.5 / 8 = 0.0625`.

### Configuration

| Parameter | Value |
|-----------|-------|
| arch_type | `tabm-lora` |
| k | 32 |
| rank | 8 |
| lora_alpha | 0.5 |
| effective adapter scale | 0.0625 |
| lora_input_scaling | `true` |
| n_blocks | 3 |
| d_block | 400 |
| dropout | 0.2077 |
| optimizer | AdamW |
| lr | 8.72e-4 |
| weight_decay | 3.78e-2 |
| n_parameters | 888,736 |

### Per-seed results (15 seeds)

| Seed | Train | Val | Test |
|------|-------|-----|------|
| 0  | −0.3218 | −0.4483 | −0.4457 |
| 1  | −0.3643 | −0.4514 | −0.4485 |
| 2  | −0.3355 | −0.4504 | −0.4459 |
| 3  | −0.3364 | −0.4488 | −0.4450 |
| 4  | −0.3595 | −0.4495 | −0.4458 |
| 5  | −0.3322 | −0.4490 | −0.4448 |
| 6  | −0.3404 | −0.4489 | −0.4465 |
| 7  | −0.3418 | −0.4480 | −0.4466 |
| 8  | −0.3479 | −0.4517 | −0.4482 |
| 9  | −0.3414 | −0.4490 | −0.4460 |
| 10 | −0.3421 | −0.4491 | −0.4468 |
| 11 | −0.3439 | −0.4496 | −0.4448 |
| 12 | −0.3360 | −0.4503 | −0.4474 |
| 13 | −0.3625 | −0.4525 | −0.4481 |
| 14 | −0.3497 | −0.4479 | −0.4462 |
| **Mean** | **−0.3437** | **−0.4496** | **−0.4464** |
| **Std** | **0.0112** | **0.0013** | **0.0012** |

### Ensemble results (size 5)

| Ensemble | Seeds | Val | Test |
|----------|-------|-----|------|
| 0 | 0–4 | −0.4482 | −0.4447 |
| 1 | 5–9 | −0.4482 | −0.4452 |
| 2 | 10–14 | −0.4486 | −0.4454 |
| **Mean** | — | — | **−0.4451** |

### Analysis

Scale 0.0625 is the strongest TabLoRA checkpoint in this study. Relative to scale 0.125, mean
validation improves by 0.0016, mean test by 0.0021, and ensemble-5 test by 0.0021. Fourteen of
the 15 matched test seeds improve; seed 8 is lower by 0.0001 at the reported precision.

Compared with the tuned TabM baseline, the remaining gap is 0.0038 on validation, 0.0050 on
mean test and 0.0049 on ensemble-5 test. The configuration is frozen at this checkpoint to avoid
continuing to select adapter scales after observing test results.

---

## Input-scaled rank-8 checkpoint comparison

| Model | Params | Mean Val | Mean Test | Std (Test) | Ensemble-5 Test |
|-------|--------|----------|-----------|------------|-----------------|
| TabM baseline | 438,688 | −0.4458 | −0.4414 | ±0.0012 | −0.4402 |
| Tejasvita rank 4, scale 0.25 | 631,712 | −0.4539 | −0.4511 | ±0.0014 | −0.4500 |
| Rank 8, scale 0.25 | 888,736 | −0.4533 | −0.4505 | ±0.0011 | −0.4493 |
| **Rank 8, scale 0.125** | **888,736** | **−0.4512** | **−0.4485** | **±0.0011** | **−0.4472** |
| **Rank 8, scale 0.0625 (selected)** | **888,736** | **−0.4496** | **−0.4464** | **±0.0012** | **−0.4451** |

The two rank-8 checkpoints are retained because they capture the decisive part of the adapter
scale ablation. Scale 0.125 first demonstrated a consistent gain over 0.25; scale 0.0625 then
reduced the remaining test gap to TabM from 0.0071 to 0.0050.

---

## Master Summary Table

> All scores are negative RMSE — higher is better.
> Δ values compare the listed checkpoint with TabM; negative means TabLoRA is worse.

| # | Date | Checkpoint | Rank | Params | Mean Val | Mean Test | Test Std | Ens-5 | Δ Test | Δ Ens |
|---|------|------------|------|--------|----------|-----------|----------|-------|--------|-------|
| 1 | 2026-09-04 | Original LoRA | 4 | 631,456 | −0.5003 | −0.4988 | 0.0034 | −0.4915 | −0.0574 | −0.0513 |
| 2 | 2026-09-06 | Original LoRA | 16 | 1,402,528 | −0.4914 | −0.4926 | 0.0027 | −0.4871 | −0.0512 | −0.0469 |
| 3 | 2026-09-10 | Input scale, adapter 0.125 | 8 | 888,736 | −0.4512 | −0.4485 | 0.0011 | −0.4472 | −0.0071 | −0.0070 |
| 4 | 2026-09-10 | Input scale, adapter 0.0625 | 8 | 888,736 | −0.4496 | −0.4464 | 0.0012 | −0.4451 | −0.0050 | −0.0049 |
