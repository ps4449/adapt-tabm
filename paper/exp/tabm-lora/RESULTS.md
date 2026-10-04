# TabLoRA — Experiment Summary

## Motivation

TabM uses BatchEnsemble to build k ensemble heads cheaply: each head shares a weight matrix W and learns only per-head scaling vectors r and s. The per-head capacity is limited to element-wise scaling, which is expressive but inflexible in terms of subspace coverage.

TabLoRA replaces this with LoRA-style low-rank adapters. Each head learns an additive low-rank delta (A_i, B_i) on top of the shared W, giving it a richer low-dimensional subspace to personalise rather than just scaling activations. The hypothesis is that this leads to more diverse and complementary ensemble members.

---

## Experiment #1 — 2026-09-04

**Log:** `logs/results/tablora_california_20260904_214457.txt`

**Dataset:** California Housing — regression (median house value)  
**Metric:** negative RMSE (higher = better)

### Configuration

| Parameter | TabM (baseline) | TabLoRA |
|-----------|-----------------|---------|
| arch_type | tabm | tabm-lora |
| k | 32 | 32 |
| rank | — | 4 |
| n_blocks | 3 | 3 |
| d_block | 400 | 400 |
| dropout | 0.2077 | 0.2077 |
| lr | 8.72e-4 | 8.72e-4 |
| weight_decay | 3.78e-2 | 3.78e-2 |
| n_parameters | 438,688 | 631,456 (+44%) |
| Hyperparams tuned | Yes (100 Optuna trials) | No — borrowed from TabM |

### Per-seed Results (15 seeds)

| Seed | TabLoRA Val | TabLoRA Test |
|------|-------------|--------------|
| 0 | −0.5024 | −0.5054 |
| 1 | −0.5004 | −0.5019 |
| 2 | −0.5015 | −0.5052 |
| 3 | −0.4982 | −0.4985 |
| 4 | −0.5011 | −0.5013 |
| 5 | −0.5017 | −0.4969 |
| 6 | −0.4975 | −0.4943 |
| 7 | −0.5000 | −0.4960 |
| 8 | −0.4993 | −0.5006 |
| 9 | −0.4973 | −0.4974 |
| 10 | −0.5012 | −0.4971 |
| 11 | −0.4993 | −0.5004 |
| 12 | −0.4997 | −0.4966 |
| 13 | −0.5053 | −0.4949 |
| 14 | −0.4999 | −0.4953 |

### Summary

| Model | Mean Val | Mean Test | Std (Test) | Ensemble-5 Test |
|-------|----------|-----------|-----------|-----------------|
| TabM (baseline) | −0.4458 | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA | −0.5003 | −0.4988 | ±0.0034 | −0.4915 |
| **diff.** | **−0.0545** | **−0.0574** | **3× higher** | **−0.0513** |

### Analysis

**Why the gap exists.**
The TabLoRA run reused hyperparameters tuned specifically for BatchEnsemble dynamics. LoRA adapters have a product structure (A × B) that typically benefits from a higher learning rate and different regularisation. Running with TabM's lr/wd is effectively mis-specified for the new architecture.

**Why variance is higher.**
std ±0.0034 is nearly 3x TabM's ±0.0012. The adapter matrices are sensitive to initialisation under this configuration - a sign that rank or regularisation needs tuning.

**Parameter count.**
At rank=4, TabLoRA uses 192K more parameters than TabM (+44%). This is partly structural: each LoRA delta costs rank × (in + out) vs BatchEnsemble's in + out. Rank=2 would bring this to ~+25% — worth testing before concluding LoRA is less efficient.

**Ensemble diversity.**
Ensemble-5 (−0.4915) is noticeably better than the single-model mean (−0.4988), confirming heads are still learning diverse solutions despite the overall gap.

---

## Experiment #2 — 2026-09-05

**Dataset:** California Housing — regression (median house value)  
**Metric:** negative RMSE (higher = better)

### Configuration

| Parameter | TabM (baseline) | TabLoRA |
|-----------|-----------------|---------|
| arch_type | tabm | tabm-lora |
| k | 32 | 32 |
| rank | — | 8 |
| n_blocks | 3 | 3 |
| d_block | 400 | 400 |
| dropout | 0.2077 | 0.2077 |
| lr | 8.72e-4 | 8.72e-4 |
| weight_decay | 3.78e-2 | 3.78e-2 |
| n_parameters | 438,688 | 888,480 (+103%) |
| Hyperparams tuned | Yes (100 Optuna trials) | No — borrowed from TabM |

### Per-seed Results (15 seeds)

| Seed | TabLoRA Val | TabLoRA Test |
|------|-------------|--------------|
| 0 | −0.4970 | −0.4983 |
| 1 | −0.4957 | −0.4963 |
| 2 | −0.4986 | −0.4966 |
| 3 | −0.4999 | −0.5010 |
| 4 | −0.4997 | −0.4942 |
| 5 | −0.4970 | −0.4954 |
| 6 | −0.4939 | −0.4927 |
| 7 | −0.4976 | −0.4880 |
| 8 | −0.4984 | −0.4986 |
| 9 | −0.4975 | −0.4995 |
| 10 | −0.4956 | −0.5002 |
| 11 | −0.4940 | −0.4959 |
| 12 | −0.4951 | −0.4949 |
| 13 | −0.4998 | −0.4972 |
| 14 | −0.4974 | −0.4963 |

### Summary

| Model | Mean Val | Mean Test | Std (Test) | Ensemble-5 Test |
|-------|----------|-----------|-----------|-----------------|
| TabM (baseline) | −0.4458 | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank 4 | −0.5003 | −0.4988 | ±0.0034 | −0.4915 |
| TabLoRA rank 8 | −0.4971 | −0.4963 | ±0.0031 | −0.4894 |
| **rank 8 vs rank 4** | **+0.0032** | **+0.0025** | **−0.0003** | **+0.0021** |
| **rank 8 vs TabM** | **−0.0513** | **−0.0549** | **+0.0019** | **−0.0492** |

### Analysis

**Increasing rank continues to help.**
Moving from rank 4 to rank 8 improves the mean test score from −0.4988 to −0.4963, a gain of 0.0025. The five-model ensemble score also improves from −0.4915 to −0.4894 (+0.0021), while test-score variability decreases slightly from ±0.0034 to ±0.0031. This provides additional evidence that the low-rank capacity of the adapters is a meaningful bottleneck at rank 4.

**The improvement is modest relative to the parameter cost.**
Rank 8 uses 888,480 parameters, approximately 41% more than rank 4 and 103% more than the TabM baseline. The 0.0025 test-score improvement over rank 4 therefore comes at a substantial increase in model size.

**The gap to TabM remains.**
Despite the improvement over rank 4, rank 8 remains 0.0549 below the tuned TabM baseline on mean test score and 0.0492 below it for the five-model ensemble. As with the other TabLoRA experiments, the optimizer hyperparameters were inherited from TabM rather than tuned for the LoRA architecture, so this does not yet represent a tuned comparison between the two methods.

**Ensembling remains beneficial.**
The rank-8 ensemble improves from a single-model mean of −0.4963 to −0.4894, a gain of 0.0069. This gain lies between rank 4 (+0.0073) and rank 16 (+0.0055), suggesting that independently trained TabLoRA models continue to learn complementary solutions as rank increases.

---

## Experiment #3 — 2026-09-06

**Log:** `logs/results/tablora_california_20260906_153246.txt`

**Config:** `california/0-evaluation/0-rank16.toml`

**Dataset:** California Housing — regression (median house value)  
**Metric:** negative RMSE (higher = better)

### Configuration

| Parameter | TabM (baseline) | TabLoRA rank 4 | TabLoRA rank 16 |
|-----------|-----------------|---|---|
| arch_type | tabm | tabm-lora | tabm-lora |
| k | 32 | 32 | 32 |
| rank | — | 4 | 16 |
| n_blocks | 3 | 3 | 3 |
| d_block | 400 | 400 | 400 |
| dropout | 0.2077 | 0.2077 | 0.2077 |
| lr | 8.72e-4 | 8.72e-4 | 8.72e-4 |
| weight_decay | 3.78e-2 | 3.78e-2 | 3.78e-2 |
| n_parameters | 438,688 | 631,456 | 1,402,528 |
| Hyperparams tuned | Yes (100 Optuna trials) | No — borrowed from TabM | No — borrowed from TabM |

### Per-seed Results (15 seeds)

| Seed | TabLoRA Val | TabLoRA Test |
|------|-------------|--------------|
| 0 | −0.4918 | −0.4933 |
| 1 | −0.4871 | −0.4867 |
| 2 | −0.4928 | −0.4926 |
| 3 | −0.4881 | −0.4942 |
| 4 | −0.4883 | −0.4968 |
| 5 | −0.4901 | −0.4881 |
| 6 | −0.4924 | −0.4921 |
| 7 | −0.4931 | −0.4919 |
| 8 | −0.4965 | −0.4936 |
| 9 | −0.4923 | −0.4911 |
| 10 | −0.4932 | −0.4903 |
| 11 | −0.4921 | −0.4975 |
| 12 | −0.4907 | −0.4945 |
| 13 | −0.4941 | −0.4926 |
| 14 | −0.4889 | −0.4935 |

### Summary

| Model | Mean Val | Mean Test | Std (Test) | Ensemble-5 Test |
|-------|----------|-----------|-----------|-----------------|
| TabM (baseline) | −0.4458 | −0.4414 | ±0.0012 | −0.4402 |
| TabLoRA rank 4 | −0.5003 | −0.4988 | ±0.0034 | −0.4915 |
| TabLoRA rank 16 | −0.4914 | −0.4926 | ±0.0027 | −0.4871 |
| **rank 16 vs rank 4** | **+0.0089** | **+0.0062** | **−0.0007** | **+0.0044** |
| **rank 16 vs TabM** | **−0.0456** | **−0.0512** | **+0.0015** | **−0.0469** |

### Analysis

**Higher rank helps.**
Increasing the LoRA rank from 4 to 16 improves the mean test score by 0.0062 and the five-model ensemble score by 0.0044. Test-score variability also falls from ±0.0034 to ±0.0027.

**The baseline gap remains.**
Rank 16 does not close the gap to the tuned TabM baseline: its mean test score remains 0.0512 lower, and its ensemble score remains 0.0469 lower. Both TabLoRA runs reuse hyperparameters tuned for TabM, so this comparison isolates the effect of rank but not TabLoRA's tuned potential.

**Parameter cost.**
Rank 16 uses 1,402,528 parameters: 122% more than rank 4 and 220% more than TabM. The accuracy gain therefore comes with a substantial parameter-efficiency tradeoff.

**Ensemble gain.**
The rank-16 five-model ensemble improves over its single-model mean by approximately 0.0055. This confirms that independently seeded rank-16 models remain complementary, although the gain is smaller than the improvement observed at rank 4.

---

## Cross-Rank Analysis

The three experiments form a rank ablation at ranks 4, 8, and 16 while holding the remaining architecture and optimizer settings fixed.

### Rank Progression

| Rank | Parameters | Mean Val | Mean Test | Test Std | Ensemble-5 |
|------|------------|----------|-----------|----------|-----------|
| 4 | 631,456 | −0.5003 | −0.4988 | ±0.0034 | −0.4915 |
| 8 | 888,480 | −0.4971 | −0.4963 | ±0.0031 | −0.4894 |
| 16 | 1,402,528 | −0.4914 | −0.4926 | ±0.0027 | −0.4871 |

Performance improves monotonically with rank: mean test score increases from −0.4988 at rank 4, to −0.4963 at rank 8, to −0.4926 at rank 16. Ensemble performance follows the same ordering. Test variance also decreases monotonically from ±0.0034 to ±0.0031 to ±0.0027.

The gains, however, are expensive in parameter count. Doubling rank from 4 to 8 adds 257,024 parameters for a +0.0025 test improvement, while doubling it again from 8 to 16 adds 514,048 parameters for a +0.0037 improvement. Even rank 16, at more than three times the parameter count of TabM, remains substantially behind the tuned baseline.

Taken together, these results suggest that adapter rank is a genuine capacity constraint, but rank alone is unlikely to explain the TabM–TabLoRA performance gap. The next higher-value experiment is therefore hyperparameter tuning—particularly learning rate and weight decay—rather than continuing to increase rank.

---

## Experiment #4 - 2026-09-07 - Adapter scale 0.25

**Log:** [`tablora_california_20260907_023250.txt`](../../logs/results/tablora_california_20260907_023250.txt)

**Dataset:** California Housing (`data/california`), regression.

**Metric:** negative RMSE (higher is better).

### Configuration

| Parameter | Value |
|-----------|-------|
| arch_type | tabm-lora |
| k | 32 |
| rank | 4 |
| adapter_scale | 0.25 |
| lora_input_scaling | true |
| n_blocks | 3 |
| d_block | 400 |
| dropout | 0.2077 |
| optimizer | AdamW |
| lr | 8.72e-4 |
| weight_decay | 3.78e-2 |
| n_parameters | 631,712 |

The adapter-scale assignments follow the user-provided file order; the logs do not print `adapter_scale` or `lora_input_scaling`. Input scaling is recorded from the experiment setup described in this session. Other configuration values are reported at the logs' precision.

### Per-seed Results (15 seeds)

| Seed | Train | Val | Test |
|------|-------|-----|------|
| 0 | -0.3444 | -0.4527 | -0.4496 |
| 1 | -0.3456 | -0.4540 | -0.4498 |
| 2 | -0.3454 | -0.4520 | -0.4505 |
| 3 | -0.3171 | -0.4514 | -0.4506 |
| 4 | -0.3433 | -0.4535 | -0.4505 |
| 5 | -0.3379 | -0.4544 | -0.4508 |
| 6 | -0.3400 | -0.4539 | -0.4495 |
| 7 | -0.3433 | -0.4532 | -0.4499 |
| 8 | -0.3650 | -0.4563 | -0.4539 |
| 9 | -0.3429 | -0.4522 | -0.4502 |
| 10 | -0.3614 | -0.4554 | -0.4533 |
| 11 | -0.3590 | -0.4567 | -0.4529 |
| 12 | -0.3357 | -0.4542 | -0.4505 |
| 13 | -0.3522 | -0.4559 | -0.4520 |
| 14 | -0.3391 | -0.4526 | -0.4524 |
| Mean | -0.3448 | -0.4539 | -0.4511 |
| Std | 0.0113 | 0.0016 | 0.0014 |

### Ensemble Results (size=5)

| Ensemble | Val | Test |
|----------|-----|------|
| 0 | -0.4516 | -0.4490 |
| 1 | -0.4530 | -0.4498 |
| 2 | -0.4540 | -0.4512 |
| Mean | Not reported | -0.4500 |

---

## Experiment #5 - 2026-09-07 - Adapter scale 0.5

**Log:** [`tablora_california_20260907_023323.txt`](../../logs/results/tablora_california_20260907_023323.txt)

**Dataset:** California Housing (`data/california`), regression.

**Metric:** negative RMSE (higher is better).

### Configuration

| Parameter | Value |
|-----------|-------|
| arch_type | tabm-lora |
| k | 32 |
| rank | 4 |
| adapter_scale | 0.5 |
| lora_input_scaling | true |
| n_blocks | 3 |
| d_block | 400 |
| dropout | 0.2077 |
| optimizer | AdamW |
| lr | 8.72e-4 |
| weight_decay | 3.78e-2 |
| n_parameters | 631,712 |

The adapter-scale assignments follow the user-provided file order; the logs do not print `adapter_scale` or `lora_input_scaling`. Input scaling is recorded from the experiment setup described in this session. Other configuration values are reported at the logs' precision.

### Per-seed Results (15 seeds)

| Seed | Train | Val | Test |
|------|-------|-----|------|
| 0 | -0.3500 | -0.4548 | -0.4537 |
| 1 | -0.3567 | -0.4548 | -0.4526 |
| 2 | -0.3398 | -0.4532 | -0.4532 |
| 3 | -0.3463 | -0.4565 | -0.4542 |
| 4 | -0.3378 | -0.4545 | -0.4515 |
| 5 | -0.3513 | -0.4557 | -0.4537 |
| 6 | -0.3570 | -0.4560 | -0.4520 |
| 7 | -0.3385 | -0.4542 | -0.4515 |
| 8 | -0.3490 | -0.4544 | -0.4547 |
| 9 | -0.3408 | -0.4533 | -0.4513 |
| 10 | -0.3480 | -0.4552 | -0.4530 |
| 11 | -0.3612 | -0.4564 | -0.4558 |
| 12 | -0.3604 | -0.4591 | -0.4561 |
| 13 | -0.3546 | -0.4577 | -0.4552 |
| 14 | -0.3638 | -0.4578 | -0.4563 |
| Mean | -0.3503 | -0.4556 | -0.4536 |
| Std | 0.0083 | 0.0016 | 0.0016 |

### Ensemble Results (size=5)

| Ensemble | Val | Test |
|----------|-----|------|
| 0 | -0.4539 | -0.4522 |
| 1 | -0.4539 | -0.4517 |
| 2 | -0.4564 | -0.4544 |
| Mean | Not reported | -0.4528 |

---

## Experiment #6 - 2026-09-07 - Adapter scale 1.0

**Log:** [`tablora_california_20260907_015913.txt`](../../logs/results/tablora_california_20260907_015913.txt)

**Dataset:** California Housing (`data/california`), regression.

**Metric:** negative RMSE (higher is better).

### Configuration

| Parameter | Value |
|-----------|-------|
| arch_type | tabm-lora |
| k | 32 |
| rank | 4 |
| adapter_scale | 1.0 |
| lora_input_scaling | true |
| n_blocks | 3 |
| d_block | 400 |
| dropout | 0.2077 |
| optimizer | AdamW |
| lr | 8.72e-4 |
| weight_decay | 3.78e-2 |
| n_parameters | 631,712 |

The adapter scale is identified by the user, and input scaling follows the experiment setup described in this session; neither option is printed in the log. This run predates experiments 4 and 5 but is numbered by order of documentation.

### Per-seed Results (15 seeds)

| Seed | Train | Val | Test |
|------|-------|-----|------|
| 0 | -0.3564 | -0.4560 | -0.4554 |
| 1 | -0.3623 | -0.4573 | -0.4546 |
| 2 | -0.3438 | -0.4552 | -0.4532 |
| 3 | -0.3621 | -0.4568 | -0.4563 |
| 4 | -0.3469 | -0.4566 | -0.4543 |
| 5 | -0.3498 | -0.4570 | -0.4544 |
| 6 | -0.3572 | -0.4576 | -0.4545 |
| 7 | -0.3536 | -0.4565 | -0.4546 |
| 8 | -0.3620 | -0.4582 | -0.4557 |
| 9 | -0.3541 | -0.4561 | -0.4535 |
| 10 | -0.3697 | -0.4579 | -0.4566 |
| 11 | -0.3553 | -0.4570 | -0.4547 |
| 12 | -0.3570 | -0.4584 | -0.4563 |
| 13 | -0.3767 | -0.4624 | -0.4599 |
| 14 | -0.3563 | -0.4568 | -0.4553 |
| Mean | -0.3576 | -0.4573 | -0.4553 |
| Std | 0.0081 | 0.0016 | 0.0016 |

### Ensemble Results (size=5)

| Ensemble | Val | Test |
|----------|-----|------|
| 0 | -0.4557 | -0.4540 |
| 1 | -0.4563 | -0.4537 |
| 2 | -0.4576 | -0.4556 |
| Mean | Not reported | -0.4545 |

---

## Adapter-Scale Comparison

All scores below are negative RMSE (higher is better). All three scale settings are documented in the linked logs in experiments 4-6. All three use rank 4 and input scaling, with 631,712 parameters.

| Adapter scale | Mean Val | Mean Test | Test Std | Ensemble-5 Test |
|---------------|----------|-----------|----------|-----------------|
| 1.0 (reference) | -0.4573 | -0.4553 | 0.0016 | -0.4545 |
| 0.5 | -0.4556 | -0.4536 | 0.0016 | -0.4528 |
| **0.25** | **-0.4539** | **-0.4511** | **0.0014** | **-0.4500** |

Scale 0.25 has the best mean validation score among these tested settings. Relative to scale 1.0, it improves mean validation score by 0.0034, mean test score by 0.0042, and ensemble-5 test score by 0.0045, without adding parameters. Scale 0.5 is intermediate on all three metrics.

The scale-0.25 model remains behind the tuned TabM baseline by 0.0097 on mean test score and 0.0098 on ensemble-5 test score. These experiments support reducing adapter strength under the current training settings; they do not establish a globally optimal scale or isolate the mechanism. Ensemble-5 averages independently trained models, whereas each individual model already averages its 32 internal heads.

---

## Master Summary Table

All scores are negative RMSE — **higher is better**.

*Δ = TabLoRA − TabM (negative means TabLoRA is worse)*

| # | Date | Dataset | Rank | Params | TabM Test | TabLoRA Test | Δ | TabM Ens-5 | TabLoRA Ens-5 | Δ Ens |
|---|------|---------|------|--------|-----------|--------------|-------|-----------|---------------|-------|
| 1 | 2026-09-04 | california | 4 | 631,456 | −0.4414 | −0.4988 | −0.0574 | −0.4402 | −0.4915 | −0.0513 |
| 2 | 2026-09-05 | california | 8 | 888,480 | −0.4414 | −0.4963 | −0.0549 | −0.4402 | −0.4894 | −0.0492 |
| 3 | 2026-09-06 | california | 16 | 1,402,528 | −0.4414 | −0.4926 | −0.0512 | −0.4402 | −0.4871 | −0.0469 |
| 4 | 2026-09-07 | california | 4 | 631,712 | -0.4414 | -0.4511 | -0.0097 | -0.4402 | -0.4500 | -0.0098 |
| 5 | 2026-09-07 | california | 4 | 631,712 | -0.4414 | -0.4536 | -0.0122 | -0.4402 | -0.4528 | -0.0126 |
| 6 | 2026-09-07 | california | 4 | 631,712 | -0.4414 | -0.4553 | -0.0139 | -0.4402 | -0.4545 | -0.0143 |

Experiments 4 and 5 use input scaling with adapter scales 0.25 and 0.5, respectively; experiment 6 uses input scaling with adapter scale 1.0. Experiments 1-3 are the original rank ablation without input scaling.
