# TabLoRA

Extending [TabM](https://arxiv.org/abs/2410.24210) (ICLR 2025) by replacing its BatchEnsemble linear layers with LoRA-style low-rank adapters, creating **TabLoRA**.

## What changed

**`paper/lib/deep.py`** — Added `LinearLoRAEnsemble` after `LinearEfficientEnsemble`. Each ensemble member shares a base weight W and learns a low-rank delta:

```
output_i = x @ W.T + x @ A_i.T @ B_i.T + bias_i
```

B is zero-initialised so the delta starts at zero. Controlled by `rank`.

**`paper/bin/model.py`** — Added `arch_type='tabm-lora'` and a `rank` parameter. Setting `arch_type` to `tabm-lora` in a config automatically replaces all linear layers in the backbone.

## Datasets

All data lives at `paper/data/<dataset>/`. Download on the cluster with:

```bash
cd ~/adapt-tabm/paper
python bin/data.py california   # repeat for each dataset
```

Available: `california`, `adult`, `churn`, `higgs-small`, `covtype2`, `otto`, `diamond`, `house`, `microsoft`, `black-friday`.

## Config

Configs are TOML files under `paper/exp/tabm-lora/<dataset>/0-evaluation/`.

Key fields for TabLoRA:

```toml
[model]
arch_type = "tabm-lora"
k    = 32    # number of ensemble heads
rank = 4     # LoRA rank — try 2, 4, 8, 16
```

Everything else (optimizer, backbone, data) mirrors the original TabM configs.

## Running on TC2 Cluster

> Requires NTU VPN. Connect before any SSH/rsync.

```bash
ssh -l <ntu-user-id> 10.96.189.12
```

### Transfer files

```bash
rsync -av --exclude='.git/' ~/Desktop/AI6103/adapt-tabm/paper/ \
    <ntu-user-id>@10.96.189.12:~/adapt-tabm/paper/
```

### SLURM scripts (submit from `paper/` on the cluster)

| Script | Purpose |
|--------|---------|
| `slurm/train_tablora.sh` | Train one seed |
| `slurm/evaluate_tablora.sh` | Run all 15 seeds |
| `slurm/ensemble_tablora.sh` | Ensemble + save summary log |

```bash
# One seed (smoke test)
sbatch slurm/train_tablora.sh \
    exp/tabm-lora/california/0-evaluation/0.toml \
    exp/tabm-lora/california/0-evaluation/0

# All 15 seeds
sbatch slurm/evaluate_tablora.sh exp/tabm-lora/california/0-evaluation

# Ensemble + summary (run after evaluate)
sbatch slurm/ensemble_tablora.sh exp/tabm-lora/california/0-evaluation
```

Monitor with `squeue -u <ntu-user-id>`. Logs go to `slurm/logs/`.

## Result layout

```
exp/tabm-lora/california/
├── 0-evaluation/
│   ├── 0.toml          # config for seed 0
│   ├── 0/report.json   # metrics, config, timing
│   └── ...             # seeds 1–14
└── 0-ensemble-5/
    ├── 0/report.json   # ensemble of seeds 0–4
    ├── 1/report.json   # seeds 5–9
    └── 2/report.json   # seeds 10–14
```

Summary logs are saved to `logs/results/tablora_<dataset>_<timestamp>.txt`.

## Reading results

```bash
# List datasets with results
python tools/lookup_results.py

# All 15 seeds for a dataset
python tools/lookup_results.py california --arch tabm-lora

# Side-by-side vs TabM baseline
python tools/lookup_results.py california --compare

# One specific seed
python tools/lookup_results.py california --arch tabm-lora --seed 3

# Generate summary log (from paper/)
python bin/summarize_results.py exp/tabm-lora/california/0-evaluation
```

## Current results — California (15 seeds, rank=4)

| Model | Mean test | Ensemble-5 |
|-------|-----------|------------|
| TabM (baseline) | −0.4414 | −0.4402 |
| TabLoRA (ours)  | −0.4988 | −0.4915 |

Score is negative RMSE, where higher score is better. The gap is expected for an untuned rank=4 first run.
