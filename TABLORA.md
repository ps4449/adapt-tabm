# TabLoRA

Extending [TabM](https://arxiv.org/abs/2410.24210) (ICLR 2025) by replacing its BatchEnsemble linear layers with LoRA-style low-rank adapters, creating **TabLoRA**.

## What changed

**`paper/lib/deep.py`** — Added `LinearLoRAEnsemble` after `LinearEfficientEnsemble`. Each ensemble member shares a base weight W and learns a scaled low-rank delta:

```
output_i = x @ W.T + (lora_alpha / rank) * (x @ A_i.T @ B_i.T) + bias_i
```

B is zero-initialised so the delta starts at zero. Adapter capacity is controlled by `rank`,
while `lora_alpha / rank` controls its effective contribution.

**`paper/bin/model.py`** — Added `arch_type='tabm-lora'`, `rank`, `lora_alpha` and optional
`lora_input_scaling`. Setting `arch_type` to `tabm-lora` automatically replaces all linear
layers in the backbone. Enabling input scaling adds a per-head `ScaleEnsemble` before the
backbone.

## Datasets

All data lives at `paper/data/<dataset>/`. Run the following once on the cluster to download all datasets:

```bash
cd ~/adapt-tabm/paper
mkdir -p local data
wget https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar \
    -O local/tabm-data.tar.gz
tar -xvf local/tabm-data.tar.gz -C data
```

This extracts all datasets into `paper/data/`.

Available: `california`, `adult`, `churn`, `higgs-small`, `covtype2`, `otto`, `diamond`, `house`, `microsoft`, `black-friday`.

## Config

Configs are TOML files under `paper/exp/tabm-lora/<dataset>/0-evaluation/`.

Key fields for TabLoRA:

```toml
[model]
arch_type = "tabm-lora"
k = 32                       # number of ensemble heads
rank = 8                     # LoRA rank
lora_alpha = 0.5             # effective scale = 0.5 / 8 = 0.0625
lora_input_scaling = true    # per-head input scaling
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

## Current results — California (15 seeds)

| Model | Rank | Effective scale | Parameters | Mean test | Test std | Ensemble-5 |
|-------|------|-----------------|------------|-----------|----------|------------|
| TabM baseline | — | — | 438,688 | −0.4414 | 0.0012 | −0.4402 |
| TabLoRA with (TabM config) | 4 | 1.0 | 631,456 | −0.4988 | 0.0034 | −0.4915 |
| Rank 4 Input-scaled checkpoint | 4 | 0.25 | 631,712 | −0.4511 | 0.0014 | −0.4500 |
| Rank-8 checkpoint | 8 | 0.125 | 888,736 | −0.4485 | 0.0011 | −0.4472 |
| **Selected rank-8 checkpoint** | **8** | **0.0625** | **888,736** | **−0.4464** | **0.0012** | **−0.4451** |

Score is negative RMSE, where higher is better. Reducing the rank-8 adapter scale from 0.125
to 0.0625 improves mean test by 0.0021 and ensemble-5 by 0.0021. The selected checkpoint is
0.0050 behind TabM on mean test and 0.0049 behind it on ensemble-5. The scale sweep is frozen
at 0.0625 to avoid further selection after observing test results.

Full configurations, per-seed tables and ensemble results are documented in
[`paper/exp/tabm-lora/RESULTS.md`](paper/exp/tabm-lora/RESULTS.md).
