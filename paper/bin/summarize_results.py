"""
Summarize TabLoRA evaluation results across seeds and ensemble.

Run after bin/evaluate.py and bin/ensemble.py have completed.
Produces a human-readable summary log saved to logs/results/.

Usage (from paper/ directory):
    python bin/summarize_results.py <evaluation_dir> [--ensemble_size N]

Example:
    python bin/summarize_results.py exp/tabm-lora/california/0-evaluation
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads((path / 'report.json').read_text())


def summarize(evaluation_dir: Path, ensemble_size: int = 5) -> None:
    evaluation_dir = evaluation_dir.resolve()
    assert evaluation_dir.exists(), f'Directory not found: {evaluation_dir}'

    # -------------------------------------------------------------------------
    # Collect per-seed results
    # -------------------------------------------------------------------------
    seed_dirs = sorted(
        [d for d in evaluation_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    assert seed_dirs, 'No seed results found.'

    seed_reports = [load_report(d) for d in seed_dirs]
    first = seed_reports[0]

    # Pull config info from first seed
    config = first['config']
    model_cfg = config['model']
    backbone_cfg = model_cfg['backbone']
    dataset_path = config['data']['path']
    dataset_name = dataset_path.split('/')[-1]

    train_scores = [r['metrics']['train']['score'] for r in seed_reports]
    val_scores   = [r['metrics']['val']['score']   for r in seed_reports]
    test_scores  = [r['metrics']['test']['score']  for r in seed_reports]

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    # -------------------------------------------------------------------------
    # Collect ensemble results (if available)
    # -------------------------------------------------------------------------
    ensemble_dir = evaluation_dir.with_name(
        evaluation_dir.name.replace('evaluation', f'ensemble-{ensemble_size}')
    )
    ensemble_reports = []
    if ensemble_dir.exists():
        ensemble_reports = [
            load_report(d)
            for d in sorted(ensemble_dir.iterdir(), key=lambda d: d.name)
            if d.is_dir() and (d / 'report.json').exists()
        ]

    # -------------------------------------------------------------------------
    # Build summary text
    # -------------------------------------------------------------------------
    lines = []
    sep = '=' * 60

    lines += [
        sep,
        'TabLoRA Evaluation Summary',
        f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        sep,
        '',
        '--- Dataset ---',
        f'  Name         : {dataset_name}',
        f'  Path         : {dataset_path}',
        '',
        '--- Model Configuration ---',
        f'  arch_type    : {model_cfg.get("arch_type")}',
        f'  k            : {model_cfg.get("k")}',
        f'  rank         : {model_cfg.get("rank")}',
        f'  lora_alpha   : {model_cfg.get("lora_alpha", "N/A")}',
        f'  adapter_scale: {model_cfg.get("adapter_scale", "N/A")}',
        f'  input_scaling: {model_cfg.get("lora_input_scaling", False)}',
        f'  n_blocks     : {backbone_cfg.get("n_blocks")}',
        f'  d_block      : {backbone_cfg.get("d_block")}',
        f'  dropout      : {backbone_cfg.get("dropout"):.4f}',
        f'  n_parameters : {first.get("n_parameters", "N/A")}',
        '',
        '--- Optimizer ---',
        f'  type         : {config["optimizer"]["type"]}',
        f'  lr           : {config["optimizer"]["lr"]:.2e}',
        f'  weight_decay : {config["optimizer"]["weight_decay"]:.2e}',
        '',
        f'--- Per-Seed Results ({len(seed_dirs)} seeds) ---',
        f'  {"Seed":<6} {"Train":>10} {"Val":>10} {"Test":>10}',
        f'  {"-"*6} {"-"*10} {"-"*10} {"-"*10}',
    ]
    for i, r in enumerate(seed_reports):
        lines.append(
            f'  {i:<6} {r["metrics"]["train"]["score"]:>10.4f}'
            f' {r["metrics"]["val"]["score"]:>10.4f}'
            f' {r["metrics"]["test"]["score"]:>10.4f}'
        )
    lines += [
        f'  {"-"*6} {"-"*10} {"-"*10} {"-"*10}',
        f'  {"Mean":<6} {mean(train_scores):>10.4f} {mean(val_scores):>10.4f} {mean(test_scores):>10.4f}',
        f'  {"Std":<6} {std(train_scores):>10.4f}  {std(val_scores):>10.4f}  {std(test_scores):>10.4f}',
        '',
    ]

    if ensemble_reports:
        lines.append(f'--- Ensemble Results (size={ensemble_size}) ---')
        lines.append(f'  {"Ensemble":<10} {"Val":>10} {"Test":>10}')
        lines.append(f'  {"-"*10} {"-"*10} {"-"*10}')
        ens_test_scores = []
        for i, r in enumerate(ensemble_reports):
            val  = r['metrics']['val']['score'] if 'metrics' in r else r['scores']['val']
            test = r['metrics']['test']['score'] if 'metrics' in r else r['scores']['test']
            ens_test_scores.append(test)
            lines.append(f'  {i:<10} {val:>10.4f} {test:>10.4f}')
        lines += [
            f'  {"-"*10} {"-"*10} {"-"*10}',
            f'  {"Mean":<10} {"":>10} {mean(ens_test_scores):>10.4f}',
            '',
        ]
    else:
        lines.append('  (No ensemble results found — run bin/ensemble.py first)')
        lines.append('')

    lines.append(sep)
    summary = '\n'.join(lines)

    # -------------------------------------------------------------------------
    # Save to logs/results/
    # -------------------------------------------------------------------------
    log_dir = Path('logs/results')
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'tablora_{dataset_name}_{timestamp}.txt'
    log_file.write_text(summary)

    print(summary)
    print(f'Saved to: {log_file}')


if __name__ == '__main__':
    if __name__ != '__main__' or not Path.cwd().joinpath('pixi.toml').exists():
        pass  # allow import without assertion
    else:
        assert Path.cwd().joinpath(
            'pixi.toml'
        ).exists(), 'Must be run from the paper/ directory'

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_dir', type=Path)
    parser.add_argument('--ensemble_size', type=int, default=5)
    args = parser.parse_args()
    summarize(args.evaluation_dir, args.ensemble_size)
