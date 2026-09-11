"""
Look up and compare TabM vs TabLoRA results for a given dataset and optional seed.

Usage (from paper/ directory):
    python tools/lookup_results.py <dataset>
    python tools/lookup_results.py <dataset> --seed <N>
    python tools/lookup_results.py <dataset> --arch tabm-lora
    python tools/lookup_results.py <dataset> --arch tabm-nora-init
    python tools/lookup_results.py <dataset> --arch tabm-nora
    python tools/lookup_results.py <dataset> --compare

Examples:
    python tools/lookup_results.py california
    python tools/lookup_results.py california --seed 3
    python tools/lookup_results.py california --compare
    python tools/lookup_results.py adult --arch tabm-lora
"""

import argparse
import json
from pathlib import Path


EXP_ROOT = Path('exp')
ARCHS = [
    'tabm',
    'tabm-lora',
    'tabm-nora-init',
    'tabm-nora',
    'tabm-mini',
    'mlp',
]


def load_report(path: Path) -> dict:
    p = path / 'report.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def get_eval_dir(arch: str, dataset: str) -> Path:
    return EXP_ROOT / arch / dataset / '0-evaluation'


def get_ensemble_dir(arch: str, dataset: str, ensemble_size: int = 5) -> Path:
    return EXP_ROOT / arch / dataset / f'0-ensemble-{ensemble_size}'


def collect_seeds(eval_dir: Path) -> list[tuple[int, dict]]:
    if not eval_dir.exists():
        return []
    dirs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    return [(int(d.name), load_report(d)) for d in dirs if load_report(d)]


def score_of(r: dict, split: str) -> float | None:
    try:
        return r['metrics'][split]['score']
    except (KeyError, TypeError):
        return None


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def std(xs):
    if not xs:
        return float('nan')
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def print_arch_summary(arch: str, dataset: str) -> list[float]:
    eval_dir = get_eval_dir(arch, dataset)
    seeds = collect_seeds(eval_dir)

    if not seeds:
        print(f'  No results found at {eval_dir}')
        return []

    first_r = seeds[0][1]
    model_cfg = first_r.get('config', {}).get('model', {})
    n_params = first_r.get('n_parameters', 'N/A')

    print(f'  arch_type    : {model_cfg.get("arch_type", arch)}')
    print(f'  k            : {model_cfg.get("k", "N/A")}')
    if 'rank' in model_cfg:
        print(f'  rank         : {model_cfg["rank"]}')
    print(f'  n_parameters : {n_params}')
    print(f'  seeds found  : {len(seeds)}')
    print()

    test_scores = []
    print(f'  {"Seed":<6} {"Train":>10} {"Val":>10} {"Test":>10}')
    print(f'  {"-"*6} {"-"*10} {"-"*10} {"-"*10}')
    for seed_id, r in seeds:
        tr = score_of(r, 'train')
        va = score_of(r, 'val')
        te = score_of(r, 'test')
        if te is not None:
            test_scores.append(te)
        print(
            f'  {seed_id:<6} {tr:>10.4f} {va:>10.4f} {te:>10.4f}'
            if all(v is not None for v in [tr, va, te])
            else f'  {seed_id:<6} (incomplete)'
        )

    if len(seeds) > 1:
        vals = [score_of(r, 'val') for _, r in seeds if score_of(r, 'val') is not None]
        tests = [score_of(r, 'test') for _, r in seeds if score_of(r, 'test') is not None]
        print(f'  {"-"*6} {"-"*10} {"-"*10} {"-"*10}')
        print(f'  {"Mean":<6} {"":>10} {mean(vals):>10.4f} {mean(tests):>10.4f}')
        print(f'  {"Std":<6} {"":>10}  {std(vals):>10.4f}  {std(tests):>10.4f}')

    # Ensemble
    ens_dir = get_ensemble_dir(arch, dataset)
    if ens_dir.exists():
        ens_reports = [
            load_report(d)
            for d in sorted(ens_dir.iterdir(), key=lambda d: d.name)
            if d.is_dir() and (d / 'report.json').exists()
        ]
        if ens_reports:
            ens_tests = [score_of(r, 'test') for r in ens_reports if score_of(r, 'test') is not None]
            print()
            print(f'  Ensemble-5 test scores: {[f"{x:.4f}" for x in ens_tests]}')
            print(f'  Ensemble-5 mean test  : {mean(ens_tests):.4f}')

    return test_scores


def print_seed_detail(arch: str, dataset: str, seed: int) -> None:
    path = get_eval_dir(arch, dataset) / str(seed)
    r = load_report(path)
    if not r:
        print(f'  No report found at {path}')
        return

    cfg = r.get('config', {})
    model_cfg = cfg.get('model', {})
    opt_cfg = cfg.get('optimizer', {})
    backbone_cfg = model_cfg.get('backbone', {})

    print(f'  seed         : {cfg.get("seed")}')
    print(f'  arch_type    : {model_cfg.get("arch_type")}')
    print(f'  k            : {model_cfg.get("k")}')
    if 'rank' in model_cfg:
        print(f'  rank         : {model_cfg["rank"]}')
    print(f'  n_blocks     : {backbone_cfg.get("n_blocks")}')
    print(f'  d_block      : {backbone_cfg.get("d_block")}')
    print(f'  dropout      : {backbone_cfg.get("dropout")}')
    print(f'  optimizer    : {opt_cfg.get("type")}  lr={opt_cfg.get("lr"):.2e}  wd={opt_cfg.get("weight_decay"):.2e}')
    print(f'  n_parameters : {r.get("n_parameters")}')
    print(f'  best_step    : {r.get("best_step")}')
    print(f'  train time   : {r.get("time")}')
    print()
    print(f'  train score  : {score_of(r, "train"):.4f}')
    print(f'  val score    : {score_of(r, "val"):.4f}')
    print(f'  test score   : {score_of(r, "test"):.4f}')


def list_datasets(arch: str) -> list[str]:
    arch_dir = EXP_ROOT / arch
    if not arch_dir.exists():
        return []
    return sorted(d.name for d in arch_dir.iterdir() if d.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', nargs='?', help='Dataset name (e.g. california). Omit to list available datasets.')
    parser.add_argument('--arch', default='tabm', choices=ARCHS, help='Architecture to look up (default: tabm)')
    parser.add_argument('--seed', type=int, default=None, help='Show detail for a specific seed index')
    parser.add_argument('--compare', action='store_true', help='Compare tabm vs tabm-lora side by side')
    args = parser.parse_args()

    sep = '=' * 60

    if args.dataset is None:
        print('Available datasets per architecture:\n')
        for arch in ARCHS:
            datasets = list_datasets(arch)
            if datasets:
                print(f'  {arch}: {", ".join(datasets)}')
        return

    dataset = args.dataset

    if args.compare:
        archs_to_show = ['tabm', 'tabm-lora']
        print(sep)
        print(f'Comparison: tabm vs tabm-lora  |  dataset={dataset}')
        print(sep)
        summary = {}
        for arch in archs_to_show:
            print(f'\n[{arch}]')
            tests = print_arch_summary(arch, dataset)
            if tests:
                summary[arch] = {'mean': mean(tests), 'std': std(tests)}
        if len(summary) == 2:
            delta = summary['tabm-lora']['mean'] - summary['tabm']['mean']
            print(f'\n  Delta (tabm-lora - tabm) mean test: {delta:+.4f}')
        return

    if args.seed is not None:
        print(sep)
        print(f'Seed detail  |  arch={args.arch}  dataset={dataset}  seed={args.seed}')
        print(sep)
        print()
        print_seed_detail(args.arch, dataset, args.seed)
        return

    print(sep)
    print(f'Results  |  arch={args.arch}  dataset={dataset}')
    print(sep)
    print()
    print_arch_summary(args.arch, dataset)


if __name__ == '__main__':
    main()
