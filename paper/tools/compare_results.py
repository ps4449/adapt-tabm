"""
Compare TabM vs TabLoRA results across datasets.

Usage (from paper/ directory):
    python tools/compare_results.py
    python tools/compare_results.py --datasets california adult higgs-small
"""

import argparse
import json
from pathlib import Path


def load_scores(eval_dir: Path):
    seed_dirs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not seed_dirs:
        return None, None, None
    scores = []
    for d in seed_dirs:
        rp = d / 'report.json'
        if not rp.exists():
            return None, None, None
        r = json.loads(rp.read_text())
        scores.append(r['metrics']['test']['score'])
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    return mean, std, len(scores)


def load_ensemble(eval_dir: Path, ensemble_size: int = 5):
    ens_dir = eval_dir.parent / eval_dir.name.replace('evaluation', f'ensemble-{ensemble_size}')
    if not ens_dir.exists():
        return None
    scores = []
    for d in sorted(ens_dir.iterdir()):
        rp = d / 'report.json'
        if rp.exists():
            r = json.loads(rp.read_text())
            scores.append(r['metrics']['test']['score'])
    return sum(scores) / len(scores) if scores else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--datasets', nargs='+',
        default=['california', 'adult', 'higgs-small', 'covtype2', 'diamond'],
    )
    parser.add_argument('--tabm_dir', default='exp/tabm')
    parser.add_argument('--lora_dir', default='exp/tabm-lora')
    args = parser.parse_args()

    tabm_base = Path(args.tabm_dir)
    lora_base = Path(args.lora_dir)

    header = f"{'Dataset':<15} {'TabM Test':>10} {'Std':>7} {'Ens-5':>8}  {'LoRA Test':>10} {'Std':>7} {'Ens-5':>8}  {'Δ Test':>8} {'Δ Ens':>8}"
    sep = '-' * len(header)
    print(sep)
    print('TabM vs TabLoRA — Multi-Dataset Comparison')
    print(sep)
    print(header)
    print(sep)

    for ds in args.datasets:
        tabm_eval = tabm_base / ds / '0-evaluation'
        lora_eval = lora_base / ds / '0-evaluation'

        tm, ts, tn = load_scores(tabm_eval) if tabm_eval.exists() else (None, None, None)
        te = load_ensemble(tabm_eval) if tabm_eval.exists() else None

        lm, ls, ln = load_scores(lora_eval) if lora_eval.exists() else (None, None, None)
        le = load_ensemble(lora_eval) if lora_eval.exists() else None

        def fmt(v, fmt='.4f'):
            return f'{v:{fmt}}' if v is not None else '  N/A  '

        delta_m = f'{lm - tm:+.4f}' if lm is not None and tm is not None else '   N/A'
        delta_e = f'{le - te:+.4f}' if le is not None and te is not None else '   N/A'

        print(
            f'{ds:<15} {fmt(tm):>10} {fmt(ts):>7} {fmt(te):>8}  '
            f'{fmt(lm):>10} {fmt(ls):>7} {fmt(le):>8}  '
            f'{delta_m:>8} {delta_e:>8}'
        )

    print(sep)

    # Save to logs/results/
    from datetime import datetime
    log_dir = Path('logs/results')
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'comparison_{timestamp}.txt'
    lines = [sep, 'TabM vs TabLoRA — Multi-Dataset Comparison', sep, header, sep]
    for ds in args.datasets:
        tabm_eval = tabm_base / ds / '0-evaluation'
        lora_eval = lora_base / ds / '0-evaluation'
        tm, ts, _ = load_scores(tabm_eval) if tabm_eval.exists() else (None, None, None)
        te = load_ensemble(tabm_eval) if tabm_eval.exists() else None
        lm, ls, _ = load_scores(lora_eval) if lora_eval.exists() else (None, None, None)
        le = load_ensemble(lora_eval) if lora_eval.exists() else None
        def fmt(v): return f'{v:.4f}' if v is not None else '  N/A  '
        delta_m = f'{lm - tm:+.4f}' if lm is not None and tm is not None else '   N/A'
        delta_e = f'{le - te:+.4f}' if le is not None and te is not None else '   N/A'
        lines.append(
            f'{ds:<15} {fmt(tm):>10} {fmt(ts):>7} {fmt(te):>8}  '
            f'{fmt(lm):>10} {fmt(ls):>7} {fmt(le):>8}  '
            f'{delta_m:>8} {delta_e:>8}'
        )
    lines.append(sep)
    log_file.write_text('\n'.join(lines))
    print(f'\nSaved to: {log_file}')


if __name__ == '__main__':
    main()
