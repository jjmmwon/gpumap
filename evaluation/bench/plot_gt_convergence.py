"""
plot_gt_convergence.py — GT (ground-truth neighbor union) k/h convergence chart.

prepare_baselines.py's `ground_truth` phase repeatedly runs UMAP and accumulates
the union of each run's k-NN, tracking h/k = the mean fraction of a run's k
neighbors that are *new* to the union. It stops once the simple moving average
(SMA) of h/k drops to/below a threshold, and stores the full h/k history as the
`convergence_{k}` array inside `ground_truth_neighbors.npz`.

This script loads that array (no GPU / re-run needed) and renders per-run h/k
alongside its SMA, the convergence threshold, and the run at which convergence
was reached — so you can sanity-check --gt-max-runs / --gt-ma-window /
--gt-ma-threshold choices without re-running ground-truth generation.

Usage (from core/):
    python -m evaluation.bench.plot_gt_convergence --dataset svhn
    python -m evaluation.bench.plot_gt_convergence --dataset svhn --k 15 --ma-window 5 --ma-threshold 0.2
    python -m evaluation.bench.plot_gt_convergence --datasets svhn laion2M sift1m   # overlay comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.runner import DEFAULT_DATA_ROOT  # noqa: E402

# Fixed categorical order (validated palette) — never cycled/reassigned.
DATASET_COLORS = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
GOOD = "#0ca30c"  # status color: reserved for "converged", never a series identity


def _baselines_dir(dataset: str, data_root: Path) -> Path:
    return data_root / dataset / "baselines"


def _load_params(baselines_dir: Path) -> dict[str, Any]:
    p = baselines_dir / "params.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def load_hk_history(dataset: str, k: int, data_root: Path) -> np.ndarray:
    gt_path = _baselines_dir(dataset, data_root) / "ground_truth_neighbors.npz"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{gt_path} not found. Run:\n"
            f"  python -m evaluation.bench.prepare_baselines --dataset {dataset} --phases ground_truth"
        )
    key = f"convergence_{k}"
    with np.load(gt_path) as f:
        if key not in f.files:
            raise KeyError(f"{key!r} not in {gt_path} — available: {list(f.files)}")
        return f[key].astype(np.float64)


def compute_sma(hk_history: np.ndarray, ma_window: int) -> np.ndarray:
    n = len(hk_history)
    sma = np.full(n, np.nan)
    for i in range(ma_window - 1, n):
        sma[i] = hk_history[i - ma_window + 1 : i + 1].mean()
    return sma


def find_convergence(sma: np.ndarray, ma_threshold: float) -> tuple[int | None, float | None]:
    for i, s in enumerate(sma):
        if not np.isnan(s) and s <= ma_threshold:
            return i + 1, float(s)  # 1-indexed run
    return None, None


def print_stats(dataset: str, hk: np.ndarray, sma: np.ndarray, ma_window: int,
                 ma_threshold: float, conv_run: int | None, conv_sma: float | None) -> None:
    print(f"\n{'─'*60}")
    print(f"  GT convergence: {dataset}  (n_runs={len(hk)}, ma_window={ma_window}, "
          f"threshold={ma_threshold})")
    print(f"{'─'*60}")
    print(f"  h/k       : first={hk[0]:.4f}  last={hk[-1]:.4f}  mean={hk.mean():.4f}")
    valid_sma = sma[~np.isnan(sma)]
    if len(valid_sma):
        print(f"  SMA       : first_valid={valid_sma[0]:.4f}  last={valid_sma[-1]:.4f}")
    if conv_run is not None:
        print(f"  Converged : run {conv_run}  (sma={conv_sma:.4f} <= {ma_threshold})")
    else:
        print(f"  Converged : NOT reached within {len(hk)} runs "
              f"(final sma={valid_sma[-1] if len(valid_sma) else float('nan'):.4f})")
    print(f"{'─'*60}")


def plot_single(dataset: str, k: int, hk: np.ndarray, sma: np.ndarray, ma_window: int,
                 ma_threshold: float, conv_run: int | None, out_path: Path) -> None:
    run_ids = np.arange(1, len(hk) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(run_ids, hk, color="#2a78d6", linewidth=2.5, label="Mean h/k")
    valid = ~np.isnan(sma)
    ax.plot(run_ids[valid], sma[valid], color="#1baf7a", linewidth=2.5,
            label=f"SMA (window={ma_window})")
    ax.axhline(ma_threshold, color="#898781", linestyle="--", linewidth=2,
               label=f"Threshold ({ma_threshold:.2f})")

    if conv_run is not None:
        ax.plot(conv_run, sma[conv_run - 1], marker="o", markersize=10,
                color=GOOD, zorder=5, label=f"Converged @ run {conv_run}")

    ax.set_xlabel("Run", fontsize=13)
    ax.set_ylabel("h/k", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title(f"GT union convergence — {dataset} (k={k})", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart -> {out_path}")


def plot_multi(datasets: list[str], k: int, data: dict[str, dict[str, Any]],
               ma_threshold: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for i, dataset in enumerate(datasets):
        d = data[dataset]
        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        sma, valid = d["sma"], ~np.isnan(d["sma"])
        run_ids = np.arange(1, len(sma) + 1)
        ax.plot(run_ids[valid], sma[valid], color=color, linewidth=2.5, label=dataset)
        if d["conv_run"] is not None:
            ax.plot(d["conv_run"], sma[d["conv_run"] - 1], marker="o", markersize=8,
                    color=color, markeredgecolor="#0b0b0b", markeredgewidth=1, zorder=5)

    ax.axhline(ma_threshold, color="#898781", linestyle="--", linewidth=2,
               label=f"Threshold ({ma_threshold:.2f})")
    ax.set_xlabel("Run", fontsize=13)
    ax.set_ylabel(f"SMA(h/k)", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.set_title(f"GT union convergence — dataset comparison (k={k})", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default=None, help="Single dataset (detailed chart)")
    p.add_argument("--datasets", nargs="+", default=None, help="Multiple datasets (SMA overlay comparison)")
    p.add_argument("--k", type=int, default=15, help="Ground truth key k_{k} (default: 15)")
    p.add_argument("--ma-window", type=int, default=None,
                   help="SMA window (default: read from baselines/params.json, else 5)")
    p.add_argument("--ma-threshold", type=float, default=None,
                   help="Convergence threshold (default: read from baselines/params.json, else 0.2)")
    p.add_argument("--data-root", default=None, metavar="DIR",
                   help="Root dir containing {dataset}/baselines/ (default: evaluation/data_scaled)")
    p.add_argument("--out", default=None, metavar="FILE", help="Output PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT

    if not args.dataset and not args.datasets:
        print("Pass --dataset <name> or --datasets <name...>", file=sys.stderr)
        sys.exit(1)

    datasets = args.datasets if args.datasets else [args.dataset]

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        params = _load_params(_baselines_dir(dataset, data_root))
        ma_window = args.ma_window or params.get("gt_ma_window", 5)
        ma_threshold = args.ma_threshold if args.ma_threshold is not None else params.get(
            "gt_ma_threshold", 0.2
        )
        hk = load_hk_history(dataset, args.k, data_root)
        sma = compute_sma(hk, ma_window)
        conv_run, conv_sma = find_convergence(sma, ma_threshold)
        print_stats(dataset, hk, sma, ma_window, ma_threshold, conv_run, conv_sma)
        per_dataset[dataset] = {
            "hk": hk, "sma": sma, "ma_window": ma_window,
            "ma_threshold": ma_threshold, "conv_run": conv_run, "conv_sma": conv_sma,
        }

    if args.datasets and len(args.datasets) > 1:
        out_path = Path(args.out) if args.out else Path(f"gt_convergence_compare_k{args.k}.png")
        # Use the threshold from the first dataset for the shared reference line.
        shared_threshold = per_dataset[datasets[0]]["ma_threshold"]
        plot_multi(datasets, args.k, per_dataset, shared_threshold, out_path)
    else:
        dataset = datasets[0]
        d = per_dataset[dataset]
        out_path = Path(args.out) if args.out else Path(f"gt_convergence_{dataset}_k{args.k}.png")
        plot_single(dataset, args.k, d["hk"], d["sma"], d["ma_window"],
                    d["ma_threshold"], d["conv_run"], out_path)


if __name__ == "__main__":
    main()
