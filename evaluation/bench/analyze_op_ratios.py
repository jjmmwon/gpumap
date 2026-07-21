"""
analyze_op_ratios.py — Per-operation time/count ratio statistics and chart.

Aggregates one or more benchmark_results.json files (typically every run of a
given --strategy, e.g. "sequential") and reports, per operation
(insertion / update / embedding):
  - total time and its share of total component time
  - total op count and its share of total op count
  - throughput (ops/sec)

Also renders a per-run stacked-area chart of time spent per operation across
the run sequence. This is most informative for the "sequential" strategy,
which hands the entire time budget to one phase at a time (insertion, then
update, then embedding) rather than interleaving them — the chart makes that
cascade visible.

Handles both the current nested run schema (operations.{op}.run_time, under
top-level "summary"/"runs") and the legacy flat schema (top-level
"{op}_time" / "actual_ops" per run, no "summary" key), matching the
base.get("summary") or base fallback already used in bench/compare.py.

Usage (from core/):
    python -m evaluation.bench.analyze_op_ratios --dataset svhn --strategy sequential
    python -m evaluation.bench.analyze_op_ratios --result-dirs DIR1 DIR2
    python -m evaluation.bench.analyze_op_ratios --dataset svhn --strategy sequential \\
        --out op_ratios_svhn_sequential.png --json-out op_ratios_svhn_sequential.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.runner import DEFAULT_OUTPUT_ROOT  # noqa: E402

OPS = ("insertion", "update", "embedding")
# Fixed categorical order (validated palette, slots 1/2/3) — never cycled/reassigned.
OP_COLOR = {"insertion": "#2a78d6", "update": "#1baf7a", "embedding": "#eda100"}


def _find_benchmark_json(path: Path) -> Path | None:
    if path.is_file():
        return path
    candidate = path / "benchmark_results.json"
    return candidate if candidate.exists() else None


def discover_result_dirs(
    results_root: Path, dataset: str | None, strategy: str | None
) -> list[Path]:
    if not results_root.exists():
        return []
    dirs = []
    for p in sorted(results_root.iterdir()):
        if not p.is_dir():
            continue
        if dataset and not p.name.startswith(f"{dataset}_"):
            continue
        if strategy and f"_{strategy}_" not in p.name:
            continue
        if _find_benchmark_json(p) is not None:
            dirs.append(p)
    return dirs


def _normalize_run(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt one run record (either schema) into a flat {op: {ops, time}} shape."""
    ops: dict[str, int] = {}
    times: dict[str, float] = {}

    if "operations" in record:
        for op in OPS:
            entry = record["operations"].get(op, {})
            ops[op] = int(entry.get("actual_ops", 0))
            times[op] = float(entry.get("run_time", 0.0))
    else:
        actual_ops = record.get("actual_ops", {})
        for op in OPS:
            ops[op] = int(actual_ops.get(op, 0))
            times[op] = float(record.get(f"{op}_time", 0.0))

    return {
        "run_id": int(record["run_id"]),
        "ops": ops,
        "times": times,
        "wall_time": float(record.get("wall_time", 0.0)),
        "cumulative_wall_time": float(record.get("cumulative_wall_time", 0.0)),
    }


def load_runs(result_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bpath = _find_benchmark_json(result_dir)
    assert bpath is not None
    with bpath.open() as f:
        benchmark = json.load(f)
    rows = [_normalize_run(r) for r in benchmark.get("runs", [])]
    rows.sort(key=lambda r: r["run_id"])
    return benchmark, rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_time = {op: sum(r["times"][op] for r in rows) for op in OPS}
    total_ops = {op: sum(r["ops"][op] for r in rows) for op in OPS}
    grand_time = sum(total_time.values())
    grand_ops = sum(total_ops.values())

    def safe_div(num: float, den: float) -> float:
        return float("nan") if den == 0 else num / den

    return {
        "time_share_pct": {op: safe_div(total_time[op] * 100.0, grand_time) for op in OPS},
        "ops_share_pct": {op: safe_div(total_ops[op] * 100.0, grand_ops) for op in OPS},
        "throughput_ops_per_sec": {
            op: safe_div(total_ops[op], total_time[op]) for op in OPS
        },
        "total_time_sec": total_time,
        "total_ops": total_ops,
        "n_runs": len(rows),
        "total_wall_time_sec": rows[-1]["cumulative_wall_time"] if rows else float("nan"),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    filtered = [v for v in values if not math.isnan(v)]
    if not filtered:
        return float("nan"), float("nan")
    if len(filtered) == 1:
        return filtered[0], 0.0
    arr = np.array(filtered)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"n_dirs": len(summaries), "per_op": {}}
    for op in OPS:
        ts_mean, ts_std = mean_std([s["time_share_pct"][op] for s in summaries])
        os_mean, os_std = mean_std([s["ops_share_pct"][op] for s in summaries])
        tp_mean, tp_std = mean_std([s["throughput_ops_per_sec"][op] for s in summaries])
        agg["per_op"][op] = {
            "time_share_pct_mean": ts_mean,
            "time_share_pct_std": ts_std,
            "ops_share_pct_mean": os_mean,
            "ops_share_pct_std": os_std,
            "throughput_mean": tp_mean,
            "throughput_std": tp_std,
        }
    return agg


def print_stats(dataset: str | None, strategy: str, agg: dict[str, Any]) -> None:
    print()
    print(f"{'─'*68}")
    print(f"  Op ratio stats: dataset={dataset or '(all)'}  strategy={strategy}  "
          f"n_dirs={agg['n_dirs']}")
    print(f"{'─'*68}")
    print(f"  {'Operation':<12} {'Time %':>10} {'Ops %':>10} {'Throughput (ops/s)':>22}")
    for op in OPS:
        po = agg["per_op"][op]
        ts = f"{po['time_share_pct_mean']:.1f}±{po['time_share_pct_std']:.1f}"
        os_ = f"{po['ops_share_pct_mean']:.1f}±{po['ops_share_pct_std']:.1f}"
        tp = f"{po['throughput_mean']:,.0f}±{po['throughput_std']:,.0f}"
        print(f"  {op:<12} {ts:>10} {os_:>10} {tp:>22}")
    print(f"{'─'*68}")


def plot(
    dataset: str | None,
    strategy: str,
    cascade_rows: list[dict[str, Any]],
    agg: dict[str, Any],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    # ── Top: per-run time-share cascade (stacked area, raw seconds) ──────────
    ax_top = fig.add_subplot(gs[0, :])
    run_ids = [r["run_id"] for r in cascade_rows]
    stacks = [[r["times"][op] for r in cascade_rows] for op in OPS]
    ax_top.stackplot(
        run_ids, *stacks,
        labels=[op.title() for op in OPS],
        colors=[OP_COLOR[op] for op in OPS],
        alpha=0.9,
    )
    ax_top.set_xlabel("Run", fontsize=12)
    ax_top.set_ylabel("Time per run (s)", fontsize=12)
    title_ds = dataset or "all datasets"
    ax_top.set_title(
        f"Per-run operation time — {title_ds} / {strategy}"
        + (f"  (dir: {cascade_rows[0].get('_dir', '')})" if cascade_rows else ""),
        fontsize=13,
    )
    ax_top.grid(True, linestyle="--", alpha=0.35)
    ax_top.legend(loc="upper right", fontsize=10)

    # ── Bottom-left: aggregate time share % ───────────────────────────────────
    def _bar(ax, key_mean: str, key_std: str, ylabel: str, fmt: str, log: bool = False) -> None:
        vals = [agg["per_op"][op][key_mean] for op in OPS]
        errs = [agg["per_op"][op][key_std] for op in OPS]
        bars = ax.bar(
            [op.title() for op in OPS], vals, yerr=errs, capsize=4,
            color=[OP_COLOR[op] for op in OPS],
        )
        for b, v, e in zip(bars, vals, errs):
            top = v + e
            label_y = top * 1.08 if log else top + max(vals) * 0.03
            ax.text(
                b.get_x() + b.get_width() / 2, label_y,
                fmt.format(v), ha="center", va="bottom", fontsize=10,
                color="#0b0b0b",
            )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if log:
            ax.set_ylim(top=max(v + e for v, e in zip(vals, errs)) * 2.0)

    ax_ts = fig.add_subplot(gs[1, 0])
    _bar(ax_ts, "time_share_pct_mean", "time_share_pct_std", "Time share (%)", "{:.1f}%")
    ax_ts.set_title("Time share", fontsize=12)

    ax_os = fig.add_subplot(gs[1, 1])
    _bar(ax_os, "ops_share_pct_mean", "ops_share_pct_std", "Op-count share (%)", "{:.1f}%")
    ax_os.set_title("Op-count share", fontsize=12)

    ax_tp = fig.add_subplot(gs[1, 2])
    ax_tp.set_yscale("log")
    _bar(ax_tp, "throughput_mean", "throughput_std", "Throughput (ops/s)", "{:,.0f}", log=True)
    ax_tp.set_title("Throughput", fontsize=12)

    fig.suptitle(
        f"GPUMAP operation ratios — {title_ds} / {strategy} strategy "
        f"(n_dirs={agg['n_dirs']})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default=None, help="Dataset name filter (e.g. svhn)")
    p.add_argument("--strategy", default="sequential", help="Strategy filter (default: sequential)")
    p.add_argument(
        "--results-root", default=None, metavar="DIR",
        help="Root dir to search for result dirs (default: evaluation/progressive_results)",
    )
    p.add_argument(
        "--result-dirs", nargs="+", default=None, metavar="DIR",
        help="Explicit result dirs (or direct benchmark_results.json paths); overrides discovery",
    )
    p.add_argument("--out", default=None, metavar="FILE", help="Output PNG path")
    p.add_argument("--json-out", default=None, metavar="FILE", help="Output stats JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root) if args.results_root else DEFAULT_OUTPUT_ROOT

    if args.result_dirs:
        dirs = []
        for d in args.result_dirs:
            p = Path(d)
            if not p.exists():
                p = results_root / d
            if _find_benchmark_json(p) is None:
                print(f"WARNING: no benchmark_results.json found for {d}", file=sys.stderr)
                continue
            dirs.append(p)
    else:
        dirs = discover_result_dirs(results_root, args.dataset, args.strategy)

    if not dirs:
        print(
            f"No result dirs found (dataset={args.dataset!r}, strategy={args.strategy!r}, "
            f"root={results_root}). Pass --result-dirs explicitly, or run e.g.:\n"
            f"  python -m evaluation.bench.e2e --dataset {args.dataset or '<dataset>'} "
            f"--strategy {args.strategy} --tag opratios",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Using {len(dirs)} result dir(s):")
    for d in dirs:
        print(f"  {d}")

    all_rows: list[list[dict[str, Any]]] = []
    summaries = []
    for d in dirs:
        _benchmark, rows = load_runs(d)
        if not rows:
            print(f"WARNING: no runs in {d}", file=sys.stderr)
            continue
        for r in rows:
            r["_dir"] = d.name
        all_rows.append(rows)
        summaries.append(summarize_rows(rows))

    if not summaries:
        print("No usable runs found in any result dir.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(summaries)
    print_stats(args.dataset, args.strategy, agg)

    dataset_tag = args.dataset or "all"
    out_path = Path(args.out) if args.out else Path(f"op_ratios_{dataset_tag}_{args.strategy}.png")
    plot(args.dataset, args.strategy, all_rows[0], agg, out_path)

    if args.json_out:
        out = {
            "dataset": args.dataset,
            "strategy": args.strategy,
            "result_dirs": [str(d) for d in dirs],
            "per_dir_summary": summaries,
            "aggregate": agg,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved stats → {args.json_out}")


if __name__ == "__main__":
    main()
