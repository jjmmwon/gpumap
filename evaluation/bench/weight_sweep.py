"""
weight_sweep.py — Sweep WeightedStrategy (insertion, update, embedding) weights
to find a good default.

Runs bench/e2e.py once per weight config (as a subprocess, one config per
process so a single CUDA crash only loses that config, not the whole sweep),
then writes a single summary JSON that weight_sweep_analysis.ipynb consumes to
chart normalized fidelity and final recall across configs.

Resumable: a config already present under progressive_results/ with
stop_reason == "completed" and final_quality computed is skipped unless
--force. Safe to re-launch after a crash or to extend the grid later.

Usage (from core/):
    python -m evaluation.bench.weight_sweep --dataset svhn
    python -m evaluation.bench.weight_sweep --dataset svhn \\
        --weights 1,1,1 1,4,1 1,8,1 1,12,4 1,16,1 4,12,1
    python -m evaluation.bench.weight_sweep --dataset svhn --tag-prefix sweepA --retries 2
    python -m evaluation.bench.weight_sweep --dataset svhn --force   # rerun everything

Then open evaluation/bench/weight_sweep_analysis.ipynb, set DATASET / TAG_PREFIX
to match, and run all cells.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.runner import DEFAULT_OUTPUT_ROOT, sanitize_path_component  # noqa: E402

CORE_DIR = REPO_ROOT

# A reasonable starting grid: keep insertion/embedding weight fixed relative to
# each other and vary update weight, since update dominates GPUMAP's steady-state
# workload; mirrors weight configs already explored ad hoc in progressive_results/.
DEFAULT_WEIGHTS_GRID: list[tuple[float, float, float]] = [
    (1.0, 1.0, 1.0),
    (1.0, 4.0, 1.0),
    (1.0, 8.0, 1.0),
    (1.0, 12.0, 1.0),
    (1.0, 12.0, 4.0),
    (1.0, 16.0, 1.0),
    (4.0, 12.0, 1.0),
]


def _parse_weight_arg(s: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'INS,UPD,EMB', got {s!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _weight_tag(weights: tuple[float, float, float], prefix: str) -> str:
    def fmt(v: float) -> str:
        return f"{v:g}".replace(".", "p")
    label = "_".join(fmt(v) for v in weights)
    return sanitize_path_component(f"{prefix}_{label}" if prefix else label)


def find_result_dir(dataset: str, strategy: str, tag: str) -> Path | None:
    if not DEFAULT_OUTPUT_ROOT.exists():
        return None
    matches = sorted(
        p for p in DEFAULT_OUTPUT_ROOT.glob(f"{dataset}_{strategy}_*_{tag}_*")
        if p.is_dir() and (p / "benchmark_results.json").exists()
    )
    return matches[-1] if matches else None


def is_config_done(result_dir: Path | None) -> dict[str, Any] | None:
    if result_dir is None:
        return None
    with open(result_dir / "benchmark_results.json") as f:
        b = json.load(f)
    if b.get("stop_reason") == "completed" and b.get("final_quality"):
        return b
    return None


def run_config(
    python_bin: str,
    dataset: str,
    strategy: str,
    weights: tuple[float, float, float] | None,
    target_latency: float,
    tag: str,
    retries: int,
    retry_cooldown: float,
) -> dict[str, Any] | None:
    cmd = [
        python_bin, "-m", "evaluation.bench.e2e",
        "--dataset", dataset,
        "--strategy", strategy,
        "--target-latency", str(target_latency),
        "--tag", tag,
    ]
    if weights is not None:
        cmd += ["--weights", *[str(w) for w in weights]]

    for attempt in range(1, retries + 1):
        print(f"\n>>> [{tag}] attempt {attempt}/{retries}: {' '.join(cmd)}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=CORE_DIR)
        elapsed = time.perf_counter() - t0
        result_dir = find_result_dir(dataset, strategy, tag)
        benchmark = is_config_done(result_dir)
        if proc.returncode == 0 and benchmark is not None:
            print(f"    OK ({elapsed:.1f}s) -> {result_dir}")
            return benchmark
        print(
            f"    FAILED (exit={proc.returncode}, elapsed={elapsed:.1f}s"
            f"{', dir=' + str(result_dir) if result_dir else ''})"
        )
        if attempt < retries:
            print(f"    Retrying in {retry_cooldown:.0f}s ...")
            time.sleep(retry_cooldown)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--weights", nargs="+", type=_parse_weight_arg, default=None,
        metavar="INS,UPD,EMB",
        help="Weight configs to sweep, e.g. 1,1,1 1,12,4 (default: built-in grid)",
    )
    p.add_argument("--target-latency", type=float, default=10.0, metavar="SEC")
    p.add_argument("--tag-prefix", default="sweep", help="Prefix for each config's --tag")
    p.add_argument(
        "--include-sequential-baseline", action="store_true",
        help="Also run --strategy sequential once, for a reference point",
    )
    p.add_argument("--retries", type=int, default=2, help="Retries per config on failure (default: 2)")
    p.add_argument("--retry-cooldown", type=float, default=15.0, metavar="SEC")
    p.add_argument("--force", action="store_true", help="Rerun configs even if already completed")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--summary-out", default=None, metavar="FILE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights_grid = args.weights or DEFAULT_WEIGHTS_GRID

    configs: list[dict[str, Any]] = []
    if args.include_sequential_baseline:
        configs.append({"strategy": "sequential", "weights": None})
    for w in weights_grid:
        configs.append({"strategy": "weighted", "weights": w})

    results: list[dict[str, Any]] = []
    for cfg in configs:
        strategy = cfg["strategy"]
        weights = cfg["weights"]
        tag = (
            _weight_tag(weights, args.tag_prefix) if weights is not None
            else sanitize_path_component(f"{args.tag_prefix}_sequential")
        )

        existing = None if args.force else is_config_done(find_result_dir(args.dataset, strategy, tag))
        if existing is not None:
            print(f"\n>>> [{tag}] already completed, skipping (--force to rerun)")
            benchmark = existing
        else:
            benchmark = run_config(
                args.python_bin, args.dataset, strategy, weights,
                args.target_latency, tag, args.retries, args.retry_cooldown,
            )

        if benchmark is None:
            results.append({
                "tag": tag, "strategy": strategy, "weights": list(weights) if weights else None,
                "status": "failed",
            })
            continue

        q = benchmark.get("final_quality") or {}
        results.append({
            "tag": tag,
            "strategy": strategy,
            "weights": list(weights) if weights else None,
            "status": "ok",
            "fidelity": q.get("fidelity"),
            "fidelity_baseline_mean": q.get("fidelity_baseline_mean"),
            "nng_recall": q.get("nng_recall"),
            "total_wall_time_sec": benchmark.get("total_wall_time_sec"),
            "cumulative_model_wall_time_sec": benchmark.get("cumulative_model_wall_time_sec"),
            "n_runs": benchmark.get("n_runs"),
            "phase_fractions": (benchmark.get("summary") or {}).get("phase_fractions"),
        })

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*60}\n  Sweep complete: {n_ok}/{len(results)} configs ok\n{'='*60}")

    out_path = Path(args.summary_out) if args.summary_out else (
        DEFAULT_OUTPUT_ROOT / f"weight_sweep_{args.dataset}_{args.tag_prefix}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"dataset": args.dataset, "target_latency": args.target_latency, "configs": results},
            f, indent=2,
        )
    print(f"Saved summary -> {out_path}")
    print(
        "\nNext: open evaluation/bench/weight_sweep_analysis.ipynb, set\n"
        f"  SUMMARY_PATH = Path({str(out_path)!r})\n"
        "and run all cells."
    )


if __name__ == "__main__":
    main()
