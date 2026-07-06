"""
Compare two benchmark results (micro or e2e) side by side.

Reads two JSON files produced by bench/micro.py or bench/e2e.py and prints
a formatted diff table. Positive throughput delta = improvement; negative
time delta = improvement.

Usage (from core/):
    python -m evaluation.bench.compare baseline.json optimized.json
    python -m evaluation.bench.compare bench_results/micro_A.json bench_results/micro_B.json
    python -m evaluation.bench.compare bench_results/e2e_A.json bench_results/e2e_B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with p.open() as f:
        return json.load(f)


def _delta(base: float | None, opt: float | None) -> tuple[str, str, str]:
    """Returns (base_str, opt_str, rel_change_str)."""
    if base is None or opt is None:
        return _fmt(base), _fmt(opt), "  n/a"
    d = opt - base
    rel = (d / abs(base) * 100) if base != 0 else 0.0
    sign = "+" if rel >= 0 else ""
    color = _green if rel >= 0 else _red
    return _fmt(base), _fmt(opt), color(f"{sign}{rel:.1f}%")


def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return " " * 10 + "n/a"
    if abs(v) >= 1e6:
        return f"{v:>13,.0f}"
    if abs(v) >= 1e3:
        return f"{v:>13,.1f}"
    return f"{v:>13.{decimals}f}"


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


def _header(label: str, w: int = 72) -> None:
    print(f"\n{'─'*w}")
    print(f"  {label}")
    print(f"{'─'*w}")


def _row(label: str, base_s: str, opt_s: str, delta_s: str, *, lower_is_better: bool = True) -> None:
    print(f"  {label:<30} {base_s}  {opt_s}  {delta_s}")


def _section(title: str) -> None:
    print(f"\n  [{title}]")


# ── Micro comparison ──────────────────────────────────────────────────────────

def _micro_op(base: dict, opt: dict, op_name: str) -> None:
    if op_name not in base and op_name not in opt:
        return
    _section(op_name.upper())
    col = "  {:<30} {:>13}  {:>13}  {:>8}"
    print(col.format("Metric", "Baseline", "Optimized", "ΔRel"))

    def row_time(label: str, stat_key: str) -> None:
        bv = base.get(op_name, {}).get("gpu_stats", {}).get(stat_key)
        ov = opt.get(op_name, {}).get("gpu_stats", {}).get(stat_key)
        bms = bv * 1e3 if bv is not None else None
        oms = ov * 1e3 if ov is not None else None
        bs, os_, ds = _delta(bms, oms)
        # For time: lower is better → negate sign
        if bms is not None and oms is not None:
            d = oms - bms
            rel = (d / abs(bms) * 100) if bms != 0 else 0.0
            sign = "+" if rel >= 0 else ""
            ds = _red(f"{sign}{rel:.1f}%") if rel > 0 else _green(f"{sign}{rel:.1f}%")
        print(f"  {label:<30} {bs}  {os_}  {ds}")

    def row_throughput(label: str) -> None:
        import numpy as np
        btp = base.get(op_name, {}).get("gpu_throughput_per_sec", [])
        otp = opt.get(op_name, {}).get("gpu_throughput_per_sec", [])
        bv = float(np.median(btp)) if btp else None
        ov = float(np.median(otp)) if otp else None
        bs, os_, _ = _delta(bv, ov)
        if bv is not None and ov is not None:
            d = ov - bv
            rel = (d / abs(bv) * 100) if bv != 0 else 0.0
            sign = "+" if rel >= 0 else ""
            ds = _green(f"{sign}{rel:.1f}%") if rel >= 0 else _red(f"{sign}{rel:.1f}%")
        else:
            ds = "  n/a"
        print(f"  {label:<30} {bs}  {os_}  {ds}")

    row_time(f"  gpu_time median (ms)", "median")
    row_time(f"  gpu_time p5     (ms)", "p5")
    row_time(f"  gpu_time p95    (ms)", "p95")
    row_throughput(f"  throughput median (pts/s)")

    # n_ops / batch_size
    bn = base.get(op_name, {}).get("n_ops") or base.get(op_name, {}).get("batch_size")
    on = opt.get(op_name, {}).get("n_ops") or opt.get(op_name, {}).get("batch_size")
    if bn or on:
        print(f"  {'  n_ops / batch_size':<30} {str(bn or 'n/a'):>13}  {str(on or 'n/a'):>13}")


def compare_micro(base: dict, opt: dict, base_path: str, opt_path: str) -> None:
    _header(
        f"Microbenchmark Comparison\n"
        f"  Baseline : {base_path}\n"
        f"  Optimized: {opt_path}\n"
        f"  Dataset  : {base.get('dataset')}  n={base.get('n_instances')}  "
        f"git: {base.get('git_hash')} → {opt.get('git_hash')}"
    )
    for op in ("insert", "update", "embed"):
        _micro_op(base, opt, op)
    print()


# ── E2E comparison ────────────────────────────────────────────────────────────

def compare_e2e(base: dict, opt: dict, base_path: str, opt_path: str) -> None:
    _header(
        f"E2E Benchmark Comparison\n"
        f"  Baseline : {base_path}\n"
        f"  Optimized: {opt_path}\n"
        f"  Dataset  : {base.get('dataset')}  n={base.get('n_instances')}  "
        f"git: {base.get('git_hash')} → {opt.get('git_hash')}"
    )

    col = "  {:<30} {:>13}  {:>13}  {:>8}"
    print(col.format("Metric", "Baseline", "Optimized", "ΔRel"))

    def row_time(label: str, bv: float | None, ov: float | None) -> None:
        bs, os_, _ = _delta(bv, ov)
        ds = "  n/a"
        if bv is not None and ov is not None:
            d = ov - bv
            rel = (d / abs(bv) * 100) if bv != 0 else 0.0
            sign = "+" if rel >= 0 else ""
            ds = _red(f"{sign}{rel:.1f}%") if rel > 0 else _green(f"{sign}{rel:.1f}%")
        print(f"  {label:<30} {bs}  {os_}  {ds}")

    def row_tp(label: str, bv: float | None, ov: float | None) -> None:
        bs, os_, _ = _delta(bv, ov)
        ds = "  n/a"
        if bv is not None and ov is not None:
            d = ov - bv
            rel = (d / abs(bv) * 100) if bv != 0 else 0.0
            sign = "+" if rel >= 0 else ""
            ds = _green(f"{sign}{rel:.1f}%") if rel >= 0 else _red(f"{sign}{rel:.1f}%")
        print(f"  {label:<30} {bs}  {os_}  {ds}")

    def row_qual(label: str, bv: float | None, ov: float | None, higher_is_better: bool = True) -> None:
        bs, os_, _ = _delta(bv, ov)
        ds = "  n/a"
        if bv is not None and ov is not None:
            d = ov - bv
            rel = (d / abs(bv) * 100) if bv != 0 else 0.0
            sign = "+" if rel >= 0 else ""
            better = (rel >= 0) == higher_is_better
            ds = _green(f"{sign}{rel:.1f}%") if better else _red(f"{sign}{rel:.1f}%")
        print(f"  {label:<30} {bs}  {os_}  {ds}")

    _section("OVERALL")
    row_time("total wall time (s)",
             base.get("total_wall_time_sec"), opt.get("total_wall_time_sec"))
    row_time("model cumul wall time (s)",
             base.get("cumulative_model_wall_time_sec"),
             opt.get("cumulative_model_wall_time_sec"))
    # n_runs (new format) or total_runs (old format)
    bn_runs = base.get("n_runs") or base.get("total_runs", "n/a")
    on_runs = opt.get("n_runs") or opt.get("total_runs", "n/a")
    print(f"  {'  total runs':<30} {str(bn_runs):>13}  {str(on_runs):>13}")
    print(f"  {'  stop reason':<30} {str(base.get('stop_reason', 'n/a')):>13}  "
          f"{str(opt.get('stop_reason', 'n/a')):>13}")

    _section("PHASE TIMES (cumulative seconds)")
    for ph in ("insertion", "update", "embedding"):
        # new format: summary.phase_times_sec; old format: phase_times_sec
        bv = (base.get("summary") or base).get("phase_times_sec", {}).get(ph)
        ov = (opt.get("summary") or opt).get("phase_times_sec", {}).get(ph)
        row_time(f"  {ph}", bv, ov)

    _section("PHASE FRACTIONS")
    for ph in ("insertion", "update", "embedding"):
        bv = (base.get("summary") or base).get("phase_fractions", {}).get(ph)
        ov = (opt.get("summary") or opt).get("phase_fractions", {}).get(ph)
        bvp = bv * 100 if bv is not None else None
        ovp = ov * 100 if ov is not None else None
        bs, os_, _ = _delta(bvp, ovp)
        print(f"  {'  ' + ph:<30} {bs}  {os_}")

    _section("THROUGHPUT (pts/s, higher is better)")
    for ph in ("insertion", "update", "embedding"):
        bv = (base.get("summary") or base).get("throughput_per_sec", {}).get(ph)
        ov = (opt.get("summary") or opt).get("throughput_per_sec", {}).get(ph)
        row_tp(f"  {ph}", bv, ov)

    _section("FINAL QUALITY")
    # new format: final_quality; old format: quality
    bq = base.get("final_quality") or base.get("quality") or {}
    oq = opt.get("final_quality") or opt.get("quality") or {}
    row_qual("  NNG recall", bq.get("nng_recall"), oq.get("nng_recall"), higher_is_better=True)
    if bq.get("fidelity") is not None or oq.get("fidelity") is not None:
        row_qual("  Fidelity", bq.get("fidelity"), oq.get("fidelity"), higher_is_better=True)
        if bq.get("fidelity_baseline_mean") or oq.get("fidelity_baseline_mean"):
            print(f"  {'  vs UMAP baseline (Δ)':<30} "
                  f"{_fmt(bq.get('fidelity_vs_baseline'))}  "
                  f"{_fmt(oq.get('fidelity_vs_baseline'))}")
    print()


# ── Mixed comparison ──────────────────────────────────────────────────────────

def compare_mixed(base: dict, opt: dict, base_path: str, opt_path: str) -> None:
    """Compare files of different types if they share metrics."""
    print(f"\nWARNING: comparing '{base.get('benchmark_type')}' vs '{opt.get('benchmark_type')}' "
          f"— only common metrics shown.", file=sys.stderr)
    # Fall back to best effort micro comparison if one is micro
    bt, ot = base.get("benchmark_type"), opt.get("benchmark_type")
    if bt == "micro" or ot == "micro":
        compare_micro(base, opt, base_path, opt_path)
    else:
        compare_e2e(base, opt, base_path, opt_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", help="Path to baseline JSON")
    p.add_argument("optimized", help="Path to optimized JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = _load(args.baseline)
    opt = _load(args.optimized)

    bt = base.get("benchmark_type")
    ot = opt.get("benchmark_type")

    if bt == ot == "micro":
        compare_micro(base, opt, args.baseline, args.optimized)
    elif bt == ot == "e2e":
        compare_e2e(base, opt, args.baseline, args.optimized)
    else:
        compare_mixed(base, opt, args.baseline, args.optimized)


if __name__ == "__main__":
    main()
