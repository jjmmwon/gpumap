"""
eval_runs_quality.py — post-hoc fidelity / NNG recall trend for a saved e2e run.

e2e.py only computes fidelity/nng_recall once, at the very end of the run
(`final_quality`), even though every run already has its embedding and NNG
saved to `runs/run_{id:06d}.npz`. This script goes back over an existing e2e
output directory and fills in per-run `quality` (fidelity, nng_recall) so the
trend across runs can be plotted.

Evaluating every run would mean recomputing a full KNN graph that many
times, so runs are sampled per progress stage (insertion / update /
embedding-only, based on each run's insertion_completed/update_completed/
embedding_completed flags — see EvalScheduler in quality.py). The run where a
stage's flag first flips True is always evaluated, and so is the final run,
regardless of the configured step/max.

Each stage can be sampled either every Nth run (--step-*) or capped to at
most N runs evenly spaced across the stage (--max-measurements, which
overrides --step-* for every stage).

Usage (from core/):
    python -m evaluation.bench.eval_runs_quality --run-dir evaluation/progressive_results/<dir>
    python -m evaluation.bench.eval_runs_quality --run-dir <dir> --step-insertion 10 --step-update 5
    python -m evaluation.bench.eval_runs_quality --run-dir <dir> --max-measurements 20
    python -m evaluation.bench.eval_runs_quality --run-dir <dir> --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.bench.quality import backfill_run_quality, load_baselines  # noqa: E402

LOG = logging.getLogger("bench.eval_runs_quality")


def eval_runs_quality(
    run_dir: Path,
    step_insertion: int,
    step_update: int,
    step_embedding: int,
    max_measurements: int | None,
    overwrite: bool,
) -> Path:
    results_path = run_dir / "benchmark_results.json"
    runs_dir = run_dir / "runs"
    with results_path.open() as f:
        data = json.load(f)

    records = data["runs"]
    dataset = data["dataset"]

    baselines = load_baselines(dataset)
    if "original_nn_infos_path" not in baselines and "ground_truth_path" not in baselines:
        LOG.warning(
            "No baselines found for dataset=%s under evaluation/data_scaled/%s/baselines — "
            "nothing to compute. Run: python -m evaluation.bench.prepare_baselines --dataset %s",
            dataset, dataset, dataset,
        )
        return results_path

    LOG.info(
        "Backfilling quality (step_insertion=%d, step_update=%d, step_embedding=%d) over %d run(s)",
        step_insertion, step_update, step_embedding, len(records),
    )
    n_done = backfill_run_quality(
        records, runs_dir, baselines,
        step_insertion, step_update, step_embedding,
        max_measurements,
        overwrite=overwrite, log_progress=True,
    )

    with results_path.open("w") as f:
        json.dump(data, f, indent=2)
    LOG.info("Updated %d run(s) -> %s", n_done, results_path)
    return results_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-dir", required=True, metavar="DIR",
        help="e2e output dir (contains benchmark_results.json and runs/)",
    )
    p.add_argument(
        "--step-insertion", type=int, default=1, metavar="N",
        help="Evaluate quality every Nth run while insertion is still in progress "
             "(default: 1 = every run).",
    )
    p.add_argument(
        "--step-update", type=int, default=1, metavar="N",
        help="Evaluate quality every Nth run while update is still in progress, "
             "after insertion has completed (default: 1).",
    )
    p.add_argument(
        "--step-embedding", type=int, default=1, metavar="N",
        help="Evaluate quality every Nth run while embedding is still in progress, "
             "after insertion and update have completed (default: 1).",
    )
    p.add_argument(
        "--max-measurements", type=int, default=None, metavar="N",
        help="Cap quality evaluations to at most N per stage (insertion, update, "
             "embedding each capped independently), evenly spaced across the stage — "
             "overrides --step-insertion/--step-update/--step-embedding.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recompute quality even for runs that already have it.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    eval_runs_quality(
        run_dir=Path(args.run_dir),
        step_insertion=args.step_insertion,
        step_update=args.step_update,
        step_embedding=args.step_embedding,
        max_measurements=args.max_measurements,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
