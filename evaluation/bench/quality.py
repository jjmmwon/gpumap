"""
quality.py — shared fidelity / NNG recall computation for e2e benchmark runs.

Used by both:
  - e2e.py (live, in-process quality check against arrays already in memory)
  - eval_runs_quality.py (post-hoc, loading arrays back from runs/run_*.npz)

Also provides the stage-aware run sampling used to decide *which* runs get
evaluated, since evaluating every run in a benchmark with thousands of runs
would mean recomputing a full KNN graph that many times.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.runner import DEFAULT_DATA_ROOT  # noqa: E402
from evaluation.utils import compute_knn_indices_and_distances  # noqa: E402
from evaluation.evaluate_emb_fidelity import eval_emb_fidelity  # noqa: E402

LOG = logging.getLogger("bench.quality")

BASELINES_DIR_NAME = "baselines"


def load_baselines(dataset: str) -> dict[str, Any]:
    baselines_dir = DEFAULT_DATA_ROOT / dataset / BASELINES_DIR_NAME
    result: dict[str, Any] = {}

    orig_knn_path = baselines_dir / "original_nn_infos.npz"
    if orig_knn_path.exists():
        result["original_nn_infos_path"] = str(orig_knn_path)

    gt_path = baselines_dir / "ground_truth_neighbors.npz"
    if gt_path.exists():
        result["ground_truth_path"] = str(gt_path)

    fidelity_path = baselines_dir / "umap_fidelity_baseline.json"
    if fidelity_path.exists():
        import json

        with open(fidelity_path) as f:
            result["umap_fidelity_baseline"] = json.load(f)

    return result


def compute_quality(
    nn_indices: np.ndarray,
    embedding: np.ndarray,
    baselines: dict[str, Any],
    n_neighbors: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute NNG recall and embedding fidelity for one run's saved arrays.

    `n_total = embedding.shape[0]` is read from the arrays themselves rather
    than passed in, since mid-run arrays only cover the points inserted so far
    (they grow by appending, not by preallocating to the final dataset size).

    NNG recall compares the *current* nn_indices (built from whatever's been
    inserted so far, possibly approximate/partial) against the precomputed
    exact KNN of the *complete* dataset — i.e. how close the graph is, right
    now, to where it'll end up — matching `compute_knn_overlap` from the prior
    gpumap-eval evaluation code, not a fresh exact KNN of the current subset.
    """
    quality: dict[str, Any] = {}
    n_total = embedding.shape[0]

    if "original_nn_infos_path" in baselines:
        nng_sub = nn_indices.astype(np.int32)
        with np.load(baselines["original_nn_infos_path"]) as f:
            exact_sub = f["indices"].astype(np.int32)[:n_total]

        k = min(nng_sub.shape[1], exact_sub.shape[1])
        nng_sub = nng_sub[:, :k]
        exact_sub = exact_sub[:, :k]

        valid = (nng_sub >= 0) & (nng_sub < n_total)
        nng_masked = np.where(valid, nng_sub, -1)
        hit = (nng_masked[:, :, None] == exact_sub[:, None, :]).any(axis=2)
        recall = float(hit[valid[:, : hit.shape[1]]].mean()) if valid.any() else 0.0

        quality["nng_recall"] = recall
        quality["nng_recall_n_check"] = n_total
        quality["nng_recall_k"] = k
        if verbose:
            LOG.info("NNG recall=%.4f  (n=%d, k=%d)", recall, n_total, k)

    if "ground_truth_path" in baselines:
        ground_truth = None
        with np.load(baselines["ground_truth_path"]) as f:
            key = f"k_{n_neighbors}"
            if key not in f.files:
                available = list(f.files)
                LOG.warning(
                    "Key %s not in ground_truth_neighbors.npz — available: %s",
                    key, available,
                )
                key = available[0] if available else None
            if key is not None:
                ground_truth = f[key].astype(np.int32)

        if ground_truth is not None:
            # Runs grow by appending rows in fixed insertion order, so the first
            # n_total ground-truth rows line up with the currently-inserted points.
            ground_truth = ground_truth[:n_total]
            emb_idx, _ = compute_knn_indices_and_distances(embedding, n_neighbors)
            score, _ = eval_emb_fidelity(emb_idx[:, :n_neighbors], ground_truth)
            quality["fidelity"] = float(score)
            quality["fidelity_k"] = n_neighbors

            fb = baselines.get("umap_fidelity_baseline")
            if fb:
                quality["fidelity_baseline_mean"] = fb["mean"]
                quality["fidelity_baseline_std"] = fb["std"]
                quality["fidelity_vs_baseline"] = float(score) - fb["mean"]
                if verbose:
                    LOG.info(
                        "Fidelity=%.4f  baseline_mean=%.4f  Δ=%+.4f",
                        score, fb["mean"], float(score) - fb["mean"],
                    )
            elif verbose:
                LOG.info("Fidelity=%.4f  (no fidelity baseline)", score)

    return quality


def _select_ordinals(count: int, step: int, max_n: int | None) -> set[int]:
    """1-indexed within-stage ordinals to evaluate, out of `count` total.

    If `max_n` is given (and > 0), it overrides `step`: at most `max_n`
    ordinals are picked, evenly spaced across the *entire* stage regardless of
    how long the stage is — `np.linspace(1, count, n_pick)` always includes
    ordinal 1 (and, when n_pick > 1, ordinal `count`) as its endpoints, so the
    stage-transition run stays covered for free. Otherwise every `step`-th
    ordinal is picked, plus ordinal 1 explicitly (range(step, ...) alone
    wouldn't include it unless step == 1).
    """
    if count <= 0:
        return set()
    if max_n is not None and max_n > 0:
        n_pick = min(max_n, count)
        return {int(round(x)) for x in np.linspace(1, count, n_pick)}
    ordinals = {1}
    if step > 0:
        ordinals.update(range(step, count + 1, step))
    return ordinals


class EvalScheduler:
    """Decides which runs get a quality measurement, per progress stage.

    A run's stage is derived from its completion flags: insertion -> update ->
    embedding, in that order. The run where a stage's completion flag first
    flips True is, by definition, also the first run of the *next* stage — so
    always evaluating stage-ordinal 1 covers "always measure at a completion
    transition" without extra bookkeeping. The overall final run has no next
    stage to anchor on, so it must be passed as `is_last` explicitly.

    Sampling per stage is either every `step`-th run, or — when
    `max_measurements` is set — at most that many runs evenly spaced across
    the stage regardless of its length, applied independently within each
    stage (see `_select_ordinals`). This requires knowing each stage's total
    run count up front, hence `stage_counts` rather than a purely streaming
    design.
    """

    def __init__(
        self,
        stage_counts: dict[str, int],
        step_insertion: int,
        step_update: int,
        step_embedding: int,
        max_measurements: int | None = None,
    ):
        steps = {
            "insertion": step_insertion,
            "update": step_update,
            "embedding": step_embedding,
        }
        self._selected = {
            stage: _select_ordinals(stage_counts.get(stage, 0), steps[stage], max_measurements)
            for stage in ("insertion", "update", "embedding")
        }
        self._ordinal = {"insertion": 0, "update": 0, "embedding": 0}

    @staticmethod
    def _stage(record: dict[str, Any]) -> str:
        if not record["insertion_completed"]:
            return "insertion"
        if not record["update_completed"]:
            return "update"
        return "embedding"

    def should_eval(self, record: dict[str, Any], is_last: bool = False) -> bool:
        stage = self._stage(record)
        self._ordinal[stage] += 1
        ordinal = self._ordinal[stage]
        return is_last or ordinal in self._selected[stage]


def select_eval_run_ids(
    records: list[dict[str, Any]],
    step_insertion: int,
    step_update: int,
    step_embedding: int,
    max_measurements: int | None = None,
) -> list[int]:
    stage_counts = Counter(EvalScheduler._stage(r) for r in records)
    scheduler = EvalScheduler(
        stage_counts,
        step_insertion, step_update, step_embedding,
        max_measurements,
    )
    n = len(records)
    return [
        rec["run_id"]
        for i, rec in enumerate(records)
        if scheduler.should_eval(rec, is_last=(i == n - 1))
    ]


def backfill_run_quality(
    records: list[dict[str, Any]],
    runs_dir: Path,
    baselines: dict[str, Any],
    step_insertion: int,
    step_update: int,
    step_embedding: int,
    max_measurements: int | None = None,
    overwrite: bool = False,
    log_progress: bool = False,
) -> int:
    """Compute quality for the selected runs by reloading their saved .npz.

    Must be called only after the benchmark's timing loop has finished — this
    reruns a full KNN graph per selected run, and doing that between live runs
    would bleed GPU time into the next run's `wall_time` measurement even
    though it never touches that field directly (async kernel queues don't
    respect Python-level timers). Loading from disk afterwards keeps the
    recorded per-run timings honest.

    `max_measurements` caps the number of evaluated runs per stage (applied
    independently to each of insertion/update/embedding), evenly spaced,
    overriding the corresponding `step_*` for every stage — see
    `EvalScheduler`.

    Returns the number of runs actually (re)computed.
    """
    eval_ids = select_eval_run_ids(
        records, step_insertion, step_update, step_embedding,
        max_measurements,
    )
    by_id = {r["run_id"]: r for r in records}

    n_done = 0
    for run_id in eval_ids:
        record = by_id[run_id]
        if record.get("quality") is not None and not overwrite:
            continue

        npz_path = runs_dir / f"run_{run_id:06d}.npz"
        if not npz_path.exists():
            LOG.warning("Missing %s — skipping run %d", npz_path, run_id)
            continue

        with np.load(npz_path) as f:
            embedding = f["embedding"]
            nn_indices = f["nn_indices"]

        n_neighbors = nn_indices.shape[1]
        quality = compute_quality(nn_indices, embedding, baselines, n_neighbors)
        record["quality"] = quality
        n_done += 1
        if log_progress:
            LOG.info(
                "run %d: nng_recall=%s  fidelity=%s",
                run_id,
                f"{quality['nng_recall']:.4f}" if "nng_recall" in quality else "n/a",
                f"{quality['fidelity']:.4f}" if "fidelity" in quality else "n/a",
            )

    return n_done
