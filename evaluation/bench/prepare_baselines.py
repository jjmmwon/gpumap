"""
prepare_baselines.py — Build baseline data for fidelity and NNG recall evaluation.

Outputs saved to data/{dataset}/baselines/:
  original_nn_infos.npz       — brute-force KNN in original feature space
  ground_truth_neighbors.npz  — KNN union from repeated UMAP runs (fidelity ground truth)
  nng_recall_baseline.json    — NNDescent graph recall vs brute-force (n_trials runs)
  umap_fidelity_baseline.json — UMAP fidelity vs ground truth (n_fidelity_runs runs)
  params.json                 — parameters used to generate these files

Phases (run all by default, or select via --phases):
  knn           → original_nn_infos.npz
  nng_recall    → nng_recall_baseline.json  (requires knn)
  ground_truth  → ground_truth_neighbors.npz (requires knn)
  fidelity      → umap_fidelity_baseline.json (requires knn + ground_truth)

Usage (from core/):
    python -m evaluation.bench.prepare_baselines --dataset mnist
    python -m evaluation.bench.prepare_baselines --dataset mnist --n-umap-runs 50 --n-trials 5
    python -m evaluation.bench.prepare_baselines --dataset mnist --phases ground_truth fidelity
    python -m evaluation.bench.prepare_baselines --dataset mnist --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpumap import GPUMAP, GPUMAPParams  # noqa: E402
from gpumap.gpumap import find_ab_params  # noqa: E402
from gpumap.data_loader import NumpyDataLoader  # noqa: E402
from gpumap.schedule import WeightedStrategy, DefaultLinearEstimator  # noqa: E402
from evaluation.run_progressive_gpumap import (  # noqa: E402
    resolve_dataset_path,
    to_numpy,
    DEFAULT_DATA_ROOT,
)
from evaluation.utils import (  # noqa: E402
    compute_knn_indices_and_distances,
    compute_umap_embedding,
    compute_ground_truth_neighbors,
)
from evaluation.evaluate_emb_certainty import eval_emb_certainty  # noqa: E402

LOG = logging.getLogger("bench.prepare_baselines")

ALL_PHASES = ["knn", "nng_recall", "ground_truth", "fidelity"]
RECALL_SUBSAMPLE = 10_000   # rows used for NNG recall (brute-force is O(n²))


def _setup_logger(output_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(output_dir / "prepare_baselines.log", encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def _build_gpumap_model(data: np.ndarray, n_neighbors: int) -> GPUMAP:
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    np.save(tmp.name, data)
    tmp.close()

    loader = NumpyDataLoader("baselines", tmp.name)
    os.unlink(tmp.name)

    params = GPUMAPParams()
    params.n_instances = loader.n_instances
    params.n_features = loader.n_features
    params.n_neighbors = n_neighbors
    params.a, params.b = find_ab_params(1.0, params.min_dist)

    strategy = WeightedStrategy(
        n_instances=loader.n_instances,
        n_neighbors=n_neighbors,
        insertion_estimator=DefaultLinearEstimator(),
        update_estimator=DefaultLinearEstimator(),
        embedding_estimator=DefaultLinearEstimator(),
    )
    return GPUMAP(data_loader=loader, params=params, strategy=strategy)


# ── Phase: brute-force KNN ─────────────────────────────────────────────────────


def phase_original_knn(
    data: np.ndarray,
    n_neighbors: int,
    output_dir: Path,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    out_path = output_dir / "original_nn_infos.npz"
    if out_path.exists() and not force:
        LOG.info("Loading cached original_nn_infos.npz")
        with np.load(out_path) as f:
            return f["indices"].astype(np.int32), f["distances"].astype(np.float32)

    LOG.info("Computing brute-force KNN (k=%d, n=%d)...", n_neighbors, len(data))
    t0 = time.perf_counter()
    indices, distances = compute_knn_indices_and_distances(data, n_neighbors)
    LOG.info("  done in %.1fs", time.perf_counter() - t0)

    np.savez(out_path, indices=indices, distances=distances)
    LOG.info("Saved → %s", out_path.name)
    return indices, distances


# ── Phase: NNDescent recall baseline ──────────────────────────────────────────


def phase_nng_recall(
    data: np.ndarray,
    exact_indices: np.ndarray,
    n_neighbors: int,
    n_trials: int,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run GPUMAP's insertion phase n_trials times on the full dataset and compute
    recall of the resulting NNDescent graph against brute-force exact KNN.
    """
    out_path = output_dir / "nng_recall_baseline.json"
    if out_path.exists() and not force:
        LOG.info("Loading cached nng_recall_baseline.json")
        with open(out_path) as f:
            return json.load(f)

    n = len(data)
    n_check = min(RECALL_SUBSAMPLE, n)
    exact_sub = exact_indices[:n_check]  # (n_check, k)
    k = exact_sub.shape[1]

    LOG.info(
        "NNDescent recall baseline: %d trials  n=%d  k=%d  subsample=%d",
        n_trials, n, k, n_check,
    )

    recalls: list[float] = []
    for trial in range(n_trials):
        LOG.info("  trial %d/%d: inserting %d points...", trial + 1, n_trials, n)
        t0 = time.perf_counter()
        model = _build_gpumap_model(data, n_neighbors)
        try:
            model.run_with_n_ops(n, 0, 0)
            nng = to_numpy(model.get_neighbor_indices())  # (n, k)
        finally:
            model.close()

        nng_sub = nng[:n_check, :k]
        valid = (nng_sub >= 0) & (nng_sub < n)
        nng_masked = np.where(valid, nng_sub, -1)
        hit = (nng_masked[:, :, None] == exact_sub[:, None, :]).any(axis=2)
        recall = float(hit[valid[:, : hit.shape[1]]].mean()) if valid.any() else 0.0
        recalls.append(recall)
        LOG.info(
            "  trial %d: recall=%.4f  elapsed=%.1fs",
            trial + 1, recall, time.perf_counter() - t0,
        )

    result: dict[str, Any] = {
        "n_neighbors": n_neighbors,
        "n_trials": n_trials,
        "n_check": n_check,
        "recalls": recalls,
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    LOG.info(
        "Saved → %s  mean=%.4f  std=%.6f",
        out_path.name, result["mean_recall"], result["std_recall"],
    )
    return result


# ── Phase: ground truth neighbor union ────────────────────────────────────────


def phase_ground_truth(
    data: np.ndarray,
    orig_indices: np.ndarray,
    orig_distances: np.ndarray,
    n_neighbors: int,
    n_umap_runs: int,
    output_dir: Path,
    force: bool = False,
) -> np.ndarray:
    """
    Run UMAP n_umap_runs times and accumulate the union of KNN neighborhoods
    in embedding space. The union is the ground truth for fidelity evaluation.

    Output shape: (n, k * n_umap_runs + 1) where the last column is valid count.
    """
    out_path = output_dir / "ground_truth_neighbors.npz"
    key = f"k_{n_neighbors}"
    if out_path.exists() and not force:
        with np.load(out_path) as f:
            if key in f.files:
                LOG.info("Loading cached ground_truth_neighbors.npz (key=%s)", key)
                return f[key].astype(np.int32)

    LOG.info(
        "Building ground truth: %d UMAP runs, k=%d, n=%d",
        n_umap_runs, n_neighbors, len(data),
    )
    knn_indices = orig_indices[:, :n_neighbors].astype(np.int32)
    knn_distances = orig_distances[:, :n_neighbors].astype(np.float32)
    precomputed_knn = (knn_indices, knn_distances)

    emb_nn_list: list[np.ndarray] = []
    for i in range(n_umap_runs):
        LOG.info("  UMAP run %d/%d ...", i + 1, n_umap_runs)
        t0 = time.perf_counter()
        emb = compute_umap_embedding(data, precomputed_knn)
        idx, _ = compute_knn_indices_and_distances(emb, n_neighbors)
        emb_nn_list.append(idx[:, :n_neighbors].astype(np.int32))
        LOG.info("    done in %.1fs", time.perf_counter() - t0)

    ground_truth = compute_ground_truth_neighbors(emb_nn_list, n_neighbors)
    LOG.info(
        "  union built: shape=%s  mean_unique=%.1f",
        ground_truth.shape, float(ground_truth[:, -1].mean()),
    )

    # Merge with any existing keys (other k values) to avoid overwriting
    existing: dict[str, np.ndarray] = {}
    if out_path.exists():
        with np.load(out_path) as f:
            existing = {k: f[k] for k in f.files if k != key}
    np.savez(out_path, **existing, **{key: ground_truth})
    LOG.info("Saved → %s  key=%s", out_path.name, key)
    return ground_truth


# ── Phase: UMAP fidelity baseline ─────────────────────────────────────────────


def phase_fidelity_baseline(
    data: np.ndarray,
    orig_indices: np.ndarray,
    orig_distances: np.ndarray,
    ground_truth: np.ndarray,
    n_neighbors: int,
    n_fidelity_runs: int,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run UMAP n_fidelity_runs times and compute each embedding's fidelity against
    the ground truth KNN union. Fidelity = fraction of embedding KNN that falls
    within the ground truth neighbor set.
    """
    out_path = output_dir / "umap_fidelity_baseline.json"
    if out_path.exists() and not force:
        LOG.info("Loading cached umap_fidelity_baseline.json")
        with open(out_path) as f:
            return json.load(f)

    LOG.info(
        "UMAP fidelity baseline: %d runs, k=%d, n=%d",
        n_fidelity_runs, n_neighbors, len(data),
    )
    knn_indices = orig_indices[:, :n_neighbors].astype(np.int32)
    knn_distances = orig_distances[:, :n_neighbors].astype(np.float32)
    precomputed_knn = (knn_indices, knn_distances)

    fidelity_values: list[float] = []
    for i in range(n_fidelity_runs):
        LOG.info("  fidelity run %d/%d ...", i + 1, n_fidelity_runs)
        t0 = time.perf_counter()
        emb = compute_umap_embedding(data, precomputed_knn)
        emb_idx, _ = compute_knn_indices_and_distances(emb, n_neighbors)
        score, _ = eval_emb_certainty(emb_idx[:, :n_neighbors], ground_truth)
        fidelity_values.append(float(score))
        LOG.info(
            "    fidelity=%.4f  elapsed=%.1fs", score, time.perf_counter() - t0
        )

    result: dict[str, Any] = {
        "fidelity_k": n_neighbors,
        "n_runs": n_fidelity_runs,
        "mean": float(np.mean(fidelity_values)),
        "std": float(np.std(fidelity_values)),
        "values": fidelity_values,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    LOG.info(
        "Saved → %s  mean=%.4f  std=%.4f",
        out_path.name, result["mean"], result["std"],
    )
    return result


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. mnist)")
    p.add_argument("--npy-path", default=None, metavar="FILE")
    p.add_argument(
        "--n-neighbors", type=int, default=15, metavar="K",
        help="Number of neighbors for KNN / UMAP (default: 15)",
    )
    p.add_argument(
        "--n-umap-runs", type=int, default=30, metavar="N",
        help="UMAP runs for building ground truth union (default: 30)",
    )
    p.add_argument(
        "--n-trials", type=int, default=5, metavar="N",
        help="NNDescent insertion trials for recall baseline (default: 5)",
    )
    p.add_argument(
        "--n-fidelity-runs", type=int, default=10, metavar="N",
        help="UMAP runs for UMAP fidelity baseline (default: 10)",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        choices=ALL_PHASES,
        default=ALL_PHASES,
        metavar="PHASE",
        help=f"Phases to run: {ALL_PHASES} (default: all)",
    )
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Override output directory (default: data/{dataset}/baselines/)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Recompute even if output files already exist",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    phases = set(args.phases)

    # Resolve data path
    if args.npy_path:
        npy_path = Path(args.npy_path)
    else:
        npy_path = resolve_dataset_path(args.dataset, DEFAULT_DATA_ROOT)

    # Resolve output dir: data/{dataset}/baselines/
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_DATA_ROOT / args.dataset / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logger(output_dir)

    LOG.info(
        "prepare_baselines: dataset=%s  k=%d  umap_runs=%d  trials=%d  phases=%s",
        args.dataset, args.n_neighbors, args.n_umap_runs, args.n_trials, sorted(phases),
    )
    LOG.info("  data   : %s", npy_path)
    LOG.info("  output : %s", output_dir)

    # Save run params
    params_path = output_dir / "params.json"
    if not params_path.exists() or args.force:
        with open(params_path, "w") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "n_neighbors": args.n_neighbors,
                    "n_umap_runs": args.n_umap_runs,
                    "n_trials": args.n_trials,
                    "n_fidelity_runs": args.n_fidelity_runs,
                    "phases": sorted(phases),
                },
                f,
                indent=4,
            )

    # Load data
    LOG.info("Loading data: %s", npy_path)
    data = np.load(npy_path).astype(np.float32)
    LOG.info("  shape=%s", data.shape)

    # Phase: knn
    orig_indices = orig_distances = None
    if "knn" in phases or any(ph in phases for ph in ("nng_recall", "ground_truth", "fidelity")):
        orig_indices, orig_distances = phase_original_knn(
            data, args.n_neighbors, output_dir, force=args.force
        )

    # Phase: nng_recall
    if "nng_recall" in phases:
        assert orig_indices is not None
        phase_nng_recall(
            data, orig_indices, args.n_neighbors,
            n_trials=args.n_trials,
            output_dir=output_dir,
            force=args.force,
        )

    # Phase: ground_truth
    ground_truth = None
    if "ground_truth" in phases or "fidelity" in phases:
        assert orig_indices is not None and orig_distances is not None
        ground_truth = phase_ground_truth(
            data, orig_indices, orig_distances,
            n_neighbors=args.n_neighbors,
            n_umap_runs=args.n_umap_runs,
            output_dir=output_dir,
            force=args.force,
        )

    # Phase: fidelity baseline
    if "fidelity" in phases:
        if ground_truth is None:
            gt_path = output_dir / "ground_truth_neighbors.npz"
            key = f"k_{args.n_neighbors}"
            if not gt_path.exists():
                LOG.error("ground_truth_neighbors.npz not found — run ground_truth phase first")
                sys.exit(1)
            with np.load(gt_path) as f:
                if key not in f.files:
                    LOG.error("Key %s not in ground_truth_neighbors.npz", key)
                    sys.exit(1)
                ground_truth = f[key].astype(np.int32)

        assert orig_indices is not None and orig_distances is not None
        phase_fidelity_baseline(
            data, orig_indices, orig_distances,
            ground_truth=ground_truth,
            n_neighbors=args.n_neighbors,
            n_fidelity_runs=args.n_fidelity_runs,
            output_dir=output_dir,
            force=args.force,
        )

    LOG.info("Done. Baselines saved to: %s", output_dir)
    _print_summary(output_dir, args.dataset)


def _print_summary(output_dir: Path, dataset: str) -> None:
    print()
    print(f"{'─'*60}")
    print(f"  Baselines: {dataset}  ({output_dir})")
    print(f"{'─'*60}")

    recall_path = output_dir / "nng_recall_baseline.json"
    if recall_path.exists():
        with open(recall_path) as f:
            r = json.load(f)
        print(
            f"  NND recall  : mean={r['mean_recall']:.4f}  std={r['std_recall']:.6f}"
            f"  (n_trials={r['n_trials']}, k={r['n_neighbors']})"
        )

    fidelity_path = output_dir / "umap_fidelity_baseline.json"
    if fidelity_path.exists():
        with open(fidelity_path) as f:
            fi = json.load(f)
        print(
            f"  UMAP fidelity: mean={fi['mean']:.4f}  std={fi['std']:.4f}"
            f"  (n_runs={fi['n_runs']}, k={fi['fidelity_k']})"
        )

    gt_path = output_dir / "ground_truth_neighbors.npz"
    if gt_path.exists():
        with np.load(gt_path) as f:
            keys = list(f.files)
        print(f"  Ground truth : {gt_path.name}  keys={keys}")

    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
