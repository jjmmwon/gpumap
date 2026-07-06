"""
prepare_baselines.py — Build baseline data for benchmark evaluation.

Outputs saved to data/{dataset}/baselines/:
  original_nn_infos.npz       — brute-force exact KNN in original feature space
  nng_recall_baseline.json    — NNDescent ANN recall vs exact KNN (n_trials runs, individual + mean + std)
  ground_truth_neighbors.npz  — KNN union from repeated UMAP runs (fidelity ground truth)
  umap_fidelity_baseline.json — UMAP fidelity vs ground truth (n_fidelity_runs runs, individual + mean + std)
  params.json                 — parameters used to generate these files

Phases (run all by default, or select via --phases):
  knn           → original_nn_infos.npz
  nng_recall    → nng_recall_baseline.json   (requires knn)
  ground_truth  → ground_truth_neighbors.npz (requires knn)
  fidelity      → umap_fidelity_baseline.json (requires knn + ground_truth)

Ground truth convergence criterion (SMA of h/k):
  h/k = mean fraction of neighbors that are new additions to the union per run.
  Stops when SMA(h/k, window=5) <= threshold (default 0.2) or --gt-max-runs reached.

Usage (from core/):
    python -m evaluation.bench.prepare_baselines --dataset mnist
    python -m evaluation.bench.prepare_baselines --dataset mnist --n-trials 5 --gt-max-runs 300
    python -m evaluation.bench.prepare_baselines --dataset mnist --phases nng_recall
    python -m evaluation.bench.prepare_baselines --dataset mnist --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for _p in (str(REPO_ROOT), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cupy as cp  # noqa: E402
from cuvs.neighbors import nn_descent  # noqa: E402
from evaluation.runner import (  # noqa: E402
    resolve_dataset_path,
    DEFAULT_DATA_ROOT,
)
from evaluation.utils import compute_knn_indices_and_distances  # noqa: E402
from evaluation.evaluate_emb_certainty import eval_emb_certainty  # noqa: E402

LOG = logging.getLogger("bench.prepare_baselines")

ALL_PHASES = ["knn", "nng_recall", "ground_truth", "fidelity"]


def _build_umap_runner(data: np.ndarray, n_neighbors: int, n_components: int = 2):
    """
    Build a UMAP runner with no precomputed KNN, so each call fits its own
    fresh approximate KNN graph internally (in addition to a fresh random
    init). This intentionally does NOT reuse a fixed feature-space KNN graph
    across repeated runs: ground truth / fidelity-baseline diversity must
    come from both KNN-graph and init variance, not init variance alone,
    otherwise the resulting neighbor union is an artificially narrow target.
    """
    try:
        data_gpu = cp.asarray(data, dtype=cp.float32)
        from cuml.manifold import UMAP as CuUMAP

        def run_umap() -> np.ndarray:
            emb = CuUMAP(
                n_neighbors=n_neighbors, n_components=n_components, init="random"
            ).fit_transform(data_gpu)
            return cp.asnumpy(emb).astype(np.float32)

        return run_umap
    except Exception as exc:
        LOG.warning("cuML UMAP unavailable (%s); falling back to umap-learn", exc)
        import umap

        def run_umap() -> np.ndarray:
            return (
                umap.UMAP(
                    n_neighbors=n_neighbors, n_components=n_components, init="random"
                )
                .fit_transform(data)
                .astype(np.float32)
            )

        return run_umap
RECALL_SUBSAMPLE = 10_000


def _setup_logger(output_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(output_dir / "prepare_baselines.log", encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


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


# ── Phase: NNDescent recall ───────────────────────────────────────────────────


def phase_nng_recall(
    data: np.ndarray,
    exact_indices: np.ndarray,
    n_neighbors: int,
    n_trials: int,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """
    Build NNDescent graph n_trials times using cuvs.neighbors.nn_descent.
    Each run: compute recall of the ANN graph vs brute-force exact KNN.
    Saves individual recall values + mean + std.
    """
    out_path = output_dir / "nng_recall_baseline.json"
    if out_path.exists() and not force:
        LOG.info("Loading cached nng_recall_baseline.json")
        with open(out_path) as f:
            return json.load(f)

    n = len(data)
    k = n_neighbors
    n_check = min(RECALL_SUBSAMPLE, n)
    exact_sub = exact_indices[:n_check, :k].astype(np.int32)

    LOG.info(
        "NNDescent recall (cuvs): %d trials  n=%d  k=%d  subsample=%d",
        n_trials, n, k, n_check,
    )

    data_gpu = cp.asarray(np.ascontiguousarray(data, dtype=np.float32))
    index_params = nn_descent.IndexParams(
        graph_degree=k,
        intermediate_graph_degree=2 * k,
    )

    recalls: list[float] = []
    for trial in range(n_trials):
        LOG.info("  trial %d/%d ...", trial + 1, n_trials)
        t0 = time.perf_counter()

        index = nn_descent.build(index_params, data_gpu)
        nng = np.asarray(index.graph).astype(np.int32)  # (n, graph_degree)

        nng_sub = nng[:n_check, :k]
        valid = (nng_sub >= 0) & (nng_sub < n)
        nng_masked = np.where(valid, nng_sub, -1)
        hit = (nng_masked[:, :, None] == exact_sub[:, None, :]).any(axis=2)
        recall = float(hit[valid[:, : hit.shape[1]]].mean()) if valid.any() else 0.0
        recalls.append(recall)
        LOG.info("    recall=%.4f  elapsed=%.1fs", recall, time.perf_counter() - t0)

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


# ── Phase: ground truth neighbor union (SMA convergence) ─────────────────────

_UNION_INIT_COLS_PER_RUN = 20  # initial union capacity = this many runs worth
_UNION_EXPAND_THR = 0.9        # expand union array when 90% full
_HK_CHUNK = 8_192              # points processed per chunk in _hk_update_union

GT_DEFAULT_MAX_RUNS = 300
GT_DEFAULT_MA_WINDOW = 5
GT_DEFAULT_MA_THRESHOLD = 0.2


def _hk_update_union(
    union: np.ndarray,
    counts: np.ndarray,
    new_knn: np.ndarray,
) -> float:
    """
    Update union in-place with new_knn neighbors.
    Returns mean h/k = (total new unique neighbors added) / (n * k).
    """
    n, k = new_knn.shape
    total_new = 0

    for s in range(0, n, _HK_CHUNK):
        e = min(s + _HK_CHUNK, n)
        c_chunk = counts[s:e]     # (chunk,)
        knn_chunk = new_knn[s:e]  # (chunk, k)
        max_c = int(c_chunk.max()) if c_chunk.size > 0 else 0

        is_new = np.ones((e - s, k), dtype=bool)
        if max_c > 0:
            u_chunk = union[s:e, :max_c]   # (chunk, max_c)
            for j in range(k):
                is_new[:, j] = ~(knn_chunk[:, j:j+1] == u_chunk).any(axis=1)

        total_new += int(is_new.sum())

        for j in range(k):
            rel_pts = np.where(is_new[:, j])[0]
            if rel_pts.size == 0:
                continue
            abs_pts = rel_pts + s
            in_bounds = counts[abs_pts] < union.shape[1]
            valid = abs_pts[in_bounds]
            if valid.size == 0:
                continue
            vc = counts[valid]
            union[valid, vc] = new_knn[valid, j]
            counts[valid] += 1

    return total_new / (n * k)


def phase_ground_truth(
    data: np.ndarray,
    n_neighbors: int,
    gt_max_runs: int,
    gt_ma_window: int,
    gt_ma_threshold: float,
    output_dir: Path,
    force: bool = False,
) -> np.ndarray:
    """
    Run UMAP up to gt_max_runs times, accumulating the union of KNN neighborhoods.
    Stops early when SMA(h/k, window=gt_ma_window) <= gt_ma_threshold (convergence).

    h/k = mean fraction of each run's k neighbors that are new to the union.

    Output: (n, max_unique + 1) where the last column = per-point unique count.
    Also saves h/k convergence history as 'convergence_{k}' key in the npz.
    """
    out_path = output_dir / "ground_truth_neighbors.npz"
    key = f"k_{n_neighbors}"
    convergence_key = f"convergence_{n_neighbors}"
    if out_path.exists() and not force:
        with np.load(out_path) as f:
            if key in f.files:
                LOG.info("Loading cached ground_truth_neighbors.npz (key=%s)", key)
                return f[key].astype(np.int32)

    n = len(data)
    k = n_neighbors
    init_cols = k * _UNION_INIT_COLS_PER_RUN
    union = np.full((n, init_cols), -1, dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)
    run_umap = _build_umap_runner(data, k, n_components=2)

    hk_history: list[float] = []
    stop_reason = f"max_runs={gt_max_runs}"

    LOG.info(
        "Building ground truth: max_runs=%d  ma_window=%d  ma_threshold=%.2f  k=%d  n=%d",
        gt_max_runs, gt_ma_window, gt_ma_threshold, k, n,
    )

    for run_idx in range(1, gt_max_runs + 1):
        LOG.info("  GT run %d/%d ...", run_idx, gt_max_runs)
        t0 = time.perf_counter()
        emb = run_umap()
        idx, _ = compute_knn_indices_and_distances(emb, k)

        # Expand union array if nearly full
        cur_max = int(counts.max())
        if cur_max >= union.shape[1] * _UNION_EXPAND_THR:
            new_cols = min(union.shape[1] * 2, gt_max_runs * k)
            if new_cols > union.shape[1]:
                LOG.info("  Expanding union: %d → %d cols", union.shape[1], new_cols)
                new_union = np.full((n, new_cols), -1, dtype=np.int32)
                new_union[:, :union.shape[1]] = union
                union = new_union

        mean_hk = _hk_update_union(union, counts, idx[:, :k])
        hk_history.append(mean_hk)

        LOG.info(
            "    h/k=%.4f  mean_unique=%.1f  elapsed=%.1fs",
            mean_hk, float(counts.mean()), time.perf_counter() - t0,
        )

        if len(hk_history) >= gt_ma_window:
            sma = float(np.mean(hk_history[-gt_ma_window:]))
            if sma <= gt_ma_threshold:
                stop_reason = f"converged (run={run_idx}, sma={sma:.4f})"
                LOG.info(
                    "  Converged at run %d: SMA(h/k, window=%d)=%.4f <= %.2f",
                    run_idx, gt_ma_window, sma, gt_ma_threshold,
                )
                break

    actual_max = int(counts.max())
    ground_truth = np.concatenate(
        [union[:, :actual_max], counts[:, None].astype(np.int32)],
        axis=1,
    ).astype(np.int32)

    LOG.info(
        "  union complete: shape=%s  mean_unique=%.1f  stop=%s",
        ground_truth.shape, float(counts.mean()), stop_reason,
    )

    # Merge with any existing keys to avoid overwriting other k values
    save_data: dict[str, np.ndarray] = {
        key: ground_truth,
        convergence_key: np.array(hk_history, dtype=np.float32),
    }
    if out_path.exists():
        with np.load(out_path) as f:
            for fname in f.files:
                if fname not in (key, convergence_key):
                    save_data[fname] = f[fname]
    np.savez(out_path, **save_data)  # type: ignore[arg-type]
    LOG.info("Saved → %s  key=%s", out_path.name, key)
    return ground_truth


# ── Phase: UMAP fidelity baseline ─────────────────────────────────────────────


def phase_fidelity_baseline(
    data: np.ndarray,
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
    run_umap = _build_umap_runner(data, n_neighbors, n_components=2)

    fidelity_values: list[float] = []
    for i in range(n_fidelity_runs):
        LOG.info("  fidelity run %d/%d ...", i + 1, n_fidelity_runs)
        t0 = time.perf_counter()
        emb = run_umap()
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
        "--gt-max-runs", type=int, default=GT_DEFAULT_MAX_RUNS, metavar="N",
        help=f"Ground truth: max UMAP runs before forced stop (default: {GT_DEFAULT_MAX_RUNS})",
    )
    p.add_argument(
        "--gt-ma-window", type=int, default=GT_DEFAULT_MA_WINDOW, metavar="W",
        help=f"Ground truth: SMA window for h/k convergence (default: {GT_DEFAULT_MA_WINDOW})",
    )
    p.add_argument(
        "--gt-ma-threshold", type=float, default=GT_DEFAULT_MA_THRESHOLD, metavar="T",
        help=f"Ground truth: SMA threshold for convergence (default: {GT_DEFAULT_MA_THRESHOLD})",
    )
    p.add_argument(
        "--n-trials", type=int, default=5, metavar="N",
        help="NNDescent recall: number of independent runs (default: 5)",
    )
    p.add_argument(
        "--n-fidelity-runs", type=int, default=10, metavar="N",
        help="UMAP fidelity: number of independent runs (default: 10)",
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
        "prepare_baselines: dataset=%s  k=%d  n_trials=%d  gt_max_runs=%d  phases=%s",
        args.dataset, args.n_neighbors, args.n_trials, args.gt_max_runs, sorted(phases),
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
                    "gt_max_runs": args.gt_max_runs,
                    "gt_ma_window": args.gt_ma_window,
                    "gt_ma_threshold": args.gt_ma_threshold,
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
            data=data,
            exact_indices=orig_indices,
            n_neighbors=args.n_neighbors,
            n_trials=args.n_trials,
            output_dir=output_dir,
            force=args.force,
        )

    # Phase: ground_truth
    ground_truth = None
    if "ground_truth" in phases or "fidelity" in phases:
        ground_truth = phase_ground_truth(
            data,
            n_neighbors=args.n_neighbors,
            gt_max_runs=args.gt_max_runs,
            gt_ma_window=args.gt_ma_window,
            gt_ma_threshold=args.gt_ma_threshold,
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

        phase_fidelity_baseline(
            data,
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
        vals = "  ".join(f"{v:.4f}" for v in r["recalls"])
        print(
            f"  NNDescent recall: mean={r['mean_recall']:.4f}  std={r['std_recall']:.6f}"
            f"  (n_trials={r['n_trials']}, k={r['n_neighbors']})"
        )
        print(f"    values: [{vals}]")

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
