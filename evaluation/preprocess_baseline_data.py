import argparse
import logging
import json
import os
import time
import numpy as np
from pathlib import Path

from .utils import (
    compute_knn_indices_and_distances,
    compute_umap_embedding,
    compute_ground_truth_neighbors,
)
from .evaluate_emb_quality import evaluate_TNC, evaluate_LCMC
from .evaluate_emb_certainty import eval_emb_certainty


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    )

    logger.addHandler(logging.StreamHandler())

    fh = logging.FileHandler(output_dir / "preprocess.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


PARAMS = {
    "data_path": None,  # str  : path to the input data .npy file
    "output_dir": None,  # str  : directory to save the processed results
    "n_umap_runs": 10,  # int  : number of UMAP runs
    "n_neighbors": 15,  # int  : number of neighbors
}


ALL_PHASES = [1, 2, 3, 4, 5, 6, 7, 8]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess baseline data for evaluation")
    p.add_argument("--data-path", type=str, default=PARAMS["data_path"])
    p.add_argument("--output-dir", type=str, default=PARAMS["output_dir"])
    p.add_argument("--n-umap-runs", type=int, default=PARAMS["n_umap_runs"])
    p.add_argument("--n-neighbors", type=int, default=PARAMS["n_neighbors"])
    p.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=ALL_PHASES,
        metavar="N",
        help=f"phases to run (default: all {ALL_PHASES})",
    )
    return p.parse_args()


def _require_file(path: Path, produced_by: int) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"  → Please run phase {produced_by} first to generate this file."
        )
    return np.load(path)


def _ensure_data(
    data: np.ndarray | None, args: argparse.Namespace, log: logging.Logger
) -> np.ndarray:
    if data is not None:
        return data
    return phase1_load_data(args.data_path, log)


def _ensure_orig_nn_infos(
    orig: tuple | None, output_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    if orig is not None:
        return orig
    f = _require_file(output_dir / "original_nn_infos.npz", produced_by=2)
    return f["indices"], f["distances"]


def _ensure_embeddings(embeddings: list | None, output_dir: Path) -> list[np.ndarray]:
    if embeddings is not None:
        return embeddings
    f = _require_file(output_dir / "embeddings.npz", produced_by=3)
    return list(f["embeddings"])  # (n_runs, n, 2) → list of (n, 2)


def _ensure_emb_nn_infos(
    emb_nn: tuple | None, output_dir: Path
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if emb_nn is not None:
        return emb_nn
    f = _require_file(output_dir / "embeddings_nn_infos.npz", produced_by=4)
    return list(f["nn_indices"]), list(f["nn_distances"])


def _ensure_ground_truth(ground_truth: dict[int, np.ndarray] | None, output_dir: Path):
    if ground_truth is not None:
        return ground_truth

    f = _require_file(output_dir / "ground_truth_neighbors.npz", produced_by=5)
    gt = {}
    for key in f.files:
        if key.startswith("k_"):
            k = int(key[2:])
            gt[k] = f[key]
    return gt


def phase1_load_data(data_path: str, log: logging.Logger) -> np.ndarray:
    log.info("Loading data from %s", data_path)
    data = np.load(data_path).astype(np.float32)
    log.info("  shape: %s", data.shape)
    return data


def phase2_original_nn_infos(
    data: np.ndarray,
    max_k: int,
    output_dir: Path,
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    log.info("Computing pairwise distances (original space)...")

    orig_nn_indices, orig_nn_distances = compute_knn_indices_and_distances(data, max_k)

    np.savez(
        output_dir / "original_nn_infos.npz",
        indices=orig_nn_indices,
        distances=orig_nn_distances,
    )
    log.info("Saved → original_nn_infos.npz")

    return orig_nn_indices, orig_nn_distances


def phase3_umap_embeddings(
    data: np.ndarray,
    orig_nn_indices: np.ndarray,
    orig_nn_distances: np.ndarray,
    n_neighbors: int,
    n_umap_runs: int,
    output_dir: Path,
    log: logging.Logger,
    without_save: bool = False,
) -> list[np.ndarray]:
    knn_indices = orig_nn_indices[:, :n_neighbors].astype(np.int32)
    knn_distances = orig_nn_distances[:, :n_neighbors].astype(np.float32)
    precomputed_knn = (knn_indices, knn_distances)

    embeddings = []
    for i in range(n_umap_runs):
        log.info("Running UMAP %d/%d...", i + 1, n_umap_runs)
        emb = compute_umap_embedding(data, precomputed_knn, n_components=2)
        embeddings.append(emb)
        log.info("Done UMAP %d/%d", i + 1, n_umap_runs)

    if without_save:
        return embeddings

    np.savez(
        output_dir / "embeddings.npz",
        embeddings=np.stack(embeddings, axis=0).astype(np.float32),
    )
    log.info("Saved → embeddings.npz")

    return embeddings


def phase4_embedding_nn_infos(
    embeddings: list[np.ndarray],
    max_k: int,
    output_dir: Path,
    log: logging.Logger,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    nn_indices_list, nn_distances_list = [], []

    for i, emb in enumerate(embeddings):
        log.info("Computing nn_infos for embedding %d/%d...", i + 1, len(embeddings))
        idx, dist = compute_knn_indices_and_distances(emb, max_k)

        nn_indices_list.append(idx)
        nn_distances_list.append(dist)
        log.info("Done nn_infos for embedding %d/%d", i + 1, len(embeddings))

    np.savez(
        output_dir / "embeddings_nn_infos.npz",
        nn_indices=np.stack(nn_indices_list, axis=0),
        nn_distances=np.stack(nn_distances_list, axis=0),
    )
    log.info("Saved → embeddings_nn_infos.npz")

    return nn_indices_list, nn_distances_list


def phase5_ground_truth_neighbors(
    embeddings_nn_indices: list[np.ndarray],
    k_list: list[int],
    output_dir: Path,
    log: logging.Logger,
) -> dict[int, np.ndarray]:
    ground_truth = {}
    for _k in k_list:
        log.info("Computing ground truth neighbors for k=%d...", _k)
        exact_neighbors = [idx[:, :_k] for idx in embeddings_nn_indices]
        ground_truth[_k] = compute_ground_truth_neighbors(exact_neighbors, _k)

    np.savez(
        output_dir / "ground_truth_neighbors.npz",
        **{f"k_{_k}": gt for _k, gt in ground_truth.items()},
    )
    log.info("Saved → ground_truth_neighbors.npz")

    return ground_truth


def phase6_emb_quality_scores(
    orig_nn_indices: np.ndarray,
    embeddings_nn_indices: list[np.ndarray],
    k_list: list[int],
    output_dir: Path,
    log: logging.Logger,
) -> None:
    t_scores = {_k: [] for _k in k_list}
    c_scores = {_k: [] for _k in k_list}
    lcmc_scores = {_k: [] for _k in k_list}

    for i, _ in enumerate(embeddings_nn_indices):
        log.info("Evaluating embedding %d/%d...", i + 1, len(embeddings_nn_indices))
        tnc_score = evaluate_TNC(
            orig_nn_indices,
            embeddings_nn_indices[i],
            k_list=k_list,
        )
        lcmc_score = evaluate_LCMC(
            orig_nn_indices, embeddings_nn_indices[i], k_list=k_list
        )

        for s in tnc_score:
            t_scores[s["k"]].append(s["trustworthiness"])
            c_scores[s["k"]].append(s["continuity"])
        for s in lcmc_score:
            lcmc_scores[s["k"]].append(s["lcmc"])

    trustworthiness = {k: float(np.mean(t_scores[k])) for k in k_list}
    continuity = {k: float(np.mean(c_scores[k])) for k in k_list}
    lcmc = {k: float(np.mean(lcmc_scores[k])) for k in k_list}

    with open(output_dir / "emb_qual_scores.json", "w") as f:
        json.dump(
            {
                "t_scores": t_scores,
                "c_scores": c_scores,
                "lcmc_scores": lcmc_scores,
                "trustworthiness": trustworthiness,
                "continuity": continuity,
                "LCMC": lcmc,
            },
            f,
            indent=4,
        )
    log.info("Saved → emb_qual_scores.json")


def phase7_umap_runtime(
    data: np.ndarray,
    n_neighbors: int,
    n_umap_runs: int,
    output_dir: Path,
    log: logging.Logger,
) -> list[np.ndarray]:
    from cuml.manifold import UMAP

    run_times = []
    embeddings = []

    for i in range(n_umap_runs):
        log.info("Running UMAP for runtime measurement %d/%d...", i + 1, n_umap_runs)
        umap = UMAP(n_neighbors=n_neighbors)
        start = time.time()
        emb = umap.fit_transform(data)
        end = time.time()
        embeddings.append(emb)
        run_times.append(end - start)

    results = {
        "n_neighbors": n_neighbors,
        "n_umap_runs": n_umap_runs,
        "run_times": run_times,
        "avg_runtime": float(np.mean(run_times)),
        "std_runtime": float(np.std(run_times)),
    }

    with open(output_dir / "umap_runtime.json", "w") as f:
        json.dump(results, f, indent=4)
    log.info("Saved → umap_runtime.json")

    return embeddings


def phase8_embedding_certainty(
    embeddings: list[np.ndarray],
    ground_truth: dict[int, np.ndarray],
    output_dir: Path,
    log: logging.Logger,
) -> None:
    certainties = {k_: [] for k_ in ground_truth.keys()}
    certainties_stat = {k_: {} for k_ in ground_truth.keys()}

    log.info(
        "Evaluating embedding certainty for each k in %s...",
        sorted(ground_truth.keys()),
    )

    for emb in embeddings:
        knn_indices, _ = compute_knn_indices_and_distances(
            emb, max(ground_truth.keys())
        )
        for k, gt_neighbors in ground_truth.items():
            certainty, _ = eval_emb_certainty(knn_indices[:, :k], gt_neighbors)
            certainties[k].append(float(certainty))

    for k in certainties.keys():
        certainties_stat[k] = {
            "mean": float(np.mean(certainties[k])),
            "std": float(np.std(certainties[k])),
        }

    result = {
        "certainties_stat": certainties_stat,
        "certainties": certainties,
    }

    with open(output_dir / "embedding_certainty.json", "w") as f:
        json.dump(result, f, indent=4)

    log.info("Saved → embedding_certainty.json")


def main(args: argparse.Namespace) -> None:
    assert args.data_path is not None
    assert args.output_dir is not None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(output_dir)

    phases = set(args.phases)
    log.info("실행할 phases: %s", sorted(phases))

    k_list = [15, 30, 50]
    MAX_NEIGHBORS = args.n_neighbors * 10  # to compute TNC

    if not os.path.isfile(output_dir / "params.json"):
        with open(output_dir / "params.json", "w") as f:
            json.dump(
                {
                    "n_umap_runs": args.n_umap_runs,
                    "n_neighbors": args.n_neighbors,
                    "k_list": k_list,
                    "MAX_NEIGHBORS": MAX_NEIGHBORS,
                },
                f,
                indent=4,
            )

    # 각 phase의 결과를 메모리에 보관; 건너뛴 phase는 필요 시 파일에서 로드
    data = None
    orig_nn_infos = None  # (indices, distances)
    embeddings = None
    emb_nn_infos = None  # (indices_list, distances_list, ranking_list)
    ground_truth = None
    embeddings_for_certainty = None

    if 1 in phases:
        data = phase1_load_data(args.data_path, log)

    if 2 in phases:
        data = _ensure_data(data, args, log)
        orig_nn_infos = phase2_original_nn_infos(
            data, max_k=MAX_NEIGHBORS, output_dir=output_dir, log=log
        )

    if 3 in phases:
        data = _ensure_data(data, args, log)
        orig_nn_infos = _ensure_orig_nn_infos(orig_nn_infos, output_dir)
        orig_nn_indices, orig_nn_distances = orig_nn_infos
        embeddings = phase3_umap_embeddings(
            data,
            orig_nn_indices,
            orig_nn_distances,
            args.n_neighbors,
            args.n_umap_runs,
            output_dir,
            log,
        )

    if 4 in phases:
        embeddings = _ensure_embeddings(embeddings, output_dir)
        emb_nn_infos = phase4_embedding_nn_infos(
            embeddings, MAX_NEIGHBORS, output_dir, log
        )

    if 5 in phases:
        emb_nn_infos = _ensure_emb_nn_infos(emb_nn_infos, output_dir)
        emb_nn_indices, _ = emb_nn_infos
        ground_truth = phase5_ground_truth_neighbors(
            emb_nn_indices, k_list, output_dir, log
        )

    if 6 in phases:
        orig_nn_infos = _ensure_orig_nn_infos(orig_nn_infos, output_dir)
        emb_nn_infos = _ensure_emb_nn_infos(emb_nn_infos, output_dir)
        orig_nn_indices, _ = orig_nn_infos
        emb_nn_indices, _ = emb_nn_infos
        phase6_emb_quality_scores(
            orig_nn_indices,
            emb_nn_indices,
            k_list,
            output_dir,
            log,
        )

    if 7 in phases:
        data = _ensure_data(data, args, log)
        embeddings_for_certainty = phase7_umap_runtime(
            data, args.n_neighbors, args.n_umap_runs, output_dir, log
        )

    if 8 in phases:
        ground_truth = _ensure_ground_truth(ground_truth, output_dir)
        if embeddings_for_certainty is None:
            data = _ensure_data(data, args, log)
            orig_nn_infos = _ensure_orig_nn_infos(orig_nn_infos, output_dir)
            orig_nn_indices, orig_nn_distances = orig_nn_infos
            embeddings_for_certainty = phase3_umap_embeddings(
                data,
                orig_nn_indices,
                orig_nn_distances,
                args.n_neighbors,
                args.n_umap_runs,
                output_dir,
                log,
                without_save=True,
            )

        phase8_embedding_certainty(
            embeddings_for_certainty, ground_truth, output_dir, log
        )


if __name__ == "__main__":
    main(parse_args())
