"""Populate evaluation/data_scaled/ with StandardScaler-scaled copies of the raw
datasets under evaluation/data/. All evaluation entry points load from
SCALED_DATA_ROOT by default (see runner.DEFAULT_DATA_ROOT) so baselines and
GPUMAP runs operate in the same space.

Usage (from core/):
    python -m evaluation.bench.build_scaled_data <dataset> [dataset2 ...]
    python -m evaluation.bench.build_scaled_data --all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from evaluation.runner import RAW_DATA_ROOT, SCALED_DATA_ROOT, resolve_dataset_path

LOG = logging.getLogger("build_scaled_data")

LABEL_FILENAMES = ("{name}_y.npy", "{name}_Y.npy", "labels.npy")


def scale_dataset(name: str, force: bool = False) -> None:
    try:
        raw_path = resolve_dataset_path(name, RAW_DATA_ROOT)
    except FileNotFoundError as e:
        LOG.warning("skip %s: %s", name, e)
        return

    out_dir = SCALED_DATA_ROOT / name
    out_path = out_dir / f"{name}.npy"
    if out_path.exists() and not force:
        LOG.info("skip %s (already scaled, use --force to recompute)", name)
        return

    LOG.info("loading %s from %s", name, raw_path)
    data = np.load(raw_path).astype(np.float32)
    LOG.info("  shape=%s, fitting StandardScaler", data.shape)
    scaled = StandardScaler().fit_transform(data).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, scaled)
    LOG.info("  saved -> %s", out_path)

    for pattern in LABEL_FILENAMES:
        label_path = raw_path.with_name(pattern.format(name=name))
        if label_path.is_file():
            link_path = out_dir / label_path.name
            if not link_path.exists():
                link_path.symlink_to(label_path.resolve())
                LOG.info("  linked label -> %s", link_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="Dataset names to scale")
    parser.add_argument("--all", action="store_true", help="Scale every dataset under RAW_DATA_ROOT")
    parser.add_argument("--force", action="store_true", help="Recompute even if scaled output exists")
    args = parser.parse_args()

    if args.all:
        names = sorted(p.name for p in RAW_DATA_ROOT.iterdir() if p.is_dir())
    elif args.datasets:
        names = args.datasets
    else:
        parser.error("pass dataset names or --all")
        return

    for name in names:
        scale_dataset(name, force=args.force)


if __name__ == "__main__":
    main()
