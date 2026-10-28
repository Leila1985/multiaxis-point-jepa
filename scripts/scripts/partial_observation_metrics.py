#!/usr/bin/env python3
"""Compute rho, z and Top-1 for one compatible pretrained Point-JEPA checkpoint.

Run from the repository root:
    python scripts/partial_observation_metrics.py evaluate --checkpoint checkpoints/model.ckpt --output-dir outputs/model

Add --with-chamfer to also compute the raw-geometry reference.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_CHECKPOINT = Path("checkpoints/multiaxis_occl_pretrain_shapenet.ckpt")
DEFAULT_NUM_OBJECTS = 200
DEFAULT_BATCH_SIZE = 25
DEFAULT_DRAW_SEEDS = tuple(range(100, 110))
DEFAULT_CLEAN_SEED = 0
DEFAULT_SELECTION_SEED = 0

def discover_checkpoint(run_dir: Path) -> Path:
    """Find the latest epoch-499 checkpoint under a Lightning run directory."""

    pattern = str(run_dir / "version_*" / "checkpoints" / "epoch=499-step=*.ckpt")
    candidates = [Path(path) for path in glob.glob(pattern)]
    if not candidates:
        raise FileNotFoundError(
            f"No epoch-499 checkpoint found below {run_dir}. "
            "Pass --checkpoint explicitly if the layout differs."
        )

    def key(path: Path) -> Tuple[int, int, str]:
        version_match = re.search(r"version_(\d+)", str(path))
        step_match = re.search(r"step=(\d+)", path.name)
        version = int(version_match.group(1)) if version_match else -1
        step = int(step_match.group(1)) if step_match else -1
        return version, step, str(path)

    return max(candidates, key=key)


def load_shapenet_npz(path: Path) -> np.ndarray:
    """Load the (N, points, channels) array from shapenet_test.npz."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if "data" in archive and archive["data"].ndim == 3:
            key = "data"
        else:
            keys = [
                name
                for name in archive.files
                if archive[name].ndim == 3 and archive[name].shape[-1] >= 3
            ]
            if len(keys) != 1:
                raise ValueError(f"Expected one unambiguous point-cloud array in {path}.")
            key = keys[0]
        clouds = archive[key][:, :, :3].astype(np.float32)
    return clouds


def select_objects(dataset_size: int, num_objects: int, seed: int) -> np.ndarray:
    """Match ``np.random.seed(0); np.random.choice(...)`` from the old code."""

    if num_objects > dataset_size:
        raise ValueError(
            f"Requested {num_objects} objects from a test set of size {dataset_size}."
        )
    # RandomState reproduces the legacy global NumPy RNG exactly.
    rng = np.random.RandomState(seed)
    return rng.choice(dataset_size, num_objects, replace=False).astype(np.int64)


def load_checked_model(checkpoint, device):
