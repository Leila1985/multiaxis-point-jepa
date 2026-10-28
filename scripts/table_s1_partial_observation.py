#!/usr/bin/env python3
"""Compute rho, z and Top-1 for one compatible pretrained Point-JEPA checkpoint.

Run from the repository root:
    python scripts/table_s1_partial_observation.py evaluate --checkpoint checkpoints/model.ckpt --output-dir outputs/model

The same evaluation corruption is applied to clean- and occlusion-pretrained
models. The default protocol selects 200 ShapeNet55 test objects with NumPy
RandomState(0), uses clean seed 0, fragment seeds 100--109, and batch size 25.
Features concatenate max/mean-pooled student tokens and are L2-normalized.
Rho uses the mean cosine distance between distinct clean objects as denominator.

Outputs: table_s1_metrics.json and table_s1_draws.csv. SD is population SD across
draws (ddof=0), not standard error. No images or plots are generated.
Use a separate output directory for each checkpoint. Processed-input hashes can
be compared across runs to verify identical clean and fragment inputs. Historical
scores require the corresponding checkpoints and execution environment.
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
    """Verify every inference encoder tensor despite unused teacher entries."""
    import torch
    from pointjepa.models.point_jepa import PointJepa

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PointJepa.load_from_checkpoint(
        str(checkpoint), map_location="cpu", svm_validation={}, strict=False
    )
    state = saved["state_dict"]
    actual = model.state_dict()
    prefixes = ("tokenizer.", "student.", "positional_encoding.")
    keys = sorted(k for k in set(state) | set(actual) if k.startswith(prefixes))
    if not keys:
        raise ValueError(f"No inference encoder tensors in {checkpoint}.")
    for key in keys:
        if key not in state or key not in actual:
            raise ValueError(f"Missing inference encoder tensor: {key}")
        a, b = state[key], actual[key]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError(f"Checkpoint encoder tensor not loaded exactly: {key}")
        if not torch.isfinite(b).all():
            raise ValueError(f"Nonfinite encoder tensor: {key}")
    if list(model.hparams.val_transformations) != ["subsample", "center", "unit_sphere"]:
        raise ValueError("Expected validation transforms: subsample, center, unit_sphere.")
    if int(model.hparams.transformation_subsample_points) != 1024:
        raise ValueError("Expected 1024 validation points per cloud.")
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": saved.get("epoch"),
        "global_step": saved.get("global_step"),
        "verified_encoder_entries": len(keys),
        "masking_mode": getattr(model.hparams, "masking_mode", None),
