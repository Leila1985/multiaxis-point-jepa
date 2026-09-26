#!/usr/bin/env python3
"""Reproduce the Table S1 diagnostic for the Multi-axis + Occl. pretrained checkpoint.

Usage (from the repository root):
    python scripts/table_s1_partial_observation.py evaluate \
        --checkpoint checkpoints/multiaxis_occl_pretrain_shapenet.ckpt

It preserves the random protocol behind the reported numbers:

* NumPy seed 0 selects 200 ShapeNet55 test objects without replacement;
* clean codes use PyTorch seed 0;
* ten occlusion realizations use PyTorch seeds 100, ..., 109;
* batch size is 25, because PointcloudOcclusion samples one removal ratio per
  batch and changing the batch size would therefore change the experiment;
* the displayed query is the upper median-z_i object in draw 109.

The raw fragments of draw 109 are captured during feature extraction, so the
geometry shown on the left is the exact query whose similarity row appears on
the right.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

# make `import pointjepa` work when the script is started as `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN_NAME = "multiaxis_occl_pretrain_shapenet"
DEFAULT_CHECKPOINT = Path("checkpoints") / "multiaxis_occl_pretrain_shapenet.ckpt"
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
            if not keys:
                raise ValueError(f"No point-cloud array was found in {path}.")
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


def load_release_model(checkpoint: Path, device: str) -> Tuple[Any, Any]:
    """Load current PointJepa code and obtain its own occlusion transform."""

    try:
        from pointjepa.models.point_jepa import PointJepa
    except ImportError as exc:
        raise RuntimeError(
            "Could not import pointjepa.models.point_jepa.PointJepa. Run this "
            "script from the root of this repository."
        ) from exc

    model = PointJepa.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        svm_validation={},
        strict=False,
    )
    model.to(device).eval()

    expected_hparams = {
        "masking_mode": "pj_multi",
        "ordering_mode": "xyz_axes",
        "num_orderings": 3,
    }
    mismatches = {
        name: (getattr(model.hparams, name, None), expected)
        for name, expected in expected_hparams.items()
        if getattr(model.hparams, name, None) != expected
    }
    if mismatches:
        raise RuntimeError(
            "The checkpoint is not the expected three-axis (x, y, z) model. "
            f"Hyperparameter mismatches: {mismatches}"
        )

    train_transforms = getattr(model.train_transformations, "transforms", ())
    occlusions = [
        transform
        for transform in train_transforms
        if type(transform).__name__ == "PointcloudOcclusion"
    ]
    if len(occlusions) != 1:
        raise RuntimeError(
            "Expected the checkpoint to contain exactly one "
            f"PointcloudOcclusion transform, found {len(occlusions)}."
        )
    occlusion = occlusions[0]
    occlusion.prob = 1.0
    return model, occlusion


def extract_features(
    model: Any,
    occlusion: Any,
    clouds: np.ndarray,
    selected_indices: np.ndarray,
    batch_size: int,
    seed: int,
    apply_occlusion: bool,
    device: str,
    capture_raw_fragments: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Extract the final model's normalized max/mean-pooled student codes.

    The order of all operations matches the original evaluation cell. When
    requested, raw occluded batches are copied before validation transforms
    modify or subsample them.
    """

    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    outputs: List[Any] = []
    captured_batches: List[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(selected_indices), batch_size):
            indices = selected_indices[start : start + batch_size]
            points = torch.from_numpy(clouds[indices]).float().to(device)
            if apply_occlusion:
                points = occlusion(points)
                if capture_raw_fragments:
                    captured_batches.append(points.detach().cpu().numpy().copy())
            points = model.val_transformations(points)
            embeddings, centers = model.tokenizer(points)
            positional = model.positional_encoding(centers)
            hidden = model.student(embeddings, positional).last_hidden_state
            pooled = torch.cat(
                [hidden.max(dim=1).values, hidden.mean(dim=1)], dim=-1
            )
            outputs.append(F.normalize(pooled, dim=-1).cpu())

    return torch.cat(outputs, dim=0).numpy().astype(np.float32), captured_batches


def per_draw_statistics(similarity: np.ndarray) -> Dict[str, Any]:
    """Compute z_i and top-1 from one fragment-to-clean similarity matrix."""

    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("A similarity matrix must be square.")
    n_objects = similarity.shape[0]
    diagonal = np.diag(similarity)
    mask = ~np.eye(n_objects, dtype=bool)
    distractors = similarity[mask].reshape(n_objects, n_objects - 1)
    means = distractors.mean(axis=1)
    # Population standard deviation reproduces torch.nanmean(... ** 2).
    standard_deviations = distractors.std(axis=1, ddof=0)
    if np.any(standard_deviations <= np.finfo(np.float64).eps):
        raise ValueError("A distractor distribution has zero variance.")
    z_values = (diagonal - means) / standard_deviations
    truth = np.arange(n_objects)
    top1_values = similarity.argmax(axis=1) == truth
    return {
        "z_values": z_values,
        "z": float(z_values.mean()),
        "top1_values": top1_values,
        "top1": float(top1_values.mean()),
        "fragment_clean_distance": float(np.mean(1.0 - diagonal)),
    }


def clean_distance_denominator(clean_codes: np.ndarray) -> float:
    """Mean cosine distance over all distinct clean-object pairs."""

    codes = np.asarray(clean_codes, dtype=np.float64)
    similarity = codes @ codes.T
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    denominator = float(np.mean(1.0 - similarity[mask]))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("The clean-object distance denominator is zero.")
    return denominator


def median_z_object(z_values: np.ndarray) -> int:
    """Match the pre-stated rule ``np.argsort(z)[n // 2]``."""

    z_values = np.asarray(z_values)
    return int(np.argsort(z_values)[len(z_values) // 2])


def split_retained_removed(
    clean_points: np.ndarray, fragment_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover retained/removed masks from the exact captured raw fragment."""

    clean_points = np.ascontiguousarray(clean_points[:, :3], dtype=np.float32)
    fragment_points = np.ascontiguousarray(fragment_points[:, :3], dtype=np.float32)
    row_dtype = np.dtype((np.void, clean_points.dtype.itemsize * clean_points.shape[1]))
    clean_rows = clean_points.view(row_dtype).ravel()
    fragment_rows = fragment_points.view(row_dtype).ravel()
    retained_mask = np.isin(clean_rows, fragment_rows)
    if int(retained_mask.sum()) != len(fragment_points):
        # Fallback retained for compatibility with the original plotting cell.
        clean_rounded = np.ascontiguousarray(clean_points.round(6)).view(row_dtype).ravel()
        fragment_rounded = (
            np.ascontiguousarray(fragment_points.round(6)).view(row_dtype).ravel()
        )
        retained_mask = np.isin(clean_rounded, fragment_rounded)
    if int(retained_mask.sum()) != len(fragment_points):
        raise RuntimeError(
            "Could not uniquely match the captured fragment to the clean cloud."
        )
    return clean_points[retained_mask], clean_points[~retained_mask], retained_mask


def plot_diagnostic(
    retained_points: np.ndarray,
    removed_points: np.ndarray,
    similarity_row: np.ndarray,
    true_index: int,
    z_i: float,
    output_stem: Path,
) -> None:
    """Plot the exact ball-occluded query and its clean-gallery similarities."""

    retained_points = np.asarray(retained_points, dtype=np.float32)
    removed_points = np.asarray(removed_points, dtype=np.float32)
    similarity_row = np.asarray(similarity_row, dtype=np.float64)
    distractors = np.delete(similarity_row, true_index)

    all_points = np.concatenate([retained_points, removed_points], axis=0)
    center = all_points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(all_points - center, axis=1).max()
    scale = max(float(radius), np.finfo(np.float32).eps)
    retained_display = (retained_points - center) / scale
    removed_display = (removed_points - center) / scale

    fig = plt.figure(figsize=(11.8, 4.8), constrained_layout=True)
    ax_points = fig.add_subplot(1, 2, 1, projection="3d")
    ax_points.scatter(
        *removed_display.T,
        s=1.5,
        c="lightgray",
        alpha=0.5,
        edgecolors="none",
        label=r"removed $\mathcal{N}_{\lfloor rN\rfloor}(s;P)$",
    )
    ax_points.scatter(
        *retained_display.T,
        s=1.5,
        c="tab:blue",
        alpha=0.6,
        edgecolors="none",
        label=r"retained $T_{s,r}(P)$",
    )
    ax_points.set_title(r"(a) Input geometry: $T_{s,r}(P)$", fontsize=11)
    ax_points.set_axis_off()
    ax_points.set_box_aspect((1, 1, 1))
    ax_points.view_init(elev=18, azim=-55)
    ax_points.legend(loc="upper right", fontsize=8, markerscale=4)

    ax_latent = fig.add_subplot(1, 2, 2)
    ax_latent.hist(
        distractors,
        bins=40,
        color="lightgray",
        edgecolor="gray",
        label=r"distractors $s_{ij},\ j\neq i$",
    )
    ax_latent.axvline(
        similarity_row[true_index],
        color="tab:blue",
        linewidth=2,
        label=rf"true object $s_{{ii}}$ ($z_i={z_i:.2f}$)",
    )
    ax_latent.axvline(
        distractors.mean(),
        color="gray",
        linestyle="--",
        linewidth=1,
        label=r"distractor mean $\mu_i$",
    )
    ax_latent.set_xlabel("cosine similarity of fragment code to clean-object codes")
    ax_latent.set_ylabel("number of distractor objects")
    ax_latent.set_title("(b) Latent space: query identifiability", fontsize=11)
    ax_latent.legend(fontsize=8)
    ax_latent.spines[["top", "right"]].set_visible(False)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_metrics(
    args: argparse.Namespace,
    checkpoint: Path,
    occlusion: Any,
    draw_statistics: Sequence[Mapping[str, Any]],
    rho_denominator: float,
    representative_index: int,
    representative_z: float,
    representative_rank: int,
    retained_count: int,
    original_count: int,
) -> Dict[str, Any]:
    draw_z = np.asarray([statistics["z"] for statistics in draw_statistics])
    draw_top1 = np.asarray([statistics["top1"] for statistics in draw_statistics])
    draw_numerators = np.asarray(
        [statistics["fragment_clean_distance"] for statistics in draw_statistics]
    )
    return {
        "run_name": RUN_NAME,
        "checkpoint": checkpoint.name,
        "protocol": {
            "selection_rng": "numpy RandomState / legacy np.random.choice",
            "selection_seed": int(args.selection_seed),
            "num_objects": int(args.num_objects),
            "batch_size": int(args.batch_size),
            "clean_torch_seed": int(args.clean_seed),
            "occlusion_torch_seeds": [int(seed) for seed in args.draw_seeds],
            "occlusion_probability_at_evaluation": 1.0,
            "removal_ratio_interval": [
                float(occlusion.ratio_min),
                float(occlusion.ratio_max),
            ],
            "representative_rule": "upper median z_i in final draw",
            "distractor_standard_deviation": "population (ddof=0)",
        },
        "representation": "student tokens; concatenated max and mean pooling; L2 normalized",
        "rho": float(draw_numerators.mean() / rho_denominator),
        "rho_fragment_clean_numerator": float(draw_numerators.mean()),
        "rho_clean_pair_denominator": float(rho_denominator),
        "z": float(draw_z.mean()),
        "z_standard_error_over_draws": float(draw_z.std(ddof=0) / math.sqrt(len(draw_z))),
        "top1": float(draw_top1.mean()),
        "per_draw": [
            {
                "seed": int(seed),
                "z": float(statistics["z"]),
                "top1": float(statistics["top1"]),
                "rho_numerator": float(statistics["fragment_clean_distance"]),
            }
            for seed, statistics in zip(args.draw_seeds, draw_statistics)
        ],
        "representative": {
            "draw_seed": int(args.draw_seeds[-1]),
            "local_object_index": int(representative_index),
            "dataset_object_index": int(args.selected_indices[representative_index]),
            "z_i": float(representative_z),
            "retrieval_rank": int(representative_rank),
            "retained_points": int(retained_count),
            "original_points": int(original_count),
            "retained_fraction": float(retained_count / original_count),
        },
    }


def evaluate(args: argparse.Namespace) -> None:
    """Run the reported protocol on the checkpoint and generate the plot."""

    import torch

    protocol_changes = []
    if args.num_objects != DEFAULT_NUM_OBJECTS:
        protocol_changes.append(f"num_objects={args.num_objects}")
    if args.batch_size != DEFAULT_BATCH_SIZE:
        protocol_changes.append(f"batch_size={args.batch_size}")
    if args.selection_seed != DEFAULT_SELECTION_SEED:
        protocol_changes.append(f"selection_seed={args.selection_seed}")
    if args.clean_seed != DEFAULT_CLEAN_SEED:
        protocol_changes.append(f"clean_seed={args.clean_seed}")
    if tuple(args.draw_seeds) != DEFAULT_DRAW_SEEDS:
        protocol_changes.append(f"draw_seeds={tuple(args.draw_seeds)}")
    if protocol_changes and not args.allow_protocol_change:
        raise ValueError(
            "These options change the reported protocol: "
            + ", ".join(protocol_changes)
            + ". Pass --allow-protocol-change only for a deliberately different analysis."
        )
    checkpoint = discover_checkpoint(args.run_dir) if args.run_dir else args.checkpoint
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"{checkpoint} not found. Download it from the Model Zoo (README) into checkpoints/, "
            "or pass --checkpoint / --run-dir."
        )
    clouds = load_shapenet_npz(args.data_npz)
    selected_indices = select_objects(
        len(clouds), args.num_objects, args.selection_seed
    )
    args.selected_indices = selected_indices
    model, occlusion = load_release_model(checkpoint, args.device)

    clean_codes, _ = extract_features(
        model,
        occlusion,
        clouds,
        selected_indices,
        args.batch_size,
        args.clean_seed,
        False,
        args.device,
    )
    rho_denominator = clean_distance_denominator(clean_codes)

    similarity_matrices: List[np.ndarray] = []
    draw_statistics: List[Mapping[str, Any]] = []
    final_raw_batches: List[np.ndarray] = []
    for draw_index, seed in enumerate(args.draw_seeds):
        capture = draw_index == len(args.draw_seeds) - 1
        fragment_codes, raw_batches = extract_features(
            model,
            occlusion,
            clouds,
            selected_indices,
            args.batch_size,
            seed,
            True,
            args.device,
            capture_raw_fragments=capture,
        )
        similarity = fragment_codes @ clean_codes.T
        similarity_matrices.append(similarity.astype(np.float32))
        draw_statistics.append(per_draw_statistics(similarity))
        if capture:
            final_raw_batches = raw_batches

    similarities = np.stack(similarity_matrices, axis=0)
    final_statistics = draw_statistics[-1]
    representative_index = median_z_object(final_statistics["z_values"])
    batch_index, within_batch = divmod(representative_index, args.batch_size)
    fragment_points = final_raw_batches[batch_index][within_batch]
    clean_points = clouds[selected_indices[representative_index]]
    retained, removed, retained_mask = split_retained_removed(
        clean_points, fragment_points
    )
    final_similarity = similarities[-1]
    z_i = float(final_statistics["z_values"][representative_index])
    true_similarity = final_similarity[representative_index, representative_index]
    rank = int(np.sum(final_similarity[representative_index] >= true_similarity))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / f"table_s1_similarity_seed{args.draw_seeds[-1]}.npy",
        final_similarity,
    )
    np.savez_compressed(
        args.output_dir / "table_s1_protocol.npz",
        selected_indices=selected_indices,
        selection_seed=np.asarray(args.selection_seed, dtype=np.int64),
        clean_seed=np.asarray(args.clean_seed, dtype=np.int64),
        draw_seeds=np.asarray(args.draw_seeds, dtype=np.int64),
        batch_size=np.asarray(args.batch_size, dtype=np.int64),
    )
    np.savez_compressed(
        args.output_dir / "table_s1_diagnostic_data.npz",
        similarities=similarities,
        selected_indices=selected_indices,
        draw_seeds=np.asarray(args.draw_seeds, dtype=np.int64),
        representative_draw=np.asarray(len(args.draw_seeds) - 1, dtype=np.int64),
        representative_object=np.asarray(representative_index, dtype=np.int64),
        representative_retained=retained,
        representative_removed=removed,
        representative_retained_mask=retained_mask,
        representative_z=np.asarray(z_i, dtype=np.float64),
    )
    metrics = build_metrics(
        args,
        checkpoint,
        occlusion,
        draw_statistics,
        rho_denominator,
        representative_index,
        z_i,
        rank,
        len(retained),
        len(clean_points),
    )
    with (args.output_dir / "table_s1_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")

    plot_diagnostic(
        retained,
        removed,
        final_similarity[representative_index],
        representative_index,
        z_i,
        args.output_dir / "table_s1_diagnostic",
    )
    print(json.dumps(metrics, indent=2))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def plot_from_data(args: argparse.Namespace) -> None:
    """Regenerate the figure without loading ShapeNet55 or the checkpoint."""

    with np.load(args.data, allow_pickle=False) as data:
        similarities = data["similarities"]
        draw_index = int(data["representative_draw"])
        object_index = int(data["representative_object"])
        z_i = float(data["representative_z"])
        plot_diagnostic(
            data["representative_retained"],
            data["representative_removed"],
            similarities[draw_index, object_index],
            object_index,
            z_i,
            args.output,
        )


def self_test(args: argparse.Namespace) -> None:
    """Test selection, metrics, saved data, and plotting without research assets."""

    rng = np.random.RandomState(17)
    n_objects, n_points = 200, 1024
    selected = select_objects(500, n_objects, 0)
    points = rng.normal(size=(n_points, 3)).astype(np.float32)
    distances = np.linalg.norm(points - points[0], axis=1)
    removed_indices = np.argsort(distances)[: int(0.30 * n_points)]
    retained_mask = np.ones(n_points, dtype=bool)
    retained_mask[removed_indices] = False
    fragment = points[retained_mask]
    retained, removed, recovered_mask = split_retained_removed(points, fragment)
    if not np.array_equal(recovered_mask, retained_mask):
        raise AssertionError("Fragment mask recovery failed.")

    similarities = rng.normal(0.79, 0.035, (10, n_objects, n_objects)).astype(
        np.float32
    )
    for draw in range(10):
        np.fill_diagonal(
            similarities[draw], rng.normal(0.94, 0.008, n_objects).astype(np.float32)
        )
    final_statistics = per_draw_statistics(similarities[-1])
    representative = median_z_object(final_statistics["z_values"])
    z_i = float(final_statistics["z_values"][representative])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "table_s1_diagnostic_data.npz",
        similarities=similarities,
        selected_indices=selected,
        draw_seeds=np.asarray(DEFAULT_DRAW_SEEDS, dtype=np.int64),
        representative_draw=np.asarray(9, dtype=np.int64),
        representative_object=np.asarray(representative, dtype=np.int64),
        representative_retained=retained,
        representative_removed=removed,
        representative_retained_mask=recovered_mask,
        representative_z=np.asarray(z_i, dtype=np.float64),
    )
    plot_diagnostic(
        retained,
        removed,
        similarities[-1, representative],
        representative,
        z_i,
        args.output_dir / "table_s1_diagnostic",
    )
    result = {
        "status": "passed",
        "synthetic_only": True,
        "selected_objects": int(len(selected)),
        "representative_object": int(representative),
        "representative_z": z_i,
    }
    with (args.output_dir / "self_test.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


def parse_draw_seeds(value: str) -> Tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one draw seed is required.")
    return seeds


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the Table S1 partial-observation diagnostic."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    source = evaluate_parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    source.add_argument(
        "--run-dir", type=Path, default=None,
        help="a Lightning run directory; its epoch-499 checkpoint is used",
    )
    evaluate_parser.add_argument(
        "--data-npz",
        type=Path,
        default=Path("data/ShapeNet55/shapenet_test.npz"),
    )
    evaluate_parser.add_argument("--num-objects", type=int, default=DEFAULT_NUM_OBJECTS)
    evaluate_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluate_parser.add_argument("--selection-seed", type=int, default=0)
    evaluate_parser.add_argument("--clean-seed", type=int, default=0)
    evaluate_parser.add_argument(
        "--draw-seeds", type=parse_draw_seeds, default=DEFAULT_DRAW_SEEDS
    )
    evaluate_parser.add_argument("--device", default=default_device())
    evaluate_parser.add_argument("--allow-protocol-change", action="store_true")
    evaluate_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/table_s1/multiaxis_occl")
    )
    evaluate_parser.set_defaults(function=evaluate)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--data", type=Path, required=True)
    plot_parser.add_argument("--output", type=Path, required=True)
    plot_parser.set_defaults(function=plot_from_data)

    test_parser = subparsers.add_parser("self-test")
    test_parser.add_argument("--output-dir", type=Path, required=True)
    test_parser.set_defaults(function=self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
