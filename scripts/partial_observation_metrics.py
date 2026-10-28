#!/usr/bin/env python3
"""Compute rho, z and Top-1 for one compatible pretrained Point-JEPA checkpoint.

Run from the repository root:
    python scripts/partial_observation_metrics.py evaluate --checkpoint checkpoints/model.ckpt --output-dir outputs/model

The same evaluation corruption is applied to clean- and occlusion-pretrained
models. The default protocol selects 200 ShapeNet55 test objects with NumPy
RandomState(0), uses clean seed 0, fragment seeds 100--109, and batch size 25.
Features concatenate max/mean-pooled student tokens and are L2-normalized.
Rho uses the mean cosine distance between distinct clean objects as denominator.

Outputs: partial_observation_metrics.json and partial_observation_draws.csv. SD is population SD across
draws (ddof=0), not standard error. No images or plots are generated.
Add --with-chamfer to compute the raw-geometry rho reference on the exact same
processed inputs. This uses symmetric, unsquared Euclidean Chamfer distance:
0.5 * (mean nearest-neighbor distance in each direction). Its denominator is
the mean over all distinct clean-object pairs. It is separate from latent rho.
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
        "encoder_pe_mode": getattr(model.hparams, "encoder_pe_mode", None),
    }
    # A missing PE mode in a historical checkpoint is not evidence of its
    # training execution. Frozen evaluation does not call either pretraining path.
    return model.to(device).eval(), metadata


def extract_features(
    model: Any,
    occlusion: Any,
    clouds: np.ndarray,
    selected_indices: np.ndarray,
    batch_size: int,
    seed: int,
    apply_occlusion: bool,
    device: str,
    capture_points: bool = False,
) -> Tuple[Any, str, Any]:
    """Preserve the archived RNG/forward order and hash processed inputs."""

    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    outputs: List[Any] = []
    input_hash = hashlib.sha256()
    point_batches = []

    with torch.no_grad():
        for start in range(0, len(selected_indices), batch_size):
            indices = selected_indices[start : start + batch_size]
            points = torch.from_numpy(clouds[indices]).float().to(device)
            if apply_occlusion:
                points = occlusion(points)
            points = model.val_transformations(points)
            if tuple(points.shape[1:]) != (1024, 3) or not torch.isfinite(points).all():
                raise ValueError("Expected finite (batch, 1024, 3) encoder inputs.")
            host_points = points.detach().cpu().contiguous()
            input_hash.update(host_points.numpy().tobytes())
            if capture_points:
                # Capture after validation preprocessing, before tokenization.
                # No new sampling or random draws are introduced.
                point_batches.append(host_points.clone())
            embeddings, centers = model.tokenizer(points)
            positional = model.positional_encoding(centers)
            hidden = model.student(embeddings, positional).last_hidden_state
            pooled = torch.cat(
                [hidden.max(dim=1).values, hidden.mean(dim=1)], dim=-1
            )
            outputs.append(F.normalize(pooled, dim=-1))

    codes = torch.cat(outputs, dim=0)
    if not torch.isfinite(codes).all():
        raise ValueError("Nonfinite encoder features.")
    captured = torch.cat(point_batches) if capture_points else None
    return codes, input_hash.hexdigest(), captured


def per_draw_statistics(similarity: np.ndarray) -> Dict[str, Any]:
    """Compute z_i and top-1 from one fragment-to-clean similarity matrix."""

    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("A similarity matrix must be square.")
    n_objects = similarity.shape[0]
    if n_objects < 3 or not np.isfinite(similarity).all():
        raise ValueError("At least three objects and finite similarities are required.")
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


def summarize_draws(draws, denominator):
    """Aggregate means and descriptive population SD over the evaluated draws."""
    if not draws or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Draws and a finite positive denominator are required.")
    numerators = np.asarray([row["rho_numerator"] for row in draws], dtype=np.float64)
    arrays = {
        "rho": numerators / denominator,
        "z": np.asarray([row["z"] for row in draws], dtype=np.float64),
        "top1": np.asarray([row["top1"] for row in draws], dtype=np.float64),
    }
    result = {}
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {name} values.")
        result[name] = float(values.mean())
        result[name + "_std"] = float(values.std(ddof=0))
    result["rho"] = float(numerators.mean() / denominator)
    result["rho_fragment_clean_numerator"] = float(numerators.mean())
    result["rho_clean_pair_denominator"] = float(denominator)
    return result


def chamfer_batch(a, b):
    """One symmetric unsquared Chamfer distance per pair of point clouds.

    As in the archived notebook: float32 distances and float64 reductions.
    Direct cdist avoids the matrix-multiplication distance shortcut.
    """
    import torch
    with torch.no_grad():
        distances = torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        forward = distances.min(dim=2).values.double().mean(dim=1)
        reverse = distances.min(dim=1).values.double().mean(dim=1)
        return 0.5 * (forward + reverse)


def summarize_geometry(numerators, denominator):
    numerators = np.asarray(numerators, dtype=np.float64)
    if (numerators.ndim != 1 or not len(numerators)
            or not np.isfinite(numerators).all() or np.any(numerators < 0)
            or not np.isfinite(denominator) or denominator <= 0):
        raise ValueError("Invalid Chamfer numerator or clean-clean denominator.")
    rhos = numerators / denominator
    return {
        "rho": float(numerators.mean() / denominator),
        "rho_std": float(rhos.std(ddof=0)),
        "mean_fragment_whole_chamfer": float(numerators.mean()),
        "mean_clean_clean_chamfer": float(denominator),
    }


def compute_geometry(clean_points, fragment_points, device):
    """Compute geometry after feature extraction to preserve its RNG sequence."""
    import torch
    n = len(clean_points)
    if n < 2 or not fragment_points:
        raise ValueError("At least two clean clouds and one fragment draw are required.")
    if any(points.shape != clean_points.shape for points in fragment_points):
        raise ValueError("Clean and fragment draws must have matching processed shapes.")
    pair_batch = 16
    with torch.no_grad():
        # These checks distinguish the unsquared, averaged convention from
        # squared Chamfer or the unaveraged sum of both directions.
        origin = torch.tensor([[[0., 0., 0.]]], device=device)
        offset = torch.tensor([[[3., 4., 0.]]], device=device)
        if not torch.allclose(chamfer_batch(origin, offset),
                              torch.tensor([5.], dtype=torch.float64, device=device)):
            raise RuntimeError("Chamfer distance convention check failed.")
        if float(chamfer_batch(origin, origin)[0]) != 0.0:
            raise RuntimeError("Chamfer self-distance check failed.")

        clean = clean_points.to(device)
        pairs = torch.triu_indices(n, n, offset=1)
        distances = []
        print(f"Computing Chamfer for {pairs.shape[1]} distinct clean pairs...", flush=True)
        for start in range(0, pairs.shape[1], pair_batch):
            ids = pairs[:, start:start + pair_batch].to(device)
            distances.append(chamfer_batch(clean[ids[0]], clean[ids[1]]).cpu())
        # For this symmetric distance, averaging unordered pairs is equivalent
        # to averaging all off-diagonal ordered pairs.
        denominator = float(torch.cat(distances).mean())
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid clean-clean Chamfer denominator.")
        numerators = []
        for draw, fragments in enumerate(fragment_points):
            values = []
            for start in range(0, n, pair_batch):
                values.append(chamfer_batch(
                    clean[start:start + pair_batch],
                    fragments[start:start + pair_batch].to(device),
                ).cpu())
            numerator = float(torch.cat(values).mean())
            numerators.append(numerator)
            print(f"Geometry draw {draw}: rho={numerator / denominator:.6f}", flush=True)

    summary = {
        **summarize_geometry(numerators, denominator),
        "distance": "Symmetric unsquared Euclidean Chamfer, factor 0.5",
        "coordinates": "Actual encoder inputs after subsample, center and unit_sphere",
        "clean_pairs_unordered": int(pairs.shape[1]),
        "standard_deviation": "population across draws (ddof=0), not standard error",
    }
    return summary, numerators


def evaluate(args):
    import torch
    from pointjepa.utils.transforms import PointcloudOcclusion

    if args.num_objects < 3 or args.batch_size < 1:
        raise ValueError("Use at least three objects and a positive batch size.")
    if len(set(args.draw_seeds)) != len(args.draw_seeds):
        raise ValueError("Draw seeds must be distinct.")
    changed = []
    for name, expected in (
        ("num_objects", DEFAULT_NUM_OBJECTS), ("batch_size", DEFAULT_BATCH_SIZE),
        ("selection_seed", DEFAULT_SELECTION_SEED), ("clean_seed", DEFAULT_CLEAN_SEED),
        ("draw_seeds", DEFAULT_DRAW_SEEDS),
    ):
        if getattr(args, name) != expected:
            changed.append(name)
    if changed and not args.allow_protocol_change:
        raise ValueError("Options change the reported protocol: " + ", ".join(changed)
                         + ". Use --allow-protocol-change for a different analysis.")
    checkpoint = discover_checkpoint(args.run_dir) if args.run_dir else args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    clouds = load_shapenet_npz(args.data_npz)
    selected = select_objects(len(clouds), args.num_objects, args.selection_seed)
    if clouds.shape[1] != 8192 and not args.allow_protocol_change:
        raise ValueError("The reported protocol uses 8192-point ShapeNet inputs.")
    if not np.isfinite(clouds[selected]).all():
        raise ValueError("Nonfinite input coordinates.")

    model, checkpoint_info = load_checked_model(checkpoint, args.device)
    # Evaluation corruption is independent of the checkpoint's training recipe.
    occlusion = PointcloudOcclusion(ratio_min=0.25, ratio_max=0.40, prob=1.0, min_keep=1100)
    clean, clean_hash, clean_points = extract_features(
        model, occlusion, clouds, selected, args.batch_size, args.clean_seed, False,
        args.device, capture_points=args.with_chamfer
    )
    # Match notebook arithmetic: feature-device float32 matmul, float64 reduction.
    similarity = (clean @ clean.T).cpu().double()
    offdiag = ~torch.eye(args.num_objects, dtype=torch.bool)
    denominator = float((1.0 - similarity[offdiag]).mean())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid clean-clean cosine-distance denominator.")

    draws = []
    fragment_points = []
    hashes = {"clean": clean_hash, "fragments": []}
    for seed in args.draw_seeds:
        fragments, fragment_hash, captured = extract_features(
            model, occlusion, clouds, selected, args.batch_size, seed, True,
            args.device, capture_points=args.with_chamfer
        )
        if args.with_chamfer:
            fragment_points.append(captured)
        hashes["fragments"].append(fragment_hash)
        similarity = (fragments @ clean.T).cpu().double().numpy()
        statistics = per_draw_statistics(similarity)
        row = {
            "seed": int(seed),
            "rho": statistics["fragment_clean_distance"] / denominator,
            "rho_numerator": statistics["fragment_clean_distance"],
            "z": statistics["z"], "top1": statistics["top1"],
        }
        draws.append(row)
        print(f"Seed {seed}: rho={row['rho']:.6f}, z={row['z']:.6f}, Top-1={row['top1']:.4f}",
              flush=True)
        del fragments

    metrics = {
        "checkpoint": checkpoint_info,
        **summarize_draws(draws, denominator),
        "protocol": {
            "selection_seed": args.selection_seed,
            "selected_indices": selected.tolist(),
            "num_objects": args.num_objects,
            "batch_size": args.batch_size,
            "clean_torch_seed": args.clean_seed,
            "occlusion_torch_seeds": list(args.draw_seeds),
            "removal_ratio_interval": [0.25, 0.40],
            "occlusion_probability_at_evaluation": 1.0,
            "min_keep": 1100,
            "points_per_raw_cloud": int(clouds.shape[1]),
            "preprocessing": ["subsample", "center", "unit_sphere"],
            "points_per_encoder_input": 1024,
            "draw_standard_deviation": "population (ddof=0), not standard error",
            "distractor_standard_deviation": "population (ddof=0)",
        },
        "representation": "L2-normalized concatenated max/mean-pooled student tokens",
        "processed_input_hashes": hashes,
        "execution": {
            "torch": torch.__version__, "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
            "feature_precision": "float32; no autocast",
        },
        "per_draw": draws,
    }
    if args.with_chamfer:
        # Geometry uses the saved coordinates only; release the encoder first.
        del model, clean
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        geometry, numerators = compute_geometry(clean_points, fragment_points, args.device)
        metrics["raw_geometry"] = geometry
        for row, numerator in zip(draws, numerators):
            row["raw_geometry_rho"] = numerator / geometry["mean_clean_clean_chamfer"]
            row["fragment_whole_chamfer"] = numerator
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partial_observation_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "partial_observation_draws.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(draws[0]))
        writer.writeheader()
        writer.writerows(draws)
    print("\nMean ± population SD across draws:")
    for name in ("rho", "z", "top1"):
        print(f"{name}: {metrics[name]:.6f} ± {metrics[name + '_std']:.6f}")
    if args.with_chamfer:
        print(f"Raw-geometry rho: {geometry['rho']:.6f} ± {geometry['rho_std']:.6f}")
    print(f"Saved results to {args.output_dir}")


def self_test(args):
    """Check formulas without research assets; test Chamfer on CPU if torch exists."""
    similarity = np.array([[.9, .1, .3], [.2, .8, .4], [.2, .6, .9]])
    stats = per_draw_statistics(similarity)
    np.testing.assert_allclose(stats["z_values"], [7., 5., 2.5])
    np.testing.assert_allclose(stats["fragment_clean_distance"], 2 / 15)
    assert stats["top1"] == 1.0
    summary = summarize_draws([
        {"rho_numerator": .2, "z": 2., "top1": .5},
        {"rho_numerator": .4, "z": 4., "top1": 1.},
    ], .5)
    np.testing.assert_allclose([summary[k] for k in ("rho", "z", "top1")], [.6, 3., .75])
    np.testing.assert_allclose([summary[k] for k in ("rho_std", "z_std", "top1_std")], [.2, 1., .25])
    np.testing.assert_array_equal(select_objects(500, 200, 0),
                                  np.random.RandomState(0).choice(500, 200, replace=False))
    geometry = summarize_geometry([1., 3.], 4.)
    np.testing.assert_allclose([geometry["rho"], geometry["rho_std"]], [.5, .25])
    result = {"status": "passed", "synthetic_only": True}
    try:
        import torch
    except ImportError:
        result["chamfer_distance_test"] = "skipped: PyTorch is not installed"
    else:
        # Unequal point counts exercise both directions and their 0.5 weighting.
        a = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]])
        b = torch.tensor([[[0., 0., 0.]]])
        np.testing.assert_allclose(chamfer_batch(a, b).numpy(), [.5])
        np.testing.assert_allclose(chamfer_batch(b, a).numpy(), [.5])
        clean = torch.tensor([[[0., 0., 0.]], [[2., 0., 0.]], [[4., 0., 0.]]])
        fragments = torch.tensor([[[0., 0., 0.]], [[1., 0., 0.]], [[3., 0., 0.]]])
        geometric, numerators = compute_geometry(clean, [fragments, clean.clone()], "cpu")
        np.testing.assert_allclose(numerators, [2. / 3., 0.])
        np.testing.assert_allclose(geometric["mean_clean_clean_chamfer"], 8. / 3.)
        np.testing.assert_allclose([geometric["rho"], geometric["rho_std"]], [.125, .125])
        result["chamfer_distance_test"] = "passed on CPU"
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "self_test.json").write_text(json.dumps(result) + "\n")
    print(json.dumps(result))


def build_parser():
    parser = argparse.ArgumentParser(description="Compute rho, z and Top-1 for one checkpoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluation = subparsers.add_parser("evaluate")
    source = evaluation.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    source.add_argument("--run-dir", type=Path, help="Use the latest epoch-499 checkpoint in this run")
    evaluation.add_argument("--data-npz", type=Path, default=Path("data/ShapeNet55/shapenet_test.npz"))
    evaluation.add_argument("--num-objects", type=int, default=DEFAULT_NUM_OBJECTS)
    evaluation.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluation.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    evaluation.add_argument("--clean-seed", type=int, default=DEFAULT_CLEAN_SEED)
    evaluation.add_argument("--draw-seeds", type=parse_draw_seeds, default=DEFAULT_DRAW_SEEDS)
    evaluation.add_argument("--device", default=default_device())
    evaluation.add_argument("--allow-protocol-change", action="store_true")
    evaluation.add_argument("--with-chamfer", action="store_true",
                            help="Also compute matched raw-geometry rho using symmetric unsquared Chamfer")
    evaluation.add_argument("--output-dir", type=Path, default=Path("outputs/partial_observation"))
    evaluation.set_defaults(function=evaluate)
    test = subparsers.add_parser("self-test")
    test.add_argument("--output-dir", type=Path)
    test.set_defaults(function=self_test)
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

