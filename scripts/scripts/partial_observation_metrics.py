
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
