"""Evaluate a frozen pretrained encoder using a linear SVM."""


import argparse, importlib, inspect, os, tempfile, warnings
import yaml, numpy as np, torch
warnings.filterwarnings("ignore", message="Found keys that are not in the model")
from sklearn.svm import SVC
from pointjepa.models.point_jepa import PointJepa


def build_datamodule(config_path):
    d = yaml.safe_load(open(config_path))["data"]
    mod, cls = d["class_path"].rsplit(".", 1)
    DM = getattr(importlib.import_module(mod), cls)
    kw = dict(d.get("init_args", {}))
    if "drop_last" in inspect.signature(DM.__init__).parameters:
        kw["drop_last"] = False           # keep the final partial batch
    dm = DM(**kw)
    dm.setup("fit")
    return dm


def load_encoder(ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu")
    allowed = set(inspect.signature(PointJepa.__init__).parameters)
    hp = {k: v for k, v in blob.get("hyper_parameters", {}).items() if k in allowed}
    blob["hyper_parameters"] = hp
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as fh:
        tmp = fh.name
    try:
        torch.save(blob, tmp)
        model = PointJepa.load_from_checkpoint(tmp, svm_validation={}, strict=False)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    missing = [k for k in model.state_dict()
               if k.startswith(("student.", "tokenizer.", "positional_encoding."))
               and k not in blob["state_dict"]]
    if missing:
        raise RuntimeError(f"encoder weights missing from checkpoint: {missing[:5]}")
    return model.cuda().eval()


@torch.no_grad()
def features(model, loader):
    xs, ys = [], []
    for points, labels in loader:
        points = model.val_transformations(points.cuda())
        embeddings, centers = model.tokenizer(points)
        h = model.student(embeddings, model.positional_encoding(centers)).last_hidden_state
        xs.append(torch.cat([h.max(1).values, h.mean(1)], -1).cpu().float())
        ys.append(labels.cpu())
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--fixed-c", type=float, default=0.012)
    args = p.parse_args()

    dm = build_datamodule(args.config)
    model = load_encoder(args.ckpt)
    print(f"train {len(dm.train_dataloader().dataset)}  "
          f"test {len(dm.val_dataloader().dataset)}", flush=True)

    fixed = []
    for draw in range(args.draws):
        torch.manual_seed(draw)                    # reseed: patch sampling consumes RNG
        Xtr, ytr = features(model, dm.train_dataloader())
        Xva, yva = features(model, dm.val_dataloader())
        A, B = Xtr.astype(np.float64), Xva.astype(np.float64)
        Ktr, Kva = A @ A.T, B @ A.T
        fixed.append(100 * SVC(C=args.fixed_c, kernel="precomputed")
                     .fit(Ktr, ytr).score(Kva, yva))
        print(f"draw {draw}: C={args.fixed_c} {fixed[-1]:.2f}", flush=True)

    f = np.array(fixed)
    print(f"\nfixed C={args.fixed_c} : {f.mean():6.2f} +/- {f.std():.2f}")


if __name__ == "__main__":
    main()
