# Beyond a Single Ordering: Multi-Axis Serialization under Occlusion for Point-JEPA

Official code for our WACV 2027 paper.

This repository builds on [Point-JEPA](https://github.com/Ayumu-J-S/Point-JEPA)
and introduces multi-axis serialization and ball occlusion.

1. **Multi-axis serialization** (`pointjepa/models/point_jepa.py`, `_forward_pj_multi`):
   the greedy token sequencer is replaced by `num_orderings` directional sorts of
   the patch centers (coordinate axes by default, `ordering_basis: pca` for the
   principal-axes variant, `ordering_axes: "xz"` to select axes explicitly).
   All orderings are applied to the same cloud in the same step and folded into
   the batch, with a single loss averaged over the resulting predictions.
2. **Ball occlusion** (`pointjepa/utils/transforms.py`, `PointcloudOcclusion`):
   a spatial crop that removes points inside a randomly located ball before tokenization.

A single tokenizer grouping call feeds ordering, masking, and embedding, so all
views share the same patches. `masking_mode: pj_baseline` reproduces Point-JEPA's
min-coordinate-sum + nearest-neighbor sequencing inside this codebase.

## Installation

### 1. Dependencies

Verified environment on Narval (Compute Canada):

- Python 3.10.13
- PyTorch 2.1.2 and torchvision 0.16.2
- PyTorch3D 0.7.5
- PyTorch Lightning 1.9.5
- CUDA 12.2 and cuDNN 8.9.5
- GCC/G++ 12.3.1

We also provide the original Point-JEPA installation script,
which uses different dependency versions:

```bash
bash install_py.sh
```

### 2. Datasets

See [DATASETS.md](DATASETS.md) for download instructions.

### 3. Check (optional)

```bash
python -m pointjepa.datasets.process.check
```

## Model Zoo

| Type | Dataset | Protocol | Evaluation | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Multi-axis + Occl. pre-trained | ShapeNet | – | – | [config](configs/pretraining/occl_xyz.yaml) | [checkpoint](https://drive.google.com/file/d/1KORh8U288Fc--ettbqV3FQxQkQrn_Tj5/view?usp=sharing) |
| SVM Linear | ModelNet40 / OBJ-BG | – | **93.9&pm;0.2** / **85.5&pm;0.4** | – | – |
| Classification fine-tuned | ModelNet40 | A | **93.9&pm;0.2** / **94.2&pm;0.2** (OA / Voting) | [config](configs/classification/modelnet40.yaml) | [checkpoint](https://drive.google.com/file/d/1xmGKQTRaDMX3PpK7x3mJumWKWF967EyN/view?usp=sharing) |
| Classification fine-tuned | ModelNet40 | B | **93.8&pm;0.2** / **94.0&pm;0.2** (OA / Voting) | [config](configs/classification/modelnet40_occl.yaml) | [checkpoint](https://drive.google.com/file/d/1AfTdkMLF47e1M4SAQiT6Mh35NmlR4fT4/view?usp=sharing) |
| Classification fine-tuned | ScanObjectNN OBJ-BG | A | **92.8&pm;0.2** (OA) | [config](configs/classification/scanobjectnn.yaml) + [OBJ-BG split](configs/classification/_scanobjectnn_obj_bg.yaml) | [checkpoint](https://drive.google.com/file/d/1Oo477k_fvvWqJYU2hDWRLGEnZY9C_pOC/view?usp=sharing) |
| Classification fine-tuned | ScanObjectNN OBJ-BG | B | **93.1&pm;0.3** (OA) | [config](configs/classification/scanobjectnn_occl.yaml) + [OBJ-BG split](configs/classification/_scanobjectnn_obj_bg.yaml) | [checkpoint](https://drive.google.com/file/d/1PGmKl7el4hCjhGPkmxQChRR2r3xJYKwi/view?usp=sharing) |
| Classification fine-tuned | ScanObjectNN OBJ-ONLY | A | **90.4&pm;0.3** (OA) | [config](configs/classification/scanobjectnn.yaml) + [OBJ-ONLY split](configs/classification/_scanobjectnn_obj_only.yaml) | [checkpoint](https://drive.google.com/file/d/1GJjIOVppVtsf75IWSdCWD_NnHUaK1h_k/view?usp=sharing) |
| Classification fine-tuned | ScanObjectNN OBJ-ONLY | B | **90.0&pm;0.3** (OA) | [config](configs/classification/scanobjectnn_occl.yaml) + [OBJ-ONLY split](configs/classification/_scanobjectnn_obj_only.yaml) | [checkpoint](https://drive.google.com/file/d/1VcHm7uk40b9QowLX-gnvlXPQol42UNeg/view?usp=sharing) |
| Classification fine-tuned | ScanObjectNN PB-T50-RS | A | **87.5&pm;0.2** (OA) | [config](configs/classification/scanobjectnn.yaml) | [checkpoint](https://drive.google.com/file/d/1gG6-_uSeDogMmzVkC0qMrwjOKjOYDNw_/view?usp=sharing) |
| Classification fine-tuned | ScanObjectNN PB-T50-RS | B | **89.0&pm;0.2** (OA) | [config](configs/classification/scanobjectnn_occl.yaml) | [checkpoint](https://drive.google.com/file/d/1sS8C6cL2o0FKdXnT_lYE-RHsYoB7NuX-/view?usp=sharing) |
| Part segmentation fine-tuned | ShapeNetPart | A | **84.0&pm;0.1** / **85.9&pm;0.1** (Cls. / Inst. mIoU) | [config](configs/segmentation/shapenetpart.yaml) | [checkpoint](https://drive.google.com/file/d/1hAVuC9V2bYZXaEcOfpBB3sOKEiI0Lfu-/view?usp=sharing) |

Protocol A follows the Point-JEPA downstream recipe. Protocol B adds
ball occlusion during downstream training; test inputs are unchanged.

Results are reported as mean ± standard deviation using a single
pre-trained checkpoint. The linked fine-tuned checkpoints are selected
from the 10 downstream runs.


## Reproducing the results from the paper

Download the pre-trained checkpoint into `checkpoints/`, or pre-train it yourself.

### Pre-training (ShapeNet, 500 epochs)

```bash
python -m pointjepa fit \
    -c configs/pretraining/shapenet.yaml \
    -c configs/pretraining/occl_xyz.yaml
```

Other pre-training configurations in the paper: `baseline` / `baseline_occl`
(Point-JEPA's sequencer, clean / occluded), `clean_xyz` (three coordinate
orderings, no occlusion), `clean_pca` / `occl_pca` (principal-axes orderings),
`occl_x` / `occl_xy` / `occl_xz` (axis-count ablation), and `occl_mild_xyz`
(milder occlusion range). Replace the second configuration above with the
corresponding YAML file.

### Classification fine-tuning (10 seeds)

```bash
CKPT=checkpoints/multiaxis_occl_pretrain_shapenet.ckpt

for i in $(seq 1 10)
do
    # ScanObjectNN PB-T50-RS, Protocol A
    # Protocol B: use scanobjectnn_occl.yaml
    python -m pointjepa.tasks.classification fit \
        --config configs/classification/scanobjectnn.yaml \
        --model.pretrained_ckpt_path "$CKPT" \
        --seed_everything "$i"
done
```

- OBJ-BG / OBJ-ONLY: add `--config configs/classification/_scanobjectnn_obj_bg.yaml`
  (or `_scanobjectnn_obj_only.yaml`) after the first config.
- ModelNet40: `modelnet40.yaml` (Protocol A) or `modelnet40_occl.yaml`
  (Protocol B). Both save the checkpoint with the highest non-voting validation
  accuracy for voting evaluation.

Voting on ModelNet40 uses the selected checkpoint and reports the maximum
accuracy over 300 voting iterations, with 10 votes per iteration:

```bash
python -m pointjepa.eval.voting \
    --config configs/classification/modelnet40.yaml \
    --finetuned_ckpt_path checkpoints/modelnet40_protocolA.ckpt \
    --seed_everything 0 \
    --vote_num 10 --vote_iterations 300
```

For Protocol B, use `modelnet40_occl.yaml` and the corresponding fine-tuned checkpoint.

### Few-shot classification (ModelNet40, 10 folds)

Training uses seed 0 for every fold. Results are reported as mean ±
sample standard deviation of the best-epoch accuracy across folds 0–9.

```bash
CKPT=checkpoints/multiaxis_occl_pretrain_shapenet.ckpt

for fold in $(seq 0 9)
do
    python -m pointjepa.tasks.classification fit \
        --config configs/classification/modelnet_fewshot.yaml \
        --model.pretrained_ckpt_path "$CKPT" \
        --seed_everything 0 \
        --data.way 5 --data.shot 10 --data.fold "$fold"
done
```

Protocol B: `modelnet_fewshot_occl.yaml`.

### Part segmentation (ShapeNetPart, 10 seeds)

```bash
CKPT=checkpoints/multiaxis_occl_pretrain_shapenet.ckpt

for i in $(seq 1 10)
do
    python -m pointjepa.tasks.part_segmentation fit \
        --config configs/segmentation/shapenetpart.yaml \
        --model.pretrained_ckpt_path "$CKPT" \
        --seed_everything "$i"
done
```

### Frozen linear SVM (evaluation seeds 0–9)

```bash
CKPT=checkpoints/multiaxis_occl_pretrain_shapenet.ckpt

python -m eval.linear_probe \
    --ckpt "$CKPT" \
    --config configs/classification/modelnet40.yaml \
    --draws 10 --fixed-c 0.012

python -m eval.linear_probe \
    --ckpt "$CKPT" \
    --config configs/classification/_scanobjectnn_obj_bg.yaml \
    --draws 10 --fixed-c 0.012
```

Features concatenate max- and mean-pooled frozen encoder tokens.
The script reports mean accuracy ± population standard deviation.
The final partial batch is retained.

### Partial-observation metrics

Compute rho, z and Top-1 for one compatible pre-trained Point-JEPA checkpoint:

```bash
python scripts/partial_observation_metrics.py evaluate \
    --checkpoint checkpoints/multiaxis_occl_pretrain_shapenet.ckpt \
    --output-dir outputs/multiaxis_occl
```

Add `--with-chamfer` to also compute the raw-geometry reference.

## Notes

Adaptive PE learns the shared positional map during pretraining.
Under the mixed-precision training configuration used in the paper,
Reference-style PE leaves this map unchanged because of the placement
of the teacher-side positional-encoding call inside `torch.no_grad()`.
All headline results use Adaptive PE.

To select Reference-style PE, append its configuration last:

```bash
python -m pointjepa fit \
    -c configs/pretraining/shapenet.yaml \
    -c configs/pretraining/occl_xyz.yaml \
    -c configs/pretraining/reference_style_pe.yaml
```

## Acknowledgements

This codebase is built upon [Point-JEPA](https://github.com/Ayumu-J-S/Point-JEPA),
which itself builds on [Point2Vec](https://github.com/kabouzeid/point2vec).
We thank the authors for releasing their code.

## Citation

The citation will be added after publication.

## License

MIT, inherited from Point-JEPA / Point2Vec (see [LICENSE](LICENSE)).
