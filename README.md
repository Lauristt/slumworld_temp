# Slum Detection via Computer Vision

A deep learning pipeline for binary semantic segmentation of informal settlements (slums) from panchromatic and multispectral satellite imagery. Supports a wide model zoo from vanilla U-Net to transformer-based architectures, adversarial domain adaptation for cross-city generalization, and few-shot target fine-tuning with a satellite-pretrained vision foundation model (DINOv3).

---

## Repository Structure

```
.
├── src/                                # Core library (import as a package)
│   ├── model.py                        # Model zoo — all architectures registered here
│   ├── trainer.py                      # PyTorch Lightning training wrapper + callbacks
│   ├── SatelliteDataset.py             # Dataset classes and LightningDataModule
│   ├── base_tiler.py                   # Tile / reconstruct large satellite images
│   ├── cnn_tiler.py                    # Train/val/test splits and k-fold tiling logic
│   ├── overlap_tiler.py                # 50% overlap inference tiling
│   ├── predictor.py                    # Inference and evaluation
│   ├── inspector.py                    # Visualization and shapefile generation
│   ├── custom_transformations.py       # Augmentation primitives
│   ├── transforms_loader.py            # Augmentation pipeline loader
│   ├── utilities.py                    # Optimizers, CRF, normalization helpers
│   ├── pvtv2.py                        # Pyramid Vision Transformer v2 backbone
│   └── registry.py                     # Decorator-based model registry
├── runners/                            # Entry-point scripts — run these directly
│   ├── train.py                        # Train a model
│   ├── evaluate.py                     # Evaluate on a labeled test set
│   ├── inference.py                    # Run inference on new (unlabeled) imagery
│   ├── tile_standard.py                # Tile an image into fixed-size patches
│   ├── finetune_dummy_tiler.py         # Integer-based tile split for low-shot fine-tuning
│   ├── tile_kfold.py                   # K-fold cross-validation tiling
│   ├── tile_with_overlap.py            # 50% overlap tiling for inference
│   ├── reconstruct_map_from_tiles.py   # Stitch tile predictions into a full map
│   ├── generate_shapefile.py           # Vectorize binary prediction map to shapefile
│   ├── preprocess_features.py          # Pre-compute DINOv3 features offline
│   ├── check_resolution.py             # Verify tile/image resolution consistency
│   └── crf_postprocess_pipeline.py     # Dense CRF post-processing
└── Configurations_yamls/               # YAML experiment configs (one per run)
    ├── tile_tilingMS.yml               # Example tiling config (multispectral)
    ├── test_low_show_test_ms2016_md2001.yml
    └── train_unet_v2_dinov3_low_shot_finetune_test_vanilla_ms2016_to_md2001.yml
```

---

## How to Run an Experiment

Every runner script takes a single YAML config file via `-c`. All parameters — paths, model selection, hyperparameters, compute — live in that file.

```bash
python runners/<script>.py -c Configurations_yamls/<config>.yml
```

Use `-h` to see what a script expects:
```bash
python runners/train.py -h
```

### Typical Workflow

**Step 1 — Tile the satellite image**
```bash
python runners/tile_standard.py -c Configurations_yamls/tile_tilingMS.yml
```
Produces a `dataset.csv` (tile index + train/val/test split + class balance stats) and a `dataset.json` (normalization statistics). Both are needed for training.

**Step 2 — (Optional) Pre-compute DINOv3 features**

Required only if using `unet_v2_dinov3` or `dinov3_*` models.
```bash
python runners/preprocess_features.py -c Configurations_yamls/<your_config>.yml
```
Features are saved to `dinov3_integration.features_path` and loaded lazily during training.

**Step 3 — Train**
```bash
python runners/train.py -c Configurations_yamls/train_unet_v2_dinov3_low_shot_finetune_test_vanilla_ms2016_to_md2001.yml
```
Checkpoints are saved to `paths.output_dir`. Filename encodes key metrics, e.g.:
```
unet_v2-pan-BinaryCrossEntropyWithLogits-AdamW--L2_0.002-Seed_123497-SingleFold-epoch=183-val_loss=0.2628-val_acc=0.9717-val_f1=0.8461.ckpt
```

**Step 4 — Evaluate on the test set**
```bash
python runners/evaluate.py -c Configurations_yamls/<your_config>.yml
```

**Step 5 — Reconstruct and export**
```bash
python runners/reconstruct_map_from_tiles.py -c Configurations_yamls/<your_config>.yml
python runners/generate_shapefile.py -c Configurations_yamls/<your_config>.yml
```

---

## YAML Configuration Reference

Each YAML has six top-level sections. Below are the fields you will most commonly edit.

### `model_parameters`
Only used by the vanilla `unet` model (trained from scratch). Ignored for pretrained variants.

```yaml
model_parameters:
    in_channels: 3        # 3 for both PAN and MS (tiler normalizes to 3-channel)
    out_channels: 1       # always 1 (binary segmentation)
    features: [64, 128, 256, 512]
    dropout_prob: 0.05
    dropout_2d_prob: 0.05
```

### `run_type`
Controls what the training script does.

```yaml
run_type:
    pretrained_model: "unet_v2_dinov3"   # model ID — see Model Zoo table below
    from_checkpoint: true                 # resume from paths.checkpoint_file
    foldID: -1                            # -1 = standard, 0..k = k-fold
    training_mode: 'train_all'            # 'train_all' | 'freeze' | 'overfit' | 'lr_find'
    domain_adaptation: false              # enable adversarial domain adaptation
    self_supervised_pretraining: null     # 'ssp' | 'assp' | null
```

### `training_parameters`

```yaml
training_parameters:
    image_type: 'pan'           # 'pan' (panchromatic) or 'mul' (multispectral)
    tile_size: 512
    num_epochs: 5
    batch_size: 12
    criterion: 'BinaryCrossEntropyWithLogits'   # loss function (see Losses section)
    optimizer: 'AdamW'
    threshold: 0.5              # probability cutoff for binary prediction

    # Transformer models (unet_v2, dinov3 series): use decoupled LRs
    decoupled_learning_rate: true
    learning_rate: 3e-4         # decoder LR
    encoder_learning_rate: 2e-5 # encoder LR (prevents catastrophic forgetting)

    # Low-shot fine-tuning: mix source and target tiles in the same batch
    target_finetuning:
        enabled: true
        csv_file: '/path/to/target/dataset.csv'
        mix_mode: 'in_batch'
        samples_per_batch: 6    # target tiles per batch; source = batch_size - 6

    val_metric:
        metric: 'val_f1'
        mode: 'max'             # save checkpoint when val_f1 improves
```

### `paths`

```yaml
paths:
    training_csv: "/path/to/dataset.csv"          # produced by tile_standard.py
    normalization_file: "/path/to/dataset.json"   # produced by tile_standard.py
    output_dir: "/path/to/save/checkpoints"
    checkpoint_file: "/path/to/resume.ckpt"       # only if from_checkpoint: true
    dinov3_repo_path: "/path/to/dinov3"           # local DINOv3 repo clone
    inference_dir: "/path/to/test/dataset.csv"    # optional: test set
    domain_adaptation_csv: null                   # optional: unlabeled target domain
```

### `compute_parameters`

```yaml
compute_parameters:
    precision: 32             # 16 | 32 | 64
    gpus: [1]                 # [0] for first GPU, [0,1] for two GPUs
    strategy: null            # null = single GPU, 'dp' = DataParallel, 'ddp' = DistributedDataParallel
    auto_select_gpus: true    # auto-pick least-loaded GPU
    n_nodes: 1
    n_workers: 10             # dataloader workers (match to CPU core count)
```

### `dinov3_integration`
Required when using any `dinov3_*` or `unet_v2_dinov3` model.

```yaml
dinov3_integration:
    enabled: true
    model_key: 'dinov3_sat_large'               # satellite-pretrained ViT-L/16
    features_path: "/path/to/dino_features"     # where pre-computed features are stored
```

---

## Model Zoo

Select the model by setting `run_type.pretrained_model` in the YAML.

| Model ID | Architecture | Encoder | Notes |
|---|---|---|---|
| `unet` | U-Net | Scratch | Set via `model_parameters`; no pretrained weights |
| `unet_vgg11` / `unet_vgg11_bn` | U-Net | VGG-11 (ImageNet) | |
| `unet_vgg13_bn` / `unet_vgg16_bn` / `unet_vgg19_bn` | U-Net | VGG-13/16/19 | |
| `unet_resnet18` / `unet_resnet34` / `unet_resnet50` | U-Net | ResNet (ImageNet) | |
| `unet_v2` | PVT-v2-B2 + CBAM + SDI | PVT-v2-B2 | Attention at all scales |
| `unet_v2_dinov3` | PVT-v2-B2 + DINOv3 fusion | PVT-v2-B2 + ViT-L/16 | Requires `dinov3_integration` |
| `unet_v2_dinov3_att` | `unet_v2_dinov3` + attention gate | PVT-v2-B2 + ViT-L/16 | |
| `dinov3` / `dinov3_sat_large` | ViT-L/16 + seg head | DINOv3 SAT (1024-dim) | Satellite-pretrained |
| `dinov3_small` | ViT-S/16 + seg head | DINOv3 (384-dim) | |
| `dinov3_base` | ViT-B/16 + seg head | DINOv3 (768-dim) | |
| `segformer` | SegFormer | HuggingFace | |
| `swinformer` | Swin Transformer v2 | HuggingFace | |

---

## Training Details

### Losses (`training_parameters.criterion`)

| Value | Description |
|---|---|
| `BinaryCrossEntropyWithLogits` | Standard BCE; works well with balanced batches |
| `DiceLoss` | Optimizes overlap; robust to class imbalance |
| `BinaryFocalLoss` | Down-weights easy negatives; useful for sparse slum pixels |
| `BinaryLovaszLoss` | Surrogate for IoU |
| `BinarySoftF1Loss` | Directly optimizes F1 |

### Optimizers (`training_parameters.optimizer`)

`Adam` | `AdamW` | `SGD` | `AdaBound` | `AdaBoundW` | `Yogi`

### Schedulers (`training_parameters.scheduler.name`)

`CosineAnnealingWarmRestarts` | `ReduceLROnPlateau` | `ExponentialLR` | `OneCycleLR` | `null`

> **Note:** The scheduler is automatically disabled when `target_finetuning.enabled: true`.

---

## Domain Adaptation

Set `run_type.domain_adaptation: true` and provide a `paths.domain_adaptation_csv` pointing to unlabeled tiles from the target city. A Gradient Reversal Layer (GRL) trains the encoder to produce domain-invariant features. The domain loss weight is controlled by `training_parameters.domain_loss_scaling_factor`.

---

## Low-Shot Fine-Tuning

Fine-tune a pretrained source-city model on a small labeled target-city dataset:

1. Set `run_type.from_checkpoint: true` and point `paths.checkpoint_file` to the source model.
2. Set `training_parameters.target_finetuning.enabled: true`.
3. Provide `target_finetuning.csv_file` with the target city's `dataset.csv`.
4. Set `samples_per_batch` to the number of target tiles per batch (remaining slots are source tiles).

The LR scheduler is disabled automatically. Use a small `num_epochs` (e.g. 5–10).

---

## Tiling

`tile_standard.py` splits a full satellite image into 512×512 patches and assigns each tile to train/val/test. Two split modes are available:

- **Integer-based** (current default): specify exact tile counts, e.g. `train_tiles: 800`, `val_tiles: 100`.
- **Percentage-based** (legacy): specify fractions, e.g. `train_split: 0.8`.

The output `dataset.csv` has one row per tile with columns: `tile_path`, `label_path`, `split`, `slum_ratio`. Pass this CSV to `training_parameters.training_csv` in the training config.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `val_loss` | Validation loss (configured criterion) |
| `val_acc` | Binary pixel accuracy |
| `val_f1` | Binary F1 score |
| `val_iou` | Jaccard Index (IoU) |

---

## Output

| Output | Format | Script |
|---|---|---|
| Binary probability map | PNG | `inference.py` |
| Colorized overlay | PNG (Red=predicted, Blue=GT, Magenta=overlap) | `runners/overlay.py` |
| Full reconstructed map | GeoTIFF | `reconstruct_map_from_tiles.py` |
| GIS shapefile | `.shp` via `rasterio`/`geopandas` | `generate_shapefile.py` |

CRS presets in `generate_shapefile.py`: EPSG:32643 (Mumbai/PCMC), EPSG:32630 (Ouagadougou/Bobo), EPSG:32736 (Kigali).

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Version | Role |
|---|---|---|
| `torch` | — | Core DL framework |
| `pytorch-lightning` | 1.9.0 | Training loop, multi-GPU, SLURM |
| `timm` | ≥1.0.19 | PVT-v2-B2 backbone |
| `transformers` | 4.28.0 | SegFormer, Swin Transformer |
| `rasterio` | 1.3.11 | Geospatial I/O |
| `geopandas` | 0.13.2 | Shapefile generation |
| `pydensecrf` | — | CRF post-processing |
| `pytorch-toolbelt` | 0.8.0 | Dice, Focal, Lovász, Soft-F1 losses |
| `torchmetrics` | — | F1, IoU, accuracy metrics |
| `scikit-image` | 0.21.0 | Sobel edge detection (SSP) |
| `opencv-python` | 4.11.0 | Image I/O and processing |
