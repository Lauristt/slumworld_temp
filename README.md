# SlumWorld — Informal Settlement Detection in Satellite Imagery

A deep learning pipeline for binary semantic segmentation of informal settlements (slums) from panchromatic and multispectral satellite imagery. The system supports a wide model zoo from vanilla U-Net to transformer-based architectures, adversarial domain adaptation for cross-city generalization, and few-shot target fine-tuning with a satellite-pretrained vision foundation model (DINOv3).

---

## Problem Statement

Mapping informal urban settlements from satellite imagery is a low-resource binary segmentation problem: slum pixels are rare relative to background, labels are expensive to produce, and imagery characteristics vary substantially across cities and acquisition years. This codebase addresses all three challenges through a modular architecture, domain-adaptive training, and self-supervised feature integration.

---

## Repository Structure

```
slumworldML/
├── src/                        # Core library
│   ├── model.py                # Model zoo (20+ architectures)
│   ├── trainer.py              # PyTorch Lightning training wrapper
│   ├── SatelliteDataset.py     # Dataset classes and data module
│   ├── base_tiler.py           # Tile / reconstruct large satellite images
│   ├── cnn_tiler.py            # Train/val/test splits and k-fold tiling
│   ├── overlap_tiler.py        # 50% overlap inference tiling
│   ├── predictor.py            # Inference and evaluation
│   ├── inspector.py            # Visualization and shapefile generation
│   ├── custom_transformations.py  # Augmentation primitives
│   ├── transforms_loader.py    # Augmentation pipeline loader
│   ├── utilities.py            # Optimizers, CRF, normalization helpers
│   ├── pvtv2.py                # Pyramid Vision Transformer v2 backbone
│   └── registry.py             # Decorator-based model registry
├── runners/                    # Executable scripts
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation on test set
│   ├── inference.py            # Inference on new imagery
│   ├── tile_standard.py        # Standard tiling
│   ├── tile_kfold.py           # K-fold tiling
│   ├── tile_with_overlap.py    # Overlap tiling for inference
│   ├── reconstruct_map_from_tiles.py
│   ├── generate_shapefile.py
│   ├── crf_postprocess_pipeline.py
│   ├── change_detection.py
│   └── ...                     # Additional utilities (28 scripts total)
└── Configurations_yamls/       # YAML experiment configurations
```

---

## Modeling Approach

### Model Zoo

All models are registered via a decorator pattern and selected by name in the YAML configuration. Every architecture produces a binary segmentation map (single-channel logit output) and optionally a domain classifier logit for adversarial training.

| Model ID | Architecture | Encoder | Notes |
|---|---|---|---|
| `unet` | U-Net | Scratch | Features [64,128,256,512], spatial & channel dropout |
| `unet_vgg11` / `_bn` | U-Net | VGG-11 (ImageNet) | Batch-norm variant |
| `unet_vgg13_bn` / `vgg16_bn` / `vgg19_bn` | U-Net | VGG-13/16/19 (ImageNet) | |
| `unet_resnet18` / `34` / `50` | U-Net | ResNet (ImageNet) | |
| `unet_v2` | PVT-v2-B2 + CBAM + SDI | PVT-v2-B2 | Attention gating at all scales |
| `unet_v2_dinov3` | PVT-v2-B2 + DINOv3 fusion | PVT-v2-B2 + ViT-L/16 | DINOv3 features fused at bottleneck |
| `unet_v2_dinov3_att` | `unet_v2_dinov3` + attention | PVT-v2-B2 + ViT-L/16 | Attention gate on DINO features |
| `dinov3` / `dinov3_sat_large` | ViT-L/16 + segmentation head | DINOv3 SAT (1024-dim) | Satellite-pretrained foundation model |
| `dinov3_small` | ViT-S/16 + segmentation head | DINOv3 (384-dim) | |
| `dinov3_base` | ViT-B/16 + segmentation head | DINOv3 (768-dim) | |
| `segformer` | SegFormer | HuggingFace Transformers | |
| `swinformer` | Swin Transformer v2 | HuggingFace Transformers | |

### UNetV2 Architecture

The primary production model (`unet_v2`) uses a **PVT-v2-B2** hierarchical vision transformer as the encoder, producing four feature maps at scales {1/4, 1/8, 1/16, 1/32}. Each scale is refined with **CBAM** (Convolutional Block Attention Module) consisting of:

- **Channel Attention**: parallel average- and max-pool paths compressed by ratio=16, fused via sigmoid gating
- **Spatial Attention**: 7×7 depthwise convolution over concatenated channel-pooled maps, sigmoid output

The refined multi-scale features are unified via **SDI** (Scale-wise Decorrelated Interaction), which multiplies all scale features projected to a common channel dimension (`channel=32`) at each anchor resolution via learned 3×3 convolutions. A progressive decoder reconstructs the segmentation map through transposed convolutions with residual additions.

### DINOv3 Feature Fusion (`unet_v2_dinov3`)

A satellite-image-pretrained DINOv3 **ViT-L/16** model (1024-dim patch embeddings, patch size 16) produces dense feature maps that are pre-computed offline and stored on disk for efficient loading. During training, these features are aligned to the PVT bottleneck spatial resolution via bilinear interpolation and concatenated (channel dimension 512 + 1024 = 1536), then projected back to 512 channels via a 1×1 bottleneck convolution before the CBAM and SDI modules.

This fusion allows the segmentation decoder to leverage large-scale visual semantics from a self-supervised foundation model while keeping the spatial fine-grained reasoning within the PVT encoder.

---

## Domain Adaptation

For cross-city generalization, the model supports **adversarial domain adaptation** via a **Gradient Reversal Layer** (GRL). An auxiliary `FCNHead` classifier is attached to the bottleneck features and trained to distinguish source from target domain images. The GRL negates gradients flowing back to the encoder, encouraging domain-invariant representations. Domain loss weight is controlled by a configurable scaling factor.

---

## Low-Shot Target Fine-Tuning

The `target_finetuning` module supports adaptation to a new city with minimal labeled data. Source and target batches are mixed **in-batch** at a configurable ratio (e.g., 6 source + 6 target per batch of 12). The learning rate scheduler is disabled during fine-tuning to avoid decay on the sparse target signal. This mode is configured independently from the adversarial domain adaptation path.

---

## Training

### Losses

The following losses are available and may be combined:

| Loss | Purpose |
|---|---|
| `BinaryCrossEntropyWithLogits` | Baseline, works well with class-balanced batches |
| `DiceLoss` | Directly optimizes overlap; robust to class imbalance |
| `BinaryFocalLoss` | Down-weights easy negatives; useful for sparse slum pixels |
| `BinaryLovaszLoss` | Surrogate for IoU; smooth extension of mIoU loss |
| `BinarySoftF1Loss` | Directly optimizes F1 score |

### Optimizers and Schedulers

**Optimizers**: Adam, AdamW, SGD, AdaBound, AdaBoundW, Yogi

- **AdaBound** (`src/utilities.py`): Clips adaptive learning rates within dynamic bounds that converge to a final learning rate, combining the benefits of Adam and SGD.
- **Yogi**: Adaptive optimizer for non-convex problems; uses additive rather than multiplicative updates for second-moment estimation, improving convergence on sparse gradients.

**Schedulers**: CosineAnnealingWarmRestarts, ReduceLROnPlateau, ExponentialLR, OneCycleLR

**Decoupled learning rates**: Transformer-based models support separate encoder (e.g., `2e-5`) and decoder (e.g., `3e-4`) learning rates to prevent catastrophic forgetting of pretrained encoder weights.

**Stochastic Weight Averaging (SWA)**: Optional; averages weights along the SGD trajectory to improve generalization.

### Data Augmentation

Joint image–label augmentations (applied consistently to both satellite tile and segmentation mask):

- Random horizontal and vertical flips
- `RandomRotateZoomCropTensor`: Rotates by a random angle in [0°, 90°], computes the largest valid inner square, and crops a random-sized sub-tile to avoid black padding in the output. Ensures augmented tiles have no boundary artifacts.
- Random channel shuffle (for multispectral inputs)
- Label noise injection: soft labels via `clamp(label, noise_level, 1 - noise_level)`, or distance-weighted soft labels via `LabelNoiseFromDistances`

**Self-Supervised Pre-Training (SSP)**: The `SSP_Generator` masks a configurable percentage of input pixels and uses Sobel edge detection on the unmasked image as the supervision signal. An autoencoding variant (`Autoencoding_SSP_Generator`) reconstructs masked pixel brightness from the RGB-to-grayscale transform.

---

## Inference Pipeline

### Tiling

Large satellite images (typically thousands of pixels) are decomposed into 512×512 tiles using one of two strategies:

- **Contiguous tiling** (`base_tiler.py`): Reflection-padded to allow exact reconstruction. Tile predictions are stitched back without overlap.
- **50% overlap tiling** (`overlap_tiler.py`): Each tile has 50% overlap with its neighbors plus 1/4-tile padding at the boundary. Overlapping regions are averaged at reconstruction, reducing edge discontinuities.

### Post-Processing

**Dense CRF** (`pydensecrf`) refinement is applied as a post-processing step. CRF unary potentials come from model logits; pairwise potentials are computed from the original satellite image (appearance kernel). Parallel CRF processing is supported across CPU cores.

### Ensemble and TTA

- **Model ensembling**: An odd number of checkpoints are evaluated independently; pixel-level majority voting produces the final prediction.
- **Test-Time Augmentation (TTA)**: Predictions for multiple augmented versions of each tile are majority-voted. Requires an odd number of augmentations.

---

## Evaluation Metrics

Logged per epoch and used for model selection:

| Metric | Description |
|---|---|
| `val_loss` | Validation loss (selected criterion) |
| `val_acc` | Binary pixel accuracy |
| `val_f1` | Binary F1 score |
| `val_iou` | Jaccard Index (IoU) |

Checkpoint filenames encode key metrics. Examples visible in configuration files:

- `unet_v2` (5 epochs, fine-tuned, MS→MD cross-city): `val_loss=0.1472`, `val_acc=0.9672`, `val_f1=0.8531`
- `unet_v2` (183 epochs, full training): `val_loss=0.2628`, `val_acc=0.9717`, `val_f1=0.8461`

---

## Output

Predictions are produced as:
- Binary probability maps (PNG)
- Colorized overlays on the original satellite image (Red = predicted slum, Blue = ground truth, Magenta = overlap, Green = masked region)
- **GIS Shapefiles** via `generate_shapefile.py`: binary masks are vectorized using `rasterio` / `geopandas` with coordinate reference system support for multiple cities (EPSG:32643 for Mumbai/PCMC, EPSG:32630 for Ouagadougou/Bobo, EPSG:32736 for Kigali).

---

## Quickstart

**1. Tile satellite images**
```bash
python runners/tile_standard.py -c Configurations_yamls/tile_tilingMS.yml
```

**2. Train a model**
```bash
python runners/train.py -c Configurations_yamls/train_unet_v2_dinov3_low_shot_finetune_test_vanilla_ms2016_to_md2001.yml
```

**3. Run inference on a new city**
```bash
python runners/inference.py -c <your_config>.yml
```

**4. Reconstruct and export**
```bash
python runners/reconstruct_map_from_tiles.py
python runners/generate_shapefile.py
```

---

## Dependencies

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

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Compute

Training is configured via `compute_parameters` in each YAML file:

```yaml
compute_parameters:
    precision: 32           # 16 / 32 / 64
    gpus: [1]               # specific GPU index or count
    auto_select_gpus: true  # auto-selects least-loaded GPU
    n_nodes: 1
    n_workers: 10           # dataloader worker processes
```

SLURM distributed training (DDP) is supported natively through PyTorch Lightning's `SLURMEnvironment` plugin.
