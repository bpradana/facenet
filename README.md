FaceNet-style Face Recognition
==============================

This project provides a production-oriented PyTorch implementation of a FaceNet-style face recognition system trained with triplet loss. It includes modular data loading, model training with PyTorch Lightning, evaluation utilities, TorchScript export, and a FastAPI inference service.

## Project structure

- `facenet/` – Core Python package (data pipeline, models, losses, inference utilities).
- `configs/` – YAML configuration files for training and inference.
- `scripts/` – Executable entrypoints for training, evaluation, export, and serving.
- `dataset/` – Expected dataset root (identity folders containing face images).

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Ensure you install the project in editable mode so console scripts use the local source.

## Dataset preparation

Place your face dataset under `dataset/` using the structure:

```
dataset/
└── person_id/
    ├── img1.jpg
    └── img2.jpg
```

Each identity directory must provide at least two images to support triplet sampling.

## Training

```bash
python scripts/train.py --config configs/train.yaml
```

Tweak `data.classes_per_batch` and `data.samples_per_class` (>=2) to control how many identities and samples per identity are used when performing batch-hard mining. The effective batch size is `classes_per_batch × samples_per_class` and must match `data.batch_size`.

Key features:
- PyTorch Lightning for distributed training and automatic checkpointing.
- Configurable backbone backed by `torchvision.models.get_model`, so any classification architecture shipped with your installed torchvision (ResNets, EfficientNets, ConvNeXt, ViT, etc.) can become an embedding encoder with a single config change.
- Online batch-hard triplet mining per mini-batch (hard positive/negative selection).
- Triplet loss with cosine-similarity evaluation logging.

Artifacts (logs and checkpoints) are written to `artifacts/` by default.

### Supported Torchvision backbones

We rely on `torchvision.models.get_model`, so any classification architecture exposed via `torchvision>=0.24` works out of the box. The list below mirrors `torchvision.models.list_models()` (grouped by family) and will grow automatically as PyTorch releases new architectures:

- **AlexNet** – `alexnet`
- **VGG** – `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`
- **ResNet & friends** – `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `wide_resnet50_2`, `wide_resnet101_2`
- **ConvNeXt** – `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`
- **ConvNeXt V2** – `convnextv2_atto`, `convnextv2_femto`, `convnextv2_pico`, `convnextv2_nano`, `convnextv2_tiny`, `convnextv2_base`, `convnextv2_large`, `convnextv2_huge`
- **DenseNet** – `densenet121`, `densenet161`, `densenet169`, `densenet201`
- **EfficientNet** – `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`, `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l`
- **GoogLeNet & Inception** – `googlenet`, `inception_v3`
- **RegNet** – `regnet_x_400mf`, `regnet_x_800mf`, `regnet_x_1_6gf`, `regnet_x_3_2gf`, `regnet_x_8gf`, `regnet_x_16gf`, `regnet_x_32gf`, `regnet_y_400mf`, `regnet_y_800mf`, `regnet_y_1_6gf`, `regnet_y_3_2gf`, `regnet_y_8gf`, `regnet_y_16gf`, `regnet_y_32gf`, `regnet_y_128gf`
- **MobileNet / MNAS / Shuffle / Squeeze** – `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`, `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`, `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`, `squeezenet1_0`, `squeezenet1_1`
- **Vision Transformers** – `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, `vit_h_14`, `vit_b_16_swag_e2e`, `vit_b_16_swag_linear`, `vit_l_16_swag_e2e`, `vit_l_16_swag_linear`
- **BEiT** – `beit_b_16_224`, `beit_b_16_224_in22k`, `beit_b_16_224_in22k_ft_in1k`, `beit_l_16_224`, `beit_l_16_224_in22k`, `beit_l_16_224_in22k_ft_in1k`
- **Swin / SwinV2** – `swin_t`, `swin_s`, `swin_b`, `swin_v2_t`, `swin_v2_s`, `swin_v2_b`
- **MaxViT & MViT** – `maxvit_t`, `mvit_v2_s`, `mvit_v2_m`
To confirm or discover new arrivals in future torchvision releases, run:

```bash
python - <<'PY'
from torchvision import models
print("\n".join(models.list_models()))
PY
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/train.yaml --checkpoint <checkpoint_path>
```

The evaluation script computes cosine-similarity verification metrics (accuracy, TAR/FAR) on the validation split identities.

## Export for inference

```bash
python scripts/export.py --checkpoint <checkpoint_path> --output facenet_embedding_model.ts
```

This exports a TorchScript module suitable for embedding extraction in production systems without requiring Lightning.

## Inference API

1. Update `configs/inference.yaml` with the checkpoint path generated during training.
2. Launch the FastAPI service:

```bash
python scripts/serve.py --config configs/inference.yaml
```

3. Send requests (Base64-encoded RGB images):

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"image_a":"<base64>", "image_b":"<base64>", "threshold":0.5}'
```

The API returns cosine similarity scores and match decisions.
If you set `detector_weights` in `configs/inference.yaml` to a local Ultralytics YOLO face-detector checkpoint (e.g. `yolov8n-face.pt`), incoming images are cropped to the strongest detected face before embedding.
Update the `database` block in `configs/inference.yaml` with your Postgres host, credentials, and table names so the Gradio registration/search features can persist embeddings via pgvector. The service now maintains two tables: `identities` (one centroid embedding per user plus metadata) and `embeddings` (all raw samples). Every registration inserts the raw vectors and recomputes the centroid stored in `identities`, and search compares queries against those centroids.

## Gradio recognition studio

Launch an interactive UI for registration, verification, and vector search:

```bash
python scripts/gradio_verify.py --config configs/inference.yaml
```

The app reads Postgres connection info from `configs/inference.yaml` (see `database:` block); update those values before launching.

Features:

- **Register** tab: label an identity, upload a face, and persist the embedding via Postgres + pgvector.
- **Verify** tab: compare two faces (cosine similarity vs. threshold).
- **Search** tab: embed a query face and retrieve top-K nearest identities from the vector store.

Full-frame uploads are fine; YOLO (when configured) crops to a square face region, and Roboflow Supervision overlays the bounding boxes for visual confirmation.

### Live detection & identification

Stream from a webcam, detect faces, and label them against the database:

```bash
python scripts/live_detect.py --config configs/inference.yaml
```

Press `q` to exit. Optional flags (`--camera`, `--top-k`, `--min-similarity`, `--show-similarity`) control runtime behavior. The script expects YOLO weights (`detector_weights`) and Postgres credentials in `configs/inference.yaml`, and renders bounding boxes plus identity labels in real time.

## Benchmark suite

Use `scripts/benchmark.py` to sweep architectures/embedding sizes, log verification metrics, and generate paper-ready figures plus qualitative retrieval panels.

```bash
python scripts/benchmark.py \
    --benchmark-config configs/benchmark.yaml
```

Outputs land in `artifacts/benchmark/` by default:
- Per-run Lightning checkpoints/logs under `artifacts/benchmark/<run_name>/`.
- `benchmark_results.json` capturing val metrics, TAR/FAR, and timing data.
- `benchmark_accuracy.png` plotting accuracy vs. embedding dimension for each backbone.
- `benchmark_report.md` linking the plot plus query-vs-top-K panels so you can paste the visuals straight into a paper or deck.
- `embedding_dim_<dim>_queries.png` grids: columns are `1 + top_k`, rows are backbones sorted by accuracy for that embedding size.
- `backbone_<name>_queries.png` grids: columns are `1 + top_k`, rows are embedding sizes sorted from largest to smallest for that backbone.

Optional flags include `--query-image` (force a specific probe image), `--gallery-split` (`val`/`train`/`all`), `--inference-batch-size`, and `--device` to control evaluation hardware. The script reuses the training block defined inside the benchmark YAML for optimizer, scheduler, and dataloading, so any edit you make there automatically propagates to every benchmark run.

The file `configs/benchmark.yaml` captures both the base training hyperparameters (`training:` block) and the sweep definition (backbones, embedding dims, output dir, etc.). Only include shared model settings (dropout, margin, etc.) inside `training.model`; the per-run backbone + embedding dimensionality comes from the sweep section or the optional `experiments` list, so you don’t need to duplicate them.

Advanced users can create another YAML with the same structure, point `--benchmark-config` to it, or provide an `experiments` list to enumerate arbitrary `(backbone, embedding_dim)` pairs instead of using cross-product sweeps. The aggregated query grids respect the requested ordering (best-to-worst accuracy rows for each embedding dimension; largest-to-smallest embeddings for each backbone), making them camera-ready for papers.

## Production considerations

- Configure `trainer.devices` and `trainer.accelerator` to leverage GPUs.
- Use the provided `export.py` to integrate the embedding model into existing services.
- Monitor model drift by re-running `scripts/evaluate.py` on fresh validation identities.
- Consider integrating face detection/alignment upstream for improved accuracy (outside the scope of this repository).

## License

MIT.
