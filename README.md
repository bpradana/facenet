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
- Configurable backbone (`resnet18`, `resnet34`, `resnet50`) with 512-D embeddings.
- Online batch-hard triplet mining per mini-batch (hard positive/negative selection).
- Triplet loss with cosine-similarity evaluation logging.

Artifacts (logs and checkpoints) are written to `artifacts/` by default.

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

## Production considerations

- Configure `trainer.devices` and `trainer.accelerator` to leverage GPUs.
- Use the provided `export.py` to integrate the embedding model into existing services.
- Monitor model drift by re-running `scripts/evaluate.py` on fresh validation identities.
- Consider integrating face detection/alignment upstream for improved accuracy (outside the scope of this repository).

## License

MIT.
