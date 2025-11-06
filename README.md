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

## Gradio verification demo

Launch an interactive UI for manual image comparison:

```bash
python scripts/gradio_verify.py --config configs/inference.yaml
```

Upload two photos (full frames are fine); the app will optionally run YOLO-based face detection using the configured weights, crop the faces, and report cosine similarity versus the chosen threshold.
If detection is active, the UI renders bounding boxes using Roboflow Supervision for quick visual confirmation.

## Production considerations

- Configure `trainer.devices` and `trainer.accelerator` to leverage GPUs.
- Use the provided `export.py` to integrate the embedding model into existing services.
- Monitor model drift by re-running `scripts/evaluate.py` on fresh validation identities.
- Consider integrating face detection/alignment upstream for improved accuracy (outside the scope of this repository).

## License

MIT (update as appropriate for your use case).
