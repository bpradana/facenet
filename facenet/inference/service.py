"""FastAPI service for FaceNet embeddings and verification."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from ..config import InferenceConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from ..data.transforms import build_eval_transform
from ..models.lightning_module import FaceNetLightningModule

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

try:
    import supervision as sv
except ImportError:  # pragma: no cover - optional dependency
    sv = None


def _load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> FaceNetLightningModule:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyper_params = checkpoint["hyper_parameters"]

    model_cfg = ModelConfig(**hyper_params["model"])
    optimizer_cfg = OptimizerConfig(**hyper_params["optimizer"])
    scheduler_data = hyper_params.get("scheduler")
    scheduler_cfg = SchedulerConfig(**scheduler_data) if scheduler_data else None

    module = FaceNetLightningModule(model_cfg, optimizer_cfg, scheduler_cfg)
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    module.to(device)
    return module


class EmbedRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded RGB images.")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class VerifyRequest(BaseModel):
    image_a: str = Field(..., description="Base64 encoded RGB image for subject A.")
    image_b: str = Field(..., description="Base64 encoded RGB image for subject B.")
    threshold: float = Field(
        0.5, ge=-1.0, le=1.0, description="Cosine similarity threshold."
    )


class VerifyResponse(BaseModel):
    similarity: float
    is_match: bool


class EmbeddingService:
    """Service class wrapping model inference logic."""

    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = _load_model_from_checkpoint(cfg.checkpoint_path, self.device)
        self.transform = build_eval_transform(cfg.image_size)
        self.detector = self._load_detector()
        self.box_annotator = (
            sv.BoxAnnotator() if self.detector is not None and sv is not None else None
        )

    def _load_detector(self) -> Optional["YOLO"]:
        if not self.cfg.detector_weights:
            return None
        if YOLO is None or sv is None:
            raise ImportError(
                "ultralytics and supervision must be installed to use face detection."
            )
        weights_path = Path(self.cfg.detector_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Detector weights not found at {weights_path}.")
        detector = YOLO(str(weights_path))
        try:
            detector.to(str(self.device))
        except Exception:  # pragma: no cover - fallback to default device
            pass
        return detector

    def _preprocess_image(
        self, image: Image.Image, *, return_visual: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Image.Image]:
        face, annotated = self._extract_face(image, draw=return_visual)
        rgb = face.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        if return_visual:
            annotated_image = annotated or image
            return tensor, annotated_image
        return tensor

    def _extract_face(
        self, image: Image.Image, *, draw: bool = False
    ) -> tuple[Image.Image, Optional[Image.Image]]:
        if self.detector is None:
            return image, image if draw else None

        results = self.detector.predict(
            image, conf=self.cfg.detector_confidence, verbose=False
        )
        if not results:
            raise ValueError("Face detector returned no results.")
        result = results[0]
        detections = sv.Detections.from_ultralytics(result)
        if detections.xyxy.size == 0:
            raise ValueError("No face detected in the provided image.")

        confidences = detections.confidence
        has_conf = confidences is not None and confidences.size > 0
        if has_conf:
            idx = int(np.argmax(confidences))
            conf_value = float(confidences[idx])
        else:
            idx = 0
            conf_value = 0.0

        x1, y1, x2, y2 = detections.xyxy[idx].astype(float)
        width, height = image.size
        box_w = x2 - x1
        box_h = y2 - y1
        side = min(max(box_w, box_h), float(width), float(height))
        half = side / 2.0
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        new_x1 = cx - half
        new_y1 = cy - half
        new_x2 = cx + half
        new_y2 = cy + half

        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0.0
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0.0
        if new_x2 > width:
            shift = new_x2 - width
            new_x1 -= shift
            new_x2 = float(width)
        if new_y2 > height:
            shift = new_y2 - height
            new_y1 -= shift
            new_y2 = float(height)

        new_x1 = max(0.0, new_x1)
        new_y1 = max(0.0, new_y1)
        new_x2 = min(float(width), new_x2)
        new_y2 = min(float(height), new_y2)

        side = min(new_x2 - new_x1, new_y2 - new_y1)
        new_x2 = new_x1 + side
        new_y2 = new_y1 + side

        x1_i = int(np.floor(new_x1))
        y1_i = int(np.floor(new_y1))
        x2_i = int(np.ceil(new_x2))
        y2_i = int(np.ceil(new_y2))

        if x2_i <= x1_i:
            x2_i = min(width, x1_i + 1)
        if y2_i <= y1_i:
            y2_i = min(height, y1_i + 1)

        face = image.crop((x1_i, y1_i, x2_i, y2_i))

        annotated = None
        if draw:
            if self.box_annotator is None:
                annotated = image
            else:
                scene = np.array(image.convert("RGB"))
                square_detection = sv.Detections(
                    xyxy=np.array([[x1_i, y1_i, x2_i, y2_i]], dtype=np.float32),
                    confidence=np.array([conf_value], dtype=np.float32),
                    class_id=np.zeros(1, dtype=int),
                )
                annotated_scene = self.box_annotator.annotate(
                    scene=scene, detections=square_detection
                )
                annotated = Image.fromarray(annotated_scene)

        return face, annotated

    def _decode_image(self, image_b64: str) -> torch.Tensor:
        try:
            image_bytes = base64.b64decode(image_b64)
        except base64.binascii.Error as exc:  # type: ignore[attr-defined]
            raise HTTPException(
                status_code=400, detail="Invalid base64 image payload"
            ) from exc

        with Image.open(io.BytesIO(image_bytes)) as img:
            try:
                tensor = self._preprocess_image(img)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return tensor

    @torch.inference_mode()
    def embed(self, images_b64: List[str]) -> torch.Tensor:
        tensors = [self._decode_image(image) for image in images_b64]
        return self._embed_tensors(tensors)

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for image in images:
            try:
                tensor = self._preprocess_image(image)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            tensors.append(tensor)
        return self._embed_tensors(tensors)

    @torch.inference_mode()
    def verify(self, image_a: str, image_b: str, threshold: float) -> VerifyResponse:
        embeddings = self.embed([image_a, image_b])
        return self._verify_from_embeddings(embeddings, threshold)

    @torch.inference_mode()
    def verify_images(
        self, image_a: Image.Image, image_b: Image.Image, threshold: float
    ) -> VerifyResponse:
        embeddings = self.embed_images([image_a, image_b])
        return self._verify_from_embeddings(embeddings, threshold)

    @torch.inference_mode()
    def verify_images_with_visuals(
        self, image_a: Image.Image, image_b: Image.Image, threshold: float
    ) -> tuple[VerifyResponse, List[Image.Image]]:
        tensors: List[torch.Tensor] = []
        visuals: List[Image.Image] = []
        for image in (image_a, image_b):
            try:
                tensor, visual = self._preprocess_image(image, return_visual=True)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            tensors.append(tensor)
            visuals.append(visual)
        embeddings = self._embed_tensors(tensors)
        response = self._verify_from_embeddings(embeddings, threshold)
        return response, visuals

    def _embed_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        batch = torch.cat(tensors, dim=0)
        embeddings = self.model(batch)
        if self.cfg.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    def _verify_from_embeddings(
        self, embeddings: torch.Tensor, threshold: float
    ) -> VerifyResponse:
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        ).item()
        is_match = similarity >= threshold
        return VerifyResponse(similarity=similarity, is_match=is_match)


def create_app(service: EmbeddingService) -> FastAPI:
    """Create a FastAPI application using the provided service."""

    app = FastAPI(title="FaceNet Inference Service", version="1.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/embed", response_model=EmbedResponse)
    def embed(request: EmbedRequest) -> EmbedResponse:
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided.")
        embeddings = service.embed(request.images)
        return EmbedResponse(embeddings=embeddings.tolist())

    @app.post("/verify", response_model=VerifyResponse)
    def verify(request: VerifyRequest) -> VerifyResponse:
        return service.verify(request.image_a, request.image_b, request.threshold)

    return app
