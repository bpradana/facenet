"""FastAPI service for FaceNet embeddings and verification."""

from __future__ import annotations

import base64
import io
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from ..config import InferenceConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from ..data.transforms import build_eval_transform
from ..models.lightning_module import FaceNetLightningModule


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

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def _decode_image(self, image_b64: str) -> torch.Tensor:
        try:
            image_bytes = base64.b64decode(image_b64)
        except base64.binascii.Error as exc:  # type: ignore[attr-defined]
            raise HTTPException(
                status_code=400, detail="Invalid base64 image payload"
            ) from exc

        with Image.open(io.BytesIO(image_bytes)) as img:
            tensor = self._preprocess_image(img)
        return tensor

    @torch.inference_mode()
    def embed(self, images_b64: List[str]) -> torch.Tensor:
        tensors = [self._decode_image(image) for image in images_b64]
        return self._embed_tensors(tensors)

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = [self._preprocess_image(image) for image in images]
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
