"""Configuration dataclasses and helpers for the FaceNet project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class DataConfig:
    """Parameters for data loading and preprocessing."""

    root: str = "dataset"
    image_size: int = 160
    batch_size: int = 32
    num_workers: int = 4
    classes_per_batch: int = 8
    samples_per_class: int = 2
    val_split: float = 0.1
    shuffle: bool = True
    min_images_per_class: int = 2
    augmentations: bool = True
    cache_dataset: bool = False
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    """Definition of the embedding network."""

    backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet50"
    embedding_dim: int = 512
    pretrained: bool = True
    dropout: float = 0.0
    margin: float = 0.5
    train_backbone: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters."""

    name: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    """Learning rate scheduler parameters."""

    name: Optional[Literal["cosine", "multistep"]] = "cosine"
    warmup_steps: int = 500
    t_max: int = 10000
    milestones: tuple[int, ...] = (10, 20)
    gamma: float = 0.2


@dataclass
class LoggingConfig:
    """Logging and checkpoint management."""

    output_dir: str = "artifacts"
    checkpoint_interval: int = 1
    checkpoint_top_k: int = 3
    monitor: str = "val/avg_distance"
    mode: Literal["min", "max"] = "min"
    log_every_n_steps: int = 50


@dataclass
class TrainerConfig:
    """PyTorch Lightning trainer configuration."""

    max_epochs: int = 30
    devices: int | list[int] | str = 1
    accelerator: Literal["cpu", "gpu", "auto"] = "auto"
    precision: Literal["16-mixed", "bf16-mixed", "32-true"] = "32-true"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.0
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None
    deterministic: bool = True
    enable_progress_bar: bool = True


@dataclass
class TrainingConfig:
    """Top-level configuration for training runs."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.optimizer, dict):
            optimizer_dict = dict(self.optimizer)
            betas = optimizer_dict.get("betas")
            if isinstance(betas, list):
                optimizer_dict["betas"] = tuple(betas)
            self.optimizer = OptimizerConfig(**optimizer_dict)
        if isinstance(self.scheduler, dict):
            scheduler_dict = dict(self.scheduler)
            milestones = scheduler_dict.get("milestones")
            if isinstance(milestones, list):
                scheduler_dict["milestones"] = tuple(milestones)
            self.scheduler = SchedulerConfig(**scheduler_dict)
        if isinstance(self.trainer, dict):
            self.trainer = TrainerConfig(**self.trainer)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)


@dataclass
class InferenceConfig:
    """Configuration for running inference service or batch embedding extraction."""

    checkpoint_path: str
    device: Literal["cpu", "cuda"] = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 64
    num_workers: int = 2
    image_size: int = 160
    detection_threshold: float = 0.8
    detector_weights: Optional[str] = None
    detector_confidence: float = 0.3
    database: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.checkpoint_path = str(Path(self.checkpoint_path))
        if self.detector_weights:
            self.detector_weights = str(Path(self.detector_weights))
        if isinstance(self.database, dict):
            defaults = {
                "host": "localhost",
                "port": 5432,
                "name": "facenet",
                "user": "facenet_user",
                "password": "facenet_pass",
                "table": "face_embeddings",
                "identity_table": None,
                "sslmode": None,
            }
            merged = {**defaults, **self.database}
            merged["port"] = int(merged.get("port") or 5432)
            self.database = merged
        else:
            raise ValueError("database configuration must be a mapping.")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file {path} must contain a mapping at the root."
        )
    return data


def load_config(path: str | Path, *, config_type: type[Any] = TrainingConfig) -> Any:
    """Load a YAML configuration file into the desired dataclass.

    Args:
        path: Path to the YAML configuration file.
        config_type: Dataclass type to instantiate (TrainingConfig or InferenceConfig).

    Returns:
        An instance of the requested dataclass populated with values from file.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find configuration file: {config_path}")
    raw = _load_yaml(config_path)
    return config_type(**raw)


__all__ = [
    "DataConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LoggingConfig",
    "TrainerConfig",
    "TrainingConfig",
    "InferenceConfig",
    "load_config",
]
