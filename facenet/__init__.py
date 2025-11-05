"""FaceNet-style face recognition package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("facenet")
except PackageNotFoundError:  # pragma: no cover - during local development
    __version__ = "0.0.0"

from .config import TrainingConfig, InferenceConfig, load_config
from .data.module import FaceDataModule
from .models.lightning_module import FaceNetLightningModule

__all__ = [
    "__version__",
    "TrainingConfig",
    "InferenceConfig",
    "load_config",
    "FaceDataModule",
    "FaceNetLightningModule",
]
