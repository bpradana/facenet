"""Benchmark multiple FaceNet configurations and generate paper-ready artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import random
import textwrap
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from PIL import Image

from facenet.config import ModelConfig, OptimizerConfig, SchedulerConfig, TrainingConfig
from facenet.data import FaceDataModule
from facenet.data.dataset import find_image_paths
from facenet.data.transforms import build_eval_transform
from facenet.evaluation import compute_verification_metrics
from facenet.models import FaceNetLightningModule
from facenet.utils.logging import configure_logging, get_logger
from facenet.utils.seed import seed_everything

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExperimentSpec:
    """Defines a single benchmark run."""

    idx: int
    backbone: str
    embedding_dim: int

    @property
    def run_name(self) -> str:
        return f"{self.idx:02d}_{self.backbone}_emb{self.embedding_dim}"


@dataclass
class QueryVisualization:
    """Stores metadata for a query/top-K visualization."""

    query_path: str
    retrieved: List[Dict[str, Any]]
    figure_path: str


@dataclass
class ExperimentResult:
    """Aggregated outputs from a benchmark run."""

    spec: ExperimentSpec
    run_dir: Path
    checkpoint_path: Path
    val_metrics: Dict[str, float]
    verification_metrics: Dict[str, float]
    training_seconds: float
    query_visualizations: List[QueryVisualization]


RUN_RESULT_FILENAME = "run_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate multiple FaceNet variants and generate benchmark visuals."
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="YAML file describing the benchmark sweep plus training hyperparameters.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=None,
        help="Backbone names to benchmark (defaults to backbone in config).",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        nargs="+",
        default=None,
        help="Embedding dimensions to evaluate (defaults to config value).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where benchmark artifacts will be stored.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of neighbors to visualize for each query face.",
    )
    parser.add_argument(
        "--num-random-queries",
        type=int,
        default=None,
        help="Number of random gallery queries to visualize per run.",
    )
    parser.add_argument(
        "--query-image",
        type=Path,
        default=None,
        help="Optional path to a specific query image to include in every run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used for evaluation/retrieval (training still follows config).",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Batch size used when extracting gallery embeddings.",
    )
    parser.add_argument(
        "--gallery-split",
        type=str,
        choices=["val", "train", "all"],
        default=None,
        help="Which identity split to visualize in retrieval examples.",
    )
    parser.add_argument(
        "--resume/--no-resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs that already produced artifacts and reuse their metrics.",
    )
    args = parser.parse_args()
    overrides = load_benchmark_overrides(args.benchmark_config)

    args.backbones = _ensure_str_list(args.backbones or overrides.get("backbones"))
    args.embedding_dims = _ensure_int_list(
        args.embedding_dims or overrides.get("embedding_dims")
    )
    args.output_dir = Path(
        args.output_dir or overrides.get("output_dir") or Path("artifacts/benchmark")
    )
    args.top_k = int(args.top_k or overrides.get("top_k") or 5)
    args.num_random_queries = int(
        args.num_random_queries or overrides.get("num_random_queries") or 1
    )
    args.device = (args.device or overrides.get("device") or "auto").lower()
    args.inference_batch_size = int(
        args.inference_batch_size or overrides.get("inference_batch_size") or 64
    )
    args.gallery_split = args.gallery_split or overrides.get("gallery_split") or "val"
    query_image = args.query_image or overrides.get("query_image")
    args.query_image = Path(query_image) if query_image else None
    args.experiments = _normalize_experiments(overrides.get("experiments"))

    training_section = overrides.get("training")
    if training_section is None:
        parser.error(
            "Benchmark config must include a 'training' block mirroring TrainingConfig."
        )
    if not isinstance(training_section, dict):
        parser.error("'training' block must be a mapping.")
    args.training_dict = training_section

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.num_random_queries < 0:
        parser.error("--num-random-queries must be >= 0")
    if args.inference_batch_size < 1:
        parser.error("--inference-batch-size must be >= 1")
    if args.query_image and not args.query_image.exists():
        parser.error(f"Query image not found: {args.query_image}")
    return args


def load_benchmark_overrides(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a mapping at the root.")
    return data


def _normalize_experiments(
    experiments: Optional[Iterable[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if experiments is None:
        return None
    normalized: List[Dict[str, Any]] = []
    for idx, experiment in enumerate(experiments, start=1):
        if not isinstance(experiment, dict):
            raise ValueError(
                f"Experiment #{idx} must be a mapping with 'backbone' and 'embedding_dim'."
            )
        backbone = experiment.get("backbone")
        embedding_dim = experiment.get("embedding_dim")
        if backbone is None or embedding_dim is None:
            raise ValueError(
                f"Experiment #{idx} missing 'backbone' or 'embedding_dim'."
            )
        normalized.append(
            {
                "backbone": str(backbone),
                "embedding_dim": int(embedding_dim),
            }
        )
    return normalized


def _ensure_str_list(value: Optional[Iterable[Any]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    raise ValueError("Expected a string or iterable of strings for backbones.")


def _ensure_int_list(value: Optional[Iterable[Any]]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [int(v) for v in value]
    return [int(value)]


def build_specs(
    backbones: Sequence[str],
    embedding_dims: Sequence[int],
    experiments: Optional[List[Dict[str, Any]]] = None,
) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    if experiments:
        for idx, exp in enumerate(experiments, start=1):
            specs.append(
                ExperimentSpec(
                    idx=idx,
                    backbone=str(exp["backbone"]),
                    embedding_dim=int(exp["embedding_dim"]),
                )
            )
        return specs

    idx = 1
    for backbone, emb_dim in product(backbones, embedding_dims):
        specs.append(ExperimentSpec(idx=idx, backbone=backbone, embedding_dim=emb_dim))
        idx += 1
    return specs


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():  # type: ignore[attr-defined]
            raise RuntimeError("MPS requested but not available on this system.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def run_single_experiment(
    spec: ExperimentSpec,
    base_cfg: TrainingConfig,
    args: argparse.Namespace,
    output_root: Path,
    eval_device: torch.device,
) -> ExperimentResult:
    logger.info(
        "[%s] Starting training (backbone=%s, embedding_dim=%s)",
        spec.run_name,
        spec.backbone,
        spec.embedding_dim,
    )
    cfg = copy.deepcopy(base_cfg)
    cfg.model.backbone = spec.backbone
    cfg.model.embedding_dim = spec.embedding_dim

    run_dir = output_root / spec.run_name
    train_artifacts = run_dir / "training"
    train_artifacts.mkdir(parents=True, exist_ok=True)
    cfg.logging.output_dir = str(train_artifacts)

    seed_everything(cfg.seed + spec.idx)
    data_module = FaceDataModule(cfg.data, seed=cfg.seed + spec.idx)
    model = FaceNetLightningModule(cfg.model, cfg.optimizer, cfg.scheduler)

    checkpoint_dir = Path(cfg.logging.output_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{spec.run_name}-{{epoch:02d}}",
        monitor=cfg.logging.monitor,
        mode=cfg.logging.mode,
        save_top_k=1,
        every_n_epochs=cfg.logging.checkpoint_interval,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning")

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        deterministic=cfg.trainer.deterministic,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=csv_logger,
    )

    start_time = time.perf_counter()
    trainer.fit(model, datamodule=data_module)
    training_seconds = time.perf_counter() - start_time

    val_metrics_list = trainer.validate(
        model=model, datamodule=data_module, ckpt_path="best"
    )
    val_metrics = val_metrics_list[0] if val_metrics_list else {}

    best_ckpt = Path(checkpoint_callback.best_model_path)
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint was saved for run {spec.run_name}. Check the monitor metric or logging config."
        )

    best_model = load_model_from_checkpoint(best_ckpt, eval_device)
    verification_metrics = run_verification(best_model, data_module, eval_device)

    data_module.setup(None)
    gallery_entries = build_gallery_entries(
        cfg.data.root,
        cfg.data.min_images_per_class,
        resolve_gallery_classes(data_module, args.gallery_split),
    )
    gallery_embeddings, gallery_metadata = embed_gallery(
        best_model,
        gallery_entries,
        cfg.data.image_size,
        eval_device,
        args.inference_batch_size,
    )
    query_visualizations = generate_query_visualizations(
        best_model,
        gallery_embeddings,
        gallery_metadata,
        cfg.data.image_size,
        spec,
        args,
        run_dir,
        eval_device,
    )

    logger.info(
        "[%s] Finished in %.1fs (val_acc=%.4f, tar=%.4f, far=%.4f)",
        spec.run_name,
        training_seconds,
        verification_metrics.get("best_accuracy", float("nan")),
        verification_metrics.get("tar", float("nan")),
        verification_metrics.get("far", float("nan")),
    )

    result = ExperimentResult(
        spec=spec,
        run_dir=run_dir,
        checkpoint_path=best_ckpt,
        val_metrics={k: float(v) for k, v in val_metrics.items()},
        verification_metrics=verification_metrics,
        training_seconds=training_seconds,
        query_visualizations=query_visualizations,
    )
    save_run_result(result)
    return result


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> FaceNetLightningModule:
    ckpt = torch.load(checkpoint_path, map_location=device)
    hyper_params = ckpt["hyper_parameters"]

    model_cfg = ModelConfig(**hyper_params["model"])
    optimizer_cfg = OptimizerConfig(**hyper_params["optimizer"])
    scheduler_data = hyper_params.get("scheduler")
    scheduler_cfg = SchedulerConfig(**scheduler_data) if scheduler_data else None

    module = FaceNetLightningModule(model_cfg, optimizer_cfg, scheduler_cfg)
    module.load_state_dict(ckpt["state_dict"])
    module.to(device)
    module.eval()
    return module


def run_verification(
    model: FaceNetLightningModule, data_module: FaceDataModule, device: torch.device
) -> Dict[str, float]:
    data_module.setup("validate")
    val_loader = data_module.val_dataloader()

    anchors: List[torch.Tensor] = []
    positives: List[torch.Tensor] = []
    negatives: List[torch.Tensor] = []

    with torch.inference_mode():
        for batch in val_loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            anchors.append(model(anchor).cpu())
            positives.append(model(positive).cpu())
            negatives.append(model(negative).cpu())

    anchor_tensor = torch.cat(anchors)
    positive_tensor = torch.cat(positives)
    negative_tensor = torch.cat(negatives)

    same_labels = torch.ones(anchor_tensor.size(0))
    diff_labels = torch.zeros(anchor_tensor.size(0))

    pairs_a = torch.cat([anchor_tensor, anchor_tensor])
    pairs_b = torch.cat([positive_tensor, negative_tensor])
    labels = torch.cat([same_labels, diff_labels])

    return compute_verification_metrics(pairs_a, pairs_b, labels)


def resolve_gallery_classes(
    data_module: FaceDataModule, strategy: str
) -> Optional[List[str]]:
    split = data_module.split_identities
    if not split or strategy == "all":
        return None
    train_classes, val_classes = split
    if strategy == "train" and train_classes:
        return list(train_classes)
    if strategy == "val" and val_classes:
        return list(val_classes)
    # Fallback if requested split is empty.
    return None


def build_gallery_entries(
    data_root: str, min_images_per_class: int, allowed_classes: Optional[List[str]]
) -> List[Dict[str, Any]]:
    root = Path(data_root)
    class_to_images = find_image_paths(root, min_images_per_class=min_images_per_class)
    if allowed_classes:
        allowed = [cls for cls in allowed_classes if cls in class_to_images]
        if not allowed:
            logger.warning(
                "Requested gallery split has no classes. Falling back to full dataset."
            )
        else:
            class_to_images = {name: class_to_images[name] for name in allowed}

    entries: List[Dict[str, Any]] = []
    for label_idx, class_name in enumerate(sorted(class_to_images.keys())):
        for path in class_to_images[class_name]:
            entries.append(
                {
                    "path": str(path),
                    "label": label_idx,
                    "label_name": class_name,
                }
            )
    if not entries:
        raise ValueError(
            "No gallery entries found. Check dataset root or min_images_per_class."
        )
    return entries


def embed_gallery(
    model: FaceNetLightningModule,
    entries: List[Dict[str, Any]],
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, List[Dict[str, Any]]]:
    transform = build_eval_transform(image_size)
    embeddings: List[torch.Tensor] = []
    batch_tensors: List[torch.Tensor] = []
    metadata: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for entry in entries:
            tensor = load_image_tensor(entry["path"], transform)
            batch_tensors.append(tensor)
            metadata.append(entry)
            if len(batch_tensors) == batch_size:
                embeddings.append(_forward_batch(model, batch_tensors, device))
                batch_tensors.clear()

        if batch_tensors:
            embeddings.append(_forward_batch(model, batch_tensors, device))

    gallery = torch.cat(embeddings)
    return gallery.cpu(), metadata


def _forward_batch(
    model: FaceNetLightningModule, tensors: List[torch.Tensor], device: torch.device
) -> torch.Tensor:
    batch = torch.stack(tensors).to(device)
    embeddings = model(batch)
    return embeddings.detach().cpu()


def load_image_tensor(path: str, transform) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def generate_query_visualizations(
    model: FaceNetLightningModule,
    gallery_embeddings: torch.Tensor,
    gallery_metadata: List[Dict[str, Any]],
    image_size: int,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> List[QueryVisualization]:
    if gallery_embeddings.size(0) <= 1:
        logger.warning("Gallery too small for query visualizations.")
        return []

    rng = random.Random(spec.idx + args.top_k)
    query_paths: List[Path] = []
    if args.query_image:
        query_paths.append(args.query_image)

    available_paths = [Path(meta["path"]) for meta in gallery_metadata]
    num_samples = min(args.num_random_queries, len(available_paths))
    if num_samples > 0:
        query_paths.extend(rng.sample(available_paths, k=num_samples))

    transform = build_eval_transform(image_size)
    visualizations: List[QueryVisualization] = []
    for query_idx, query_path in enumerate(query_paths):
        query_embedding = embed_query(model, query_path, transform, device)
        retrieved = retrieve_top_k(
            query_embedding,
            gallery_embeddings,
            gallery_metadata,
            args.top_k,
            exclude_path=str(query_path),
        )
        if not retrieved:
            continue
        figure_path = run_dir / f"query_{spec.idx:02d}_{query_idx:02d}.png"
        save_query_figure(query_path, retrieved, figure_path)
        visualizations.append(
            QueryVisualization(
                query_path=str(query_path),
                retrieved=retrieved,
                figure_path=str(figure_path),
            )
        )
    return visualizations


def _select_primary_visualization(
    result: ExperimentResult,
) -> Optional[QueryVisualization]:
    return result.query_visualizations[0] if result.query_visualizations else None


def embed_query(
    model: FaceNetLightningModule,
    path: Path,
    transform,
    device: torch.device,
) -> torch.Tensor:
    tensor = load_image_tensor(str(path), transform).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor)
    return embedding.detach().cpu().squeeze(0)


def retrieve_top_k(
    query_embedding: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_metadata: List[Dict[str, Any]],
    top_k: int,
    *,
    exclude_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    scores = torch.matmul(gallery_embeddings, query_embedding)
    sorted_indices = torch.argsort(scores, descending=True)
    hits: List[Dict[str, Any]] = []
    for idx in sorted_indices.tolist():
        candidate = gallery_metadata[idx]
        if exclude_path and Path(candidate["path"]) == Path(exclude_path):
            continue
        hits.append(
            {
                "path": candidate["path"],
                "label_name": candidate["label_name"],
                "similarity": float(scores[idx].item()),
            }
        )
        if len(hits) == top_k:
            break
    return hits


def save_query_figure(
    query_path: Path, retrieved: List[Dict[str, Any]], output_path: Path
) -> None:
    cols = len(retrieved) + 1
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))

    def _show(ax, path: Path, title: str) -> None:
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    _show(axes[0], query_path, "Query")
    for i, (ax, result) in enumerate(zip(axes[1:], retrieved), start=1):
        title = f"#{i}: {result['label_name']}\ncos={result['similarity']:.3f}"
        _show(ax, Path(result["path"]), title)

    fig.suptitle("Top-K Retrieval", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def result_to_dict(result: ExperimentResult) -> Dict[str, Any]:
    return {
        "spec": asdict(result.spec),
        "run_dir": str(result.run_dir),
        "checkpoint_path": str(result.checkpoint_path),
        "val_metrics": result.val_metrics,
        "verification_metrics": result.verification_metrics,
        "training_seconds": result.training_seconds,
        "query_visualizations": [asdict(viz) for viz in result.query_visualizations],
    }


def dict_to_result(
    data: Dict[str, Any], *, spec_override: Optional[ExperimentSpec] = None
) -> ExperimentResult:
    spec_dict = data.get("spec") or {}
    spec = spec_override or ExperimentSpec(
        idx=int(spec_dict.get("idx", 0)),
        backbone=spec_dict.get("backbone", "unknown"),
        embedding_dim=int(spec_dict.get("embedding_dim", 0)),
    )
    run_dir = Path(data.get("run_dir", spec.run_name))
    checkpoint_path = Path(data.get("checkpoint_path", ""))
    val_metrics = {k: float(v) for k, v in (data.get("val_metrics") or {}).items()}
    verification_metrics = {
        k: float(v) for k, v in (data.get("verification_metrics") or {}).items()
    }
    training_seconds = float(data.get("training_seconds", 0.0))
    query_visualizations = [
        QueryVisualization(**viz)
        for viz in data.get("query_visualizations", [])
        if isinstance(viz, dict)
    ]
    return ExperimentResult(
        spec=spec,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        val_metrics=val_metrics,
        verification_metrics=verification_metrics,
        training_seconds=training_seconds,
        query_visualizations=query_visualizations,
    )


def save_run_result(result: ExperimentResult) -> Path:
    path = result.run_dir / RUN_RESULT_FILENAME
    path.write_text(json.dumps(result_to_dict(result), indent=2), encoding="utf-8")
    return path


def load_existing_result(
    spec: ExperimentSpec, output_root: Path
) -> Optional[ExperimentResult]:
    run_dir = output_root / spec.run_name
    meta_path = run_dir / RUN_RESULT_FILENAME
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
    except Exception as exc:  # pragma: no cover - corrupted metadata
        logger.warning("Could not parse %s: %s", meta_path, exc)
        return None

    stored_spec = data.get("spec") or {}
    if (
        stored_spec.get("backbone") != spec.backbone
        or int(stored_spec.get("embedding_dim", -1)) != spec.embedding_dim
    ):
        return None

    result = dict_to_result(data, spec_override=spec)
    if not result.checkpoint_path.exists():
        logger.warning(
            "[%s] Checkpoint missing at %s. Run will be re-executed.",
            spec.run_name,
            result.checkpoint_path,
        )
        return None
    return result


def create_embedding_dim_query_grids(
    results: List[ExperimentResult], top_k: int, output_root: Path
) -> Dict[int, Path]:
    grouped: Dict[int, List[Tuple[ExperimentResult, QueryVisualization]]] = {}
    for result in results:
        viz = _select_primary_visualization(result)
        if viz is None:
            continue
        grouped.setdefault(result.spec.embedding_dim, []).append((result, viz))

    output_paths: Dict[int, Path] = {}
    for embedding_dim, entries in grouped.items():
        entries.sort(
            key=lambda item: item[0].verification_metrics.get(
                "best_accuracy", float("nan")
            ),
            reverse=True,
        )
        title = f"Embedding Dim {embedding_dim} Query Retrievals"
        path = output_root / f"embedding_dim_{embedding_dim}_queries.png"
        _render_query_grid(entries, top_k, path, title, row_mode="backbone")
        output_paths[embedding_dim] = path
    return output_paths


def create_backbone_query_grids(
    results: List[ExperimentResult], top_k: int, output_root: Path
) -> Dict[str, Path]:
    grouped: Dict[str, List[Tuple[ExperimentResult, QueryVisualization]]] = {}
    for result in results:
        viz = _select_primary_visualization(result)
        if viz is None:
            continue
        grouped.setdefault(result.spec.backbone, []).append((result, viz))

    output_paths: Dict[str, Path] = {}
    for backbone, entries in grouped.items():
        entries.sort(key=lambda item: item[0].spec.embedding_dim, reverse=True)
        safe_name = backbone.replace("/", "_")
        title = f"{backbone} Query Retrievals"
        path = output_root / f"backbone_{safe_name}_queries.png"
        _render_query_grid(entries, top_k, path, title, row_mode="embedding")
        output_paths[backbone] = path
    return output_paths


def _render_query_grid(
    entries: List[Tuple[ExperimentResult, QueryVisualization]],
    top_k: int,
    output_path: Path,
    title: str,
    row_mode: str,
) -> None:
    if not entries:
        return
    num_rows = len(entries)
    num_cols = top_k + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    if num_rows == 1:
        axes = [axes]

    for row_idx, (result, viz) in enumerate(entries):
        row_axes = axes[row_idx]
        row_label = _build_row_label(result, row_mode)
        _draw_image(row_axes[0], Path(viz.query_path), f"{row_label}\nQuery")
        retrieved = viz.retrieved[:top_k]
        for col_idx in range(1, num_cols):
            ax = row_axes[col_idx]
            hit_idx = col_idx - 1
            if hit_idx < len(retrieved):
                candidate = retrieved[hit_idx]
                caption = (
                    f"{candidate['label_name']}\ncos={candidate['similarity']:.3f}"
                )
                _draw_image(ax, Path(candidate["path"]), caption)
            else:
                ax.axis("off")

    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_row_label(result: ExperimentResult, row_mode: str) -> str:
    accuracy = result.verification_metrics.get("best_accuracy")
    if row_mode == "backbone":
        label = result.spec.backbone
        metric = f"acc={accuracy:.3f}" if accuracy is not None else "acc=N/A"
        return f"{label} ({metric})"
    if row_mode == "embedding":
        label = f"emb={result.spec.embedding_dim}"
        return label
    return ""


def _draw_image(ax, path: Path, title: str) -> None:
    try:
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
    except FileNotFoundError:
        ax.text(0.5, 0.5, "Missing image", ha="center", va="center")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_results(results: List[ExperimentResult], output_root: Path) -> Path:
    markers = [
        ".",
        "o",
        "v",
        "^",
        "1",
        "2",
        "s",
        "p",
        "*",
        "D",
        "P",
        "X",
    ]

    if not results:
        raise ValueError("No experiment results to plot.")

    grouped: Dict[str, List[tuple[int, float]]] = {}
    for result in results:
        backbone = result.spec.backbone
        acc = result.verification_metrics.get("best_accuracy", float("nan"))
        grouped.setdefault(backbone, []).append((result.spec.embedding_dim, acc))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (backbone, pairs) in enumerate(grouped.items()):
        pairs.sort(key=lambda item: item[0])
        dims = [p[0] for p in pairs]
        accs = [p[1] for p in pairs]
        ax.plot(dims, accs, marker=markers[i % len(markers)], label=backbone)

    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Verification Accuracy")
    ax.set_title("FaceNet Benchmark Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plot_path = output_root / "benchmark_accuracy.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_report(
    results: List[ExperimentResult],
    plot_path: Path,
    output_root: Path,
    cfg: TrainingConfig,
    args: argparse.Namespace,
    embedding_dim_grids: Dict[int, Path],
    backbone_grids: Dict[str, Path],
) -> Path:
    report_path = output_root / "benchmark_report.md"
    lines: List[str] = []
    lines.append("# FaceNet Benchmark Report\n")
    lines.append("## Dataset & Setup\n")
    lines.append(f"- Dataset root: `{cfg.data.root}`\n")
    lines.append(f"- Image size: {cfg.data.image_size}px\n")
    lines.append(
        f"- Batch size: {cfg.data.batch_size} (classes_per_batch={cfg.data.classes_per_batch}, samples_per_class={cfg.data.samples_per_class})\n"
    )
    lines.append(f"- Trainer epochs: {cfg.trainer.max_epochs}\n")
    lines.append(f"- Retrieval split: {args.gallery_split}\n")
    lines.append(f"- Top-K neighbors visualized: {args.top_k}\n")

    lines.append("\n## Summary Table\n")
    header = "| Run | Backbone | Emb Dim | Accuracy | TAR | FAR | Checkpoint |\n"
    separator = "| --- | --- | --- | --- | --- | --- | --- |\n"
    lines.extend([header, separator])
    for result in results:
        metrics = result.verification_metrics
        lines.append(
            "| {run} | {backbone} | {dim} | {acc:.4f} | {tar:.4f} | {far:.4f} | `{ckpt}` |\n".format(
                run=result.spec.run_name,
                backbone=result.spec.backbone,
                dim=result.spec.embedding_dim,
                acc=metrics.get("best_accuracy", float("nan")),
                tar=metrics.get("tar", float("nan")),
                far=metrics.get("far", float("nan")),
                ckpt=result.checkpoint_path,
            )
        )

    lines.append("\n## Figures\n")
    lines.append(f"![Accuracy Plot]({plot_path.name})\n")

    if embedding_dim_grids:
        lines.append("\n## Query grids by embedding dimension\n")
        for dim in sorted(embedding_dim_grids.keys()):
            rel = embedding_dim_grids[dim].relative_to(output_root)
            lines.append(f"### Embedding dim {dim}\n")
            lines.append(f"![Embedding dim {dim}]({rel.as_posix()})\n")

    if backbone_grids:
        lines.append("\n## Query grids by backbone\n")
        for backbone in sorted(backbone_grids.keys()):
            rel = backbone_grids[backbone].relative_to(output_root)
            lines.append(f"### Backbone {backbone}\n")
            lines.append(f"![Backbone {backbone}]({rel.as_posix()})\n")

    for result in results:
        if not result.query_visualizations:
            continue
        lines.append(f"\n### {result.spec.run_name} query results\n")
        for viz in result.query_visualizations:
            rel_path = Path(viz.figure_path).relative_to(output_root)
            caption = textwrap.dedent(
                f"""
                **Query:** `{viz.query_path}`  \\
                **Top-K:** {", ".join(f"{hit['label_name']} ({hit['similarity']:.3f})" for hit in viz.retrieved)}
                """
            ).strip()
            lines.append(f"![{result.spec.run_name}]({rel_path.as_posix()})\n")
            lines.append(caption + "\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_results_json(results: List[ExperimentResult], output_root: Path) -> Path:
    path = output_root / "benchmark_results.json"
    serializable = [result_to_dict(result) for result in results]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(**args.training_dict)

    backbones = args.backbones or [cfg.model.backbone]
    embedding_dims = args.embedding_dims or [cfg.model.embedding_dim]
    specs = build_specs(backbones, embedding_dims, args.experiments)
    if not specs:
        raise ValueError(
            "No experiments were defined. Provide backbones/embedding dims."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(str(args.output_dir))
    eval_device = select_device(args.device)
    logger.info(
        "Benchmarking %d runs (device=%s, dataset=%s)",
        len(specs),
        eval_device,
        cfg.data.root,
    )

    results: List[ExperimentResult] = []
    for spec in specs:
        existing = load_existing_result(spec, args.output_dir) if args.resume else None
        if existing:
            logger.info(
                "[%s] Resuming from %s",
                spec.run_name,
                existing.checkpoint_path,
            )
            results.append(existing)
            continue
        result = run_single_experiment(spec, cfg, args, args.output_dir, eval_device)
        results.append(result)

    plot_path = plot_results(results, args.output_dir)
    embedding_dim_grids = create_embedding_dim_query_grids(
        results, args.top_k, args.output_dir
    )
    backbone_grids = create_backbone_query_grids(results, args.top_k, args.output_dir)
    report_path = write_report(
        results,
        plot_path,
        args.output_dir,
        cfg,
        args,
        embedding_dim_grids,
        backbone_grids,
    )
    save_results_json(results, args.output_dir)
    logger.info("Benchmark complete. Report written to %s", report_path)


if __name__ == "__main__":
    main()
