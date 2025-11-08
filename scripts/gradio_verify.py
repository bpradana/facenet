"""Launch a Gradio interface for face registration, verification, and search."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from facenet.config import InferenceConfig, load_config
from facenet.db import PostgresConfig, PostgresEmbeddingStore
from facenet.inference import EmbeddingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio face recognition studio.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Path to inference YAML configuration.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host/IP for Gradio server."
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for Gradio server."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing (may require network access).",
    )
    return parser.parse_args()


def build_interface(
    service: EmbeddingService,
    store: Optional[PostgresEmbeddingStore],
) -> gr.Blocks:
    def to_identity_table(identities: List[str]) -> List[List[str]]:
        return [[name] for name in identities]

    def register_face(name: str, image, notes: str):
        if store is None:
            return None, {"error": "Embedding store is not configured."}, []
        name = (name or "").strip()
        if not name:
            return None, {"error": "Name is required."}, to_identity_table(
                store.list_identities()
            )
        if image is None:
            return None, {"error": "Please upload an image."}, to_identity_table(
                store.list_identities()
            )
        try:
            embeddings, visuals = service.embeddings_with_visuals([image])
        except ValueError as exc:
            return None, {"error": str(exc)}, to_identity_table(
                store.list_identities()
            )
        metadata = {"notes": notes} if notes else None
        inserted = store.register(name, embeddings, metadata=metadata)
        identities = store.list_identities()
        status = {
            "message": f"Registered {inserted} embedding(s) for '{name}'.",
            "total_identities": len(identities),
        }
        return visuals[0], status, to_identity_table(identities)

    def verify_faces(image_a, image_b, threshold: float):
        if image_a is None or image_b is None:
            return None, None, {"error": "Please upload two face images."}
        try:
            result, visuals = service.verify_images_with_visuals(
                image_a, image_b, threshold
            )
        except ValueError as exc:
            return None, None, {"error": str(exc)}
        response = {
            "similarity": round(result.similarity, 4),
            "threshold": threshold,
            "is_match": bool(result.is_match),
        }
        annotated_a, annotated_b = visuals
        return annotated_a, annotated_b, response

    def search_faces(image, top_k: int):
        if store is None:
            return None, [], {"error": "Embedding store is not configured."}
        if image is None:
            return None, [], {"error": "Please upload a query image."}
        if store.count() == 0:
            return None, [], {"error": "No registered embeddings yet."}
        try:
            embeddings, visuals = service.embeddings_with_visuals([image])
        except ValueError as exc:
            return None, [], {"error": str(exc)}
        results = store.search(embeddings[0], top_k=top_k)
        data = [
            [
                res.identity,
                round(res.similarity, 4),
                round(res.distance, 4),
                res.created_at.isoformat()
                if hasattr(res.created_at, "isoformat")
                else str(res.created_at),
            ]
            for res in results
        ]
        status: Dict[str, Any] = {"matches": len(data)}
        return visuals[0], data, status

    initial_identities = (
        to_identity_table(store.list_identities()) if store is not None else []
    )

    with gr.Blocks(title="FaceNet Recognition Studio") as demo:
        gr.Markdown(
            "## FaceNet Recognition Studio\n"
            "Register identities, verify two images, or search the database. "
            "Images are auto-cropped with YOLO (if configured) before embedding."
        )
        with gr.Tabs():
            with gr.Tab("Register"):
                with gr.Row():
                    with gr.Column(scale=1):
                        name_input = gr.Textbox(
                            label="Identity Name", placeholder="Jane Doe"
                        )
                        notes_input = gr.Textbox(
                            label="Notes (optional)",
                            placeholder="Source, context, etc.",
                        )
                        register_image = gr.Image(
                            label="Registration Image",
                            type="pil",
                            image_mode="RGB",
                        )
                        register_btn = gr.Button(
                            "Register Identity", variant="primary"
                        )
                    with gr.Column(scale=1):
                        annotated_reg = gr.Image(
                            label="Detected Face", interactive=False
                        )
                        register_status = gr.JSON(label="Registration Status")
                        identities_table = gr.Dataframe(
                            headers=["Identity"],
                            datatype=["str"],
                            interactive=False,
                            row_count=(0, "dynamic"),
                            col_count=(1, "fixed"),
                            label="Registered Identities",
                            value=initial_identities,
                        )
                register_btn.click(
                    fn=register_face,
                    inputs=[name_input, register_image, notes_input],
                    outputs=[annotated_reg, register_status, identities_table],
                )

            with gr.Tab("Verify"):
                with gr.Row():
                    image_a = gr.Image(label="Image A", type="pil", image_mode="RGB")
                    image_b = gr.Image(label="Image B", type="pil", image_mode="RGB")
                threshold = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                    label="Cosine Similarity Threshold",
                )
                with gr.Row():
                    annotated_a = gr.Image(label="Detected Faces A")
                    annotated_b = gr.Image(label="Detected Faces B")
                verify_status = gr.JSON(label="Verification Result")
                verify_btn = gr.Button("Verify Faces", variant="primary")
                verify_btn.click(
                    fn=verify_faces,
                    inputs=[image_a, image_b, threshold],
                    outputs=[annotated_a, annotated_b, verify_status],
                )

            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        search_image = gr.Image(
                            label="Query Image", type="pil", image_mode="RGB"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Top-K Results",
                        )
                        search_btn = gr.Button(
                            "Search Database", variant="primary"
                        )
                    with gr.Column(scale=1):
                        annotated_query = gr.Image(
                            label="Detected Query Face", interactive=False
                        )
                        search_results = gr.Dataframe(
                            headers=[
                                "identity",
                                "similarity",
                                "distance",
                                "created_at",
                            ],
                            datatype=["str", "number", "number", "str"],
                            interactive=False,
                            row_count=(0, "dynamic"),
                            col_count=(4, "fixed"),
                            label="Search Results",
                        )
                        search_status = gr.JSON(label="Search Summary")
                search_btn.click(
                    fn=search_faces,
                    inputs=[search_image, top_k],
                    outputs=[annotated_query, search_results, search_status],
                )

    return demo


def main() -> None:
    args = parse_args()
    cfg: InferenceConfig = load_config(args.config, config_type=InferenceConfig)
    service = EmbeddingService(cfg)
    store: Optional[PostgresEmbeddingStore] = None
    embedding_dim = getattr(service, "embedding_dim", None)
    if embedding_dim is None:
        raise RuntimeError("Unable to infer embedding dimension from model.")

    db_cfg = cfg.database or {}
    db_host = db_cfg.get("host")
    db_port = int(db_cfg.get("port") or 5432)
    db_name = db_cfg.get("name")
    db_user = db_cfg.get("user")
    db_password = db_cfg.get("password")
    db_table = db_cfg.get("table", "face_embeddings")
    db_identity_table = db_cfg.get("identity_table")
    db_sslmode = db_cfg.get("sslmode")

    if db_host and db_name and db_user:
        pg_config = PostgresConfig(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password or "",
            sslmode=db_sslmode,
        )
        store = PostgresEmbeddingStore(
            pg_config,
            db_table,
            embedding_dim=int(embedding_dim),
            identities_table=db_identity_table,
        )

    demo = build_interface(service, store)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
