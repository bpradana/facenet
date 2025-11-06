"""Launch a Gradio interface for face verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from facenet.config import InferenceConfig, load_config
from facenet.inference import EmbeddingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio face verification demo.")
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


def build_interface(service: EmbeddingService) -> gr.Blocks:
    def verify_faces(
        image_a, image_b, threshold: float
    ) -> Dict[str, Any]:
        if image_a is None or image_b is None:
            return {"error": "Please upload two face images."}
        result = service.verify_images(image_a, image_b, threshold)
        return {
            "similarity": round(result.similarity, 4),
            "threshold": threshold,
            "is_match": bool(result.is_match),
        }

    with gr.Blocks(title="Face Verification (FaceNet)") as demo:
        gr.Markdown(
            "## Face Verification Demo\n"
            "Upload two face images to compare embeddings using the trained FaceNet model."
        )
        with gr.Row():
            image_a = gr.Image(label="Image A", type="pil")
            image_b = gr.Image(label="Image B", type="pil")
        threshold = gr.Slider(
            minimum=-1.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
            label="Cosine Similarity Threshold",
        )
        output = gr.JSON(label="Verification Result")
        verify_btn = gr.Button("Verify")
        verify_btn.click(
            fn=verify_faces,
            inputs=[image_a, image_b, threshold],
            outputs=output,
        )
    return demo


def main() -> None:
    args = parse_args()
    cfg: InferenceConfig = load_config(args.config, config_type=InferenceConfig)
    service = EmbeddingService(cfg)
    demo = build_interface(service)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
