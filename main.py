"""Convenience entrypoint for the FaceNet project."""


def main() -> None:
    message = (
        "FaceNet project CLI:\n"
        "  - Train: python scripts/train.py --config configs/train.yaml\n"
        "  - Evaluate: python scripts/evaluate.py --checkpoint <path>\n"
        "  - Export: python scripts/export.py --checkpoint <path>\n"
        "  - Serve: python scripts/serve.py --config configs/inference.yaml\n"
    )
    print(message)


if __name__ == "__main__":
    main()
