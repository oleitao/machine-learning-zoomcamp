import argparse
from pathlib import Path

from xg_futebol.model import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training xG models (StatsBomb)")

    parser.add_argument(
        "model",
        choices=["logreg", "random_forest"],
        help="Model to train",
    )

    parser.add_argument(
        "--events-dir",
        type=str,
        default="data/statsbomb-open-data/data/events",
        help="Folder with StatsBomb event files (.json)",
    )

    parser.add_argument(
        "--calibration",
        type=str,
        default="none",
        choices=["none", "sigmoid", "isotonic"],
        help="Calibration type: none, sigmoid (Platt) or isotonic",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    events_root = Path(args.events_dir)
    if not events_root.exists():
        raise FileNotFoundError(f"Events folder not found: {events_root}")

    metrics, model_path = train_model(
        events_root=events_root,
        model_name=args.model,
        calibration=args.calibration,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Training completed.")
    print(f"Model saved to: {model_path}")
    print("Metrics on test set:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
