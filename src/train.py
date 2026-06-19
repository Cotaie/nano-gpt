"""Training entry point for the educational nanoGPT project."""

import argparse
from pathlib import Path

from src.data import DEFAULT_DATA_PATH, DEFAULT_TRAIN_RATIO, load_dataset


def main() -> None:
    """Load the dataset and print basic split information."""
    parser = argparse.ArgumentParser(description="Prepare the nanoGPT dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the input text dataset. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Fraction of text used for training. Default: {DEFAULT_TRAIN_RATIO}",
    )
    parser.add_argument(
        "--show-splits",
        action="store_true",
        help="Print the raw train and validation text after splitting.",
    )
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.data_path, args.train_ratio)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"dataset: {args.data_path}")
    print(f"train characters: {len(dataset.train_text)}")
    print(f"validation characters: {len(dataset.val_text)}")

    if args.show_splits:
        print()
        print("train text:")
        print(repr(dataset.train_text))
        print()
        print("validation text:")
        print(repr(dataset.val_text))


if __name__ == "__main__":
    main()
