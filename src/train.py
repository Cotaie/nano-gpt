"""Training entry point for the educational nanoGPT project."""

import argparse
from pathlib import Path

import torch

from src.data import DEFAULT_DATA_PATH, DEFAULT_TRAIN_RATIO, load_dataset
from src.model import BigramLanguageModel
from src.tokenizer import CharacterTokenizer


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
    parser.add_argument(
        "--show-vocab",
        action="store_true",
        help="Print the character vocabulary used by the tokenizer.",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Print tokenizer mappings and encoded train/validation token IDs.",
    )
    parser.add_argument(
        "--show-bigram-matrix",
        action="store_true",
        help="Print the randomly initialized Bigram score matrix.",
    )
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.data_path, args.train_ratio)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"dataset: {args.data_path}")
    print(f"train characters: {len(dataset.train_text)}")
    print(f"validation characters: {len(dataset.val_text)}")

    tokenizer = CharacterTokenizer.from_text(dataset.train_text + dataset.val_text)
    train_tokens = tokenizer.encode(dataset.train_text)
    val_tokens = tokenizer.encode(dataset.val_text)

    print(f"vocabulary size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(train_tokens)}")
    print(f"validation tokens: {len(val_tokens)}")

    if tokenizer.decode(train_tokens) != dataset.train_text:
        raise SystemExit("error: train text failed tokenizer round trip")
    if tokenizer.decode(val_tokens) != dataset.val_text:
        raise SystemExit("error: validation text failed tokenizer round trip")

    if len(train_tokens) < 2:
        raise SystemExit("error: training split must contain at least two tokens")

    inputs = torch.tensor(train_tokens[:-1], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(train_tokens[1:], dtype=torch.long).unsqueeze(0)
    model = BigramLanguageModel(tokenizer.vocab_size)
    logits, loss = model(inputs, targets)

    if loss is None:
        raise SystemExit("error: model did not return a loss")

    print(f"bigram input shape: {tuple(inputs.shape)}")
    print(f"bigram logits shape: {tuple(logits.shape)}")
    print(f"initial bigram loss: {loss.item():.4f}")

    if args.show_bigram_matrix:
        print()
        print("bigram score matrix:")
        print(model.token_embedding_table.weight.detach())

    if args.show_splits:
        print()
        print("train text:")
        print(repr(dataset.train_text))
        print()
        print("validation text:")
        print(repr(dataset.val_text))

    if args.show_vocab:
        print()
        print("vocabulary:")
        print(repr("".join(tokenizer.chars)))

    if args.show_tokens:
        print()
        print("character to token ID:")
        for char, token in tokenizer.stoi.items():
            print(f"{char!r}: {token}")

        print()
        print("train token IDs:")
        print(train_tokens)

        print()
        print("validation token IDs:")
        print(val_tokens)


if __name__ == "__main__":
    main()
