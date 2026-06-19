"""Dataset loading helpers for plain text language modeling."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_PATH = Path("data/input.txt")
DEFAULT_TRAIN_RATIO = 0.9


@dataclass(frozen=True)
class DatasetSplit:
    """Raw text split used before tokenization is implemented."""

    train_text: str
    val_text: str


def load_text(path: str | Path = DEFAULT_DATA_PATH) -> str:
    """Read a UTF-8 text dataset from disk."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}. "
            "Create a plain text dataset at data/input.txt."
        )

    text = data_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Dataset file is empty: {data_path}")

    return text


def split_text(text: str, train_ratio: float = DEFAULT_TRAIN_RATIO) -> DatasetSplit:
    """Split raw text into train and validation portions."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be greater than 0.0 and less than 1.0")

    if len(text) < 2:
        raise ValueError("Dataset must contain at least two characters to split")

    split_index = int(len(text) * train_ratio)
    split_index = min(max(split_index, 1), len(text) - 1)

    return DatasetSplit(
        train_text=text[:split_index],
        val_text=text[split_index:],
    )


def load_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> DatasetSplit:
    """Load raw text from disk and split it into train and validation sets."""
    return split_text(load_text(path), train_ratio)
