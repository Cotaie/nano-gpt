"""Character-level tokenization helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterTokenizer:
    """Encode text as character IDs and decode IDs back to text."""

    chars: tuple[str, ...]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        """Build a tokenizer vocabulary from the unique characters in text."""
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")

        chars = tuple(sorted(set(text)))
        stoi = {char: index for index, char in enumerate(chars)}
        itos = {index: char for char, index in stoi.items()}
        return cls(chars=chars, stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters in the vocabulary."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of integer token IDs."""
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            char = exc.args[0]
            raise ValueError(f"Character not in vocabulary: {char!r}") from exc

    def decode(self, tokens: list[int]) -> str:
        """Convert integer token IDs back into text."""
        try:
            return "".join(self.itos[token] for token in tokens)
        except KeyError as exc:
            token = exc.args[0]
            raise ValueError(f"Token ID not in vocabulary: {token}") from exc
