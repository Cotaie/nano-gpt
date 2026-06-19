"""Model definitions for the educational nanoGPT project."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """Predict the next token using only the current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-token logits and optional cross-entropy loss."""
        logits = self.token_embedding_table(inputs)

        if targets is None:
            return logits, None

        batch_size, sequence_length, vocab_size = logits.shape
        logits_for_loss = logits.view(batch_size * sequence_length, vocab_size)
        targets = targets.view(batch_size * sequence_length)
        loss = F.cross_entropy(logits_for_loss, targets)
        return logits, loss
