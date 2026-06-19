# Nano GPT Project Spec

## Status

Version: 0.1
Purpose: Learning project

## Goal

Build a small GPT-style language model from scratch using Python and PyTorch.

The main goal is educational: understand how a language model is trained to
predict the next token and how transformer-based GPT models generate text.

## Initial Scope

The first version will implement a character-level language model.

The model will:

- Load a plain text dataset.
- Build a character vocabulary from the dataset.
- Encode text into integer token IDs.
- Train a model to predict the next character.
- Generate new text from a starting prompt.

## Learning Path

The project will start simple and grow step by step:

1. Implement dataset loading and character encoding.
2. Train a Bigram language model.
3. Add token and position embeddings.
4. Add single-head self-attention.
5. Add multi-head self-attention.
6. Add transformer blocks.
7. Train a small GPT-style model.
8. Add text generation and checkpoint saving.

## Tech Stack

- Python
- PyTorch
- tqdm

## Initial Project Structure

```txt
SPEC.md
README.md
requirements.txt
data/
src/
  model.py
  train.py
  generate.py
```

## First Dataset

Use a small plain-text dataset, preferably Tiny Shakespeare.

The dataset should live at:

```txt
data/input.txt
```

## Non-Goals For Now

The first version will not include:

- Large-scale training
- GPU optimization beyond basic PyTorch support
- Distributed training
- Web UI
- API server
- BPE/tokenizer libraries
- Fine-tuning existing pretrained models

## Success Criteria

The first working version is successful when:

- Training runs locally.
- Loss decreases during training.
- The model can generate text after training.
- The code is readable enough for a beginner to follow.
