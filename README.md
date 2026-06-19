# nano-gpt

An educational GPT-style language model built from scratch with Python and
PyTorch.

The first version starts with a small character-level language model, then grows
step by step from a Bigram model into a tiny GPT-style transformer.

## Project Status

This project is in the early setup phase.

Current focus:

- Define the initial project spec.
- Create the project skeleton.
- Add setup instructions.
- Build the first Bigram language model before implementing GPT components.

## Project Structure

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

## Dataset

The first dataset should be a plain text file at:

```txt
data/input.txt
```

That file is ignored by Git so local datasets do not get committed.

## Roadmap

1. Add local setup instructions.
2. Add dataset loading.
3. Add character tokenization.
4. Train a Bigram language model.
5. Add text generation.
6. Build up toward a tiny GPT model.
