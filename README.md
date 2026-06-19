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
  data.py
  model.py
  train.py
  generate.py
```

## Local Setup

Use Python 3.11 or newer.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that PyTorch is installed:

```bash
python -c "import torch; print(torch.__version__)"
```

## Dataset

The first dataset should be a plain text file at:

```txt
data/input.txt
```

That file is ignored by Git so local datasets do not get committed.

For example, after downloading or creating a small text dataset, save it as:

```txt
data/input.txt
```

## Checking The Dataset

After creating `data/input.txt`, check that the project can load and split the
dataset:

```bash
python -m src.train
```

You can also provide a custom dataset path:

```bash
python -m src.train --data-path path/to/input.txt
```

The script prints the number of training and validation characters. The actual
training loop will be added in a later step.

## Roadmap

1. Add local setup instructions.
2. Add dataset loading.
3. Add character tokenization.
4. Train a Bigram language model.
5. Add text generation.
6. Build up toward a tiny GPT model.
