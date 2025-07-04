# ---------------------------------------------
# File: README.md
# ---------------------------------------------
# Mini GPT Shakespeare

A **character-level Mini GPT** built and trained on `tinyshakespeare.txt` for learning GPT internals.

## Features
- Custom GPT architecture using PyTorch
- Trains on Shakespeare text
- Generates Shakespeare-style text

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Place `tinyshakespeare.txt` in the project folder. Download from [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

## Training
Run:
```bash
python mini_gpt_train.py
```

## Example Output
> ELOUCESTER:
> I am looke, merry dishe counter,--

## License
MIT
