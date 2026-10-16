"""Step 1: character tokenizer + batches for Shaer-GPT.

Fill in every TODO, then run:
    PYTHONIOENCODING=utf-8 uv run --project ../ndim-infra python src/data.py
All checks at the bottom must print OK.
Concepts: CONCEPTS.md sections 1 (tokenization) and 11 (loss: why y is x shifted).
"""

import pathlib

import torch

DATA = pathlib.Path(__file__).parent.parent / "data" / "input.txt"
text = DATA.read_text(encoding="utf-8")

# --- Tokenizer -------------------------------------------------------------
# TODO 1: the sorted list of unique characters in `text`, and its length.
chars = ...
vocab_size = ...

# TODO 2: two dicts: stoi (char -> int) and itos (int -> char).
stoi = ...
itos = ...


def encode(s: str) -> list[int]:
    # TODO 3: string -> list of ints
    ...


def decode(ids: list[int]) -> str:
    # TODO 4: list of ints -> string
    ...


# --- Train / val split -------------------------------------------------------
# TODO 5: encode the whole text into a torch.long tensor, first 90% train, rest val.
data = ...
train_data = ...
val_data = ...


def get_batch(split: str, batch_size: int = 4, block_size: int = 8):
    """Return x, y of shape (batch_size, block_size); y[i, t] is the token after x[i, t]."""
    # TODO 6: pick batch_size random start positions, slice x and y (y = x shifted by one).
    ...


# --- Self-check (don't edit) ------------------------------------------------
if __name__ == "__main__":
    verse = "قِفا نَبكِ مِن ذِكرى حَبيبٍ وَمَنزِلِ"  # if you built --strip-diacritics data, use "قفا نبك من ذكرى حبيب ومنزل"
    print(f"vocab_size = {vocab_size}  (expect 72 with diacritics, 63 without)")
    assert decode(encode(verse)) == verse, "round-trip failed"
    print("OK round-trip:", encode(verse)[:10], "...")
    assert len(train_data) + len(val_data) == len(text) and len(train_data) == int(0.9 * len(text))
    print(f"OK split: {len(train_data):,} train / {len(val_data):,} val")
    torch.manual_seed(1337)
    x, y = get_batch("train")
    assert x.shape == y.shape == (4, 8), f"shape {x.shape}"
    assert torch.equal(x[0, 1:], y[0, :-1]), "y is not x shifted by one"
    print("OK batch:", x.shape)
    print("first row as text:", repr(decode(x[0].tolist())), "->", repr(decode(y[0].tolist())))
