"""Download Tiny Shakespeare (~1 MB of text) into data/input.txt."""

import pathlib
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
out = pathlib.Path(__file__).parent / "input.txt"

if not out.exists():
    urllib.request.urlretrieve(URL, out)
text = out.read_text(encoding="utf-8")
print(f"{len(text):,} characters, {len(set(text))} unique -> {out}")
