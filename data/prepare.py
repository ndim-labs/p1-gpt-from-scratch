"""Build the training text for the Arabic poetry GPT ("Shaer-GPT").

Source: arbml/ashaar on Hugging Face (~254k Arabic poems scraped from aldiwan.net,
with poet, era and meter). Licence isn't declared, so we use it for study only and
never commit the data.

    python data/prepare.py                      # classical eras, ~Shakespeare-sized
    python data/prepare.py --eras all           # everything (~100+ MB of text)
    python data/prepare.py --stats              # list eras and sizes, then exit

Writes data/input.txt: one verse per line, a blank line between poems.
Also keeps data/shakespeare.txt as a sanity check against Karpathy's numbers.
"""

import argparse
import ast
import pathlib
import re
import urllib.request
from collections import Counter

import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).parent
BASE = "https://huggingface.co/datasets/arbml/ashaar/resolve/main/data/"
SHARDS = ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Pre-Islamic + early Islamic + Umayyad: the "golden age" of classical poetry.
CLASSICAL = ["قبل الإسلام", "العصر الجاهلي", "المخضرمين", "العصر الإسلامي", "العصر الأموي"]

DIACRITICS = re.compile("[" + chr(0x064B) + "-" + chr(0x0652) + chr(0x0670) + chr(0x0640) + "]")  # harakat U+064B-0652, dagger alif, tatweel


def download(url, out):
    if not out.exists():
        print(f"downloading {out.name} ...")
        urllib.request.urlretrieve(url, out)
    return out


def load_table():
    cols = ["poem verses", "poet era", "poem meter", "poet name"]
    tables = [pq.read_table(download(BASE + s, HERE / s), columns=cols).to_pylist() for s in SHARDS]
    return [row for t in tables for row in t]


def verses(row):
    v = row["poem verses"]
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [s.strip() for s in v if s and s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eras", default="classical", help="'classical', 'all', or comma-separated era names")
    ap.add_argument("--strip-diacritics", action="store_true", help="remove harakat (smaller vocab, easier task)")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    rows = load_table()
    if args.stats:
        for era, n in Counter(r["poet era"] for r in rows).most_common():
            print(f"{n:>7}  {era}")
        return

    eras = None if args.eras == "all" else (CLASSICAL if args.eras == "classical" else args.eras.split(","))
    poems = ["\n".join(verses(r)) for r in rows if eras is None or r["poet era"] in eras]
    text = "\n\n".join(p for p in poems if p)
    if args.strip_diacritics:
        text = DIACRITICS.sub("", text)

    out = HERE / "input.txt"
    out.write_text(text, encoding="utf-8")
    print(f"{len(poems):,} poems, {len(text):,} characters, {len(set(text))} unique -> {out}")

    download(SHAKESPEARE, HERE / "shakespeare.txt")


if __name__ == "__main__":
    main()
