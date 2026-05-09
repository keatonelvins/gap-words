"""Stream the Wiktionary (kaikki.org wiktextract) dump and emit one row per leaf sense.

Input
-----
Gzipped JSONL from `https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz`.
Each line is one wiktextract entry. The same surface word can appear as multiple
top-level entries (one per etymology / part-of-speech section).

Fields we read per entry:

    word          surface form, e.g. "bank"
    lang_code     ISO code, e.g. "en"; entries not matching --lang-code are skipped
    pos           part-of-speech, e.g. "noun"
    senses[]      one item per Wiktionary sense, each with:
      glosses[]     cumulative path through the nested-list sense tree.
                    Wiktionary markup like
                        # parent
                        ## child
                    becomes glosses=["parent"] and glosses=["parent", "child"]
                    on two separate sense entries. So glosses[-1] is the leaf
                    definition and glosses[:-1] is its ancestor path. See
                    https://github.com/tatuylonen/wiktextract/blob/master/docs/new_extractor_guide.md
      tags[]        linguistic tags: "transitive", "obsolete", ...
      topics[]      subject areas: "computing", "biology", ...
      categories[]  wiki categories: "English terms with quotations", ...

Filters
-------
We drop alias / inflection senses that don't carry independent meaning — they
embed as the literal alias string ("plural of nonhomosexual") rather than the
meaning of the underlying word. ~40% of English rows are dropped.

A sense is dropped when:
- its `tags` include any of `form-of`, `alt-of`, `abbreviation`, `initialism`,
  `acronym`, `misspelling` (the structural wiktextract markers covering
  inflections, alternative forms, abbreviations, etc.), OR
- its leaf gloss starts with `Synonym of` / `Synonym for` (which slip through
  tagless).

Output
------
JSONL, one row per leaf sense:

    {
      "word": str, "pos": str, "sense_idx": int,
      "tags": [...], "topics": [...], "categories": [...],
      "gloss":      "<leaf definition>",
      "gloss_path": [<ancestor glosses>],
    }

Polysemous words ("bank" the institution vs. "bank" the riverside) get one row
per leaf sense, so each row maps cleanly to a single embedding downstream.
`sense_idx` is unique within an entry but not across the multiple etymologies
of the same word — pair it with (word, gloss) or file position for a global key.
"""

import argparse
import gzip
import json
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

WIKTIONARY_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
ALIAS_TAGS = frozenset({
    "form-of", "alt-of", "abbreviation", "initialism", "acronym", "misspelling",
})
ALIAS_GLOSS_PREFIXES = ("Synonym of ", "Synonym for ")


def stream_lines(src: str) -> Iterator[bytes]:
    """Yield raw bytes lines from a gzipped jsonl URL or local file."""
    if src.startswith(("http://", "https://")):
        resp = urllib.request.urlopen(src)
        yield from gzip.GzipFile(fileobj=resp)
    else:
        with gzip.open(src, "rb") as f:
            yield from f


def iter_rows(entry: dict, lang_code: str) -> Iterator[dict]:
    """Emit one row per leaf sense for entries matching `lang_code`."""
    if entry.get("lang_code") != lang_code or not entry.get("word"):
        return
    for sense_idx, sense in enumerate(entry.get("senses", [])):
        glosses = sense.get("glosses") or []
        if not glosses or not glosses[-1]:
            continue
        leaf = glosses[-1]
        tags = sense.get("tags") or []
        if any(t in ALIAS_TAGS for t in tags) or leaf.startswith(ALIAS_GLOSS_PREFIXES):
            continue
        yield {
            "word": entry["word"],
            "pos": entry.get("pos"),
            "sense_idx": sense_idx,
            "tags": tags,
            "topics": sense.get("topics") or [],
            "categories": sense.get("categories") or [],
            "gloss": leaf,
            "gloss_path": glosses[:-1],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=WIKTIONARY_URL,
                        help="Gzipped jsonl URL or local file path")
    parser.add_argument("--out", type=Path, default=Path("data/english_senses.jsonl"))
    parser.add_argument("--lang-code", default="en")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after writing this many rows (smoke test)",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    t0 = time.time()

    def log(prefix: str) -> None:
        dt = max(time.time() - t0, 1e-9)
        print(
            f"{prefix} in={n_in:,} out={n_out:,} ({n_in / dt:,.0f} lines/s)",
            file=sys.stderr,
        )

    with args.out.open("w", encoding="utf-8") as f:
        for raw in stream_lines(args.src):
            n_in += 1
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for row in iter_rows(entry, args.lang_code):
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
                n_out += 1
                if args.limit is not None and n_out >= args.limit:
                    log(f"hit limit ({args.out}):")
                    return
            if n_in % 100_000 == 0:
                log("progress:")
    log(f"done ({args.out}):")


if __name__ == "__main__":
    main()
