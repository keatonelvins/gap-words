"""Embed each row of english_senses.jsonl with Gemma's hidden state at a chosen inner layer.

For each input "{word}: {gloss}", we run one forward pass and capture the
hidden state at the OUTPUT of `model.model.language_model.layers[layer_idx]`
via a forward hook, then take the LAST-token slice. Default layer_frac=2/3
lands late enough to be semantically abstract, early enough to dodge the
final layer's specialization for next-token prediction.

Speed setup
-----------
- `attn_implementation="sdpa"`: FA2/FA3 reject Gemma 4's head_dim=512 (their max is 256).
  See: https://github.com/Dao-AILab/flash-attention/issues/2427
- `torch.compile(model.forward, mode="reduce-overhead")`: v5's supported
  compile path. We don't pass `fullgraph=True` because transformers'
  attention-mask helper has a data-dependent branch that would reject it.
- Every batch is left-padded to a fixed `--max-length` so the compiled
  graph reuses one shape and we don't recompile mid-run.
- A forward hook captures the chosen layer's output instead of
  `output_hidden_states=True`, which would force a graph break.

Output (default `data/`)
------------------------
    embeddings.bin   bf16 [N, hidden_dim] mmap over the raw bytes; row i ↔
                     input row i. Read back with:
                         torch.from_file(path, shared=False, size=N*D,
                                         dtype=torch.bfloat16).view(N, D)
    meta.json        run config + shape
    progress.txt     last completed row index (for crash-resume)
"""

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_float32_matmul_precision("high")

MODEL_ID = "google/gemma-4-31B"

# PromptEOL-style: end the prompt at a position where the next token IS the
# answer. The hidden state at the trailing `"` is the model's prediction of
# the first (and only) token of the one-word summary — that's our embedding.
PROMPT_TEMPLATE = '"{word}: {gloss}" can be summarized in one word as: "'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=Path("data/english_senses.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--layer-frac", type=float, default=2 / 3,
                        help="Layer index = round(layer_frac * num_hidden_layers)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64,
                        help="Every batch is left-padded/truncated to exactly this many tokens")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (useful while debugging)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N input rows")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.out_dir / "embeddings.bin"
    meta_path = args.out_dir / "meta.json"
    progress_path = args.out_dir / "progress.txt"

    n_rows = sum(1 for _ in args.rows.open())
    if args.limit is not None:
        n_rows = min(n_rows, args.limit)

    tok = AutoTokenizer.from_pretrained(
        args.model_id, padding_side="left", truncation_side="left",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    # Gemma 4 is multimodal: text decoder lives at model.model.language_model.layers;
    # hidden_size sits inside config.text_config (Gemma4Config is a composite).
    text_layers = model.model.language_model.layers
    n_layers = len(text_layers)
    hidden_dim = model.config.text_config.hidden_size
    layer_idx = round(args.layer_frac * n_layers)
    print(
        f"model={args.model_id} n_layers={n_layers} hidden_dim={hidden_dim} "
        f"-> layer_idx={layer_idx}",
        file=sys.stderr,
    )

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, output):
        captured["h"] = output  # Gemma4TextDecoderLayer returns the [B, T, D] tensor

    text_layers[layer_idx].register_forward_hook(hook)

    if not args.no_compile:
        model.forward = torch.compile(model.forward, mode="reduce-overhead")

    start = 0
    if progress_path.exists() and emb_path.exists():
        start = int(progress_path.read_text().strip() or 0)
    if start >= n_rows:
        print(f"already complete ({start:,} rows)", file=sys.stderr)
        return

    # mmap a bf16 tensor over the raw file. Creates the file at the right size
    # if it doesn't exist; otherwise opens in place without zeroing existing rows.
    embeddings = torch.from_file(
        str(emb_path), shared=True, size=n_rows * hidden_dim,
        dtype=torch.bfloat16,
    ).view(n_rows, hidden_dim)
    meta_path.write_text(json.dumps({
        "n": n_rows,
        "hidden_dim": hidden_dim,
        "layer_idx": layer_idx,
        "n_layers": n_layers,
        "layer_frac": args.layer_frac,
        "model_id": args.model_id,
        "rows_path": str(args.rows),
        "max_length": args.max_length,
        "compiled": not args.no_compile,
        "prompt_template": PROMPT_TEMPLATE,
    }, indent=2))
    print(f"writing rows [{start:,}, {n_rows:,}) to {emb_path}", file=sys.stderr)

    @torch.inference_mode()
    def embed(texts: list[str]) -> torch.Tensor:
        toks = tok(
            texts, return_tensors="pt",
            padding="max_length", truncation=True, max_length=args.max_length,
        ).to(model.device)
        model(**toks, use_cache=False)
        h = captured["h"]  # [B, T, D] — populated by the forward hook
        return h[:, -1, :].to("cpu", dtype=torch.bfloat16)  # left-pad → last is real

    written = start
    t0 = time.time()
    batch: list[str] = []

    def flush() -> None:
        nonlocal written
        if not batch:
            return
        emb = embed(batch)
        embeddings[written:written + emb.shape[0]] = emb
        written += emb.shape[0]
        progress_path.write_text(str(written))
        batch.clear()

    with args.rows.open() as f:
        for line in islice(f, start, n_rows):
            row = json.loads(line)
            batch.append(PROMPT_TEMPLATE.format(word=row["word"], gloss=row["gloss"]))
            if len(batch) >= args.batch_size:
                flush()
                if (written // args.batch_size) % 50 == 0:
                    rate = (written - start) / max(time.time() - t0, 1e-9)
                    eta_min = (n_rows - written) / max(rate, 1e-9) / 60
                    print(
                        f"written={written:,}/{n_rows:,} "
                        f"({rate:.1f} rows/s, eta {eta_min:.1f}m)",
                        file=sys.stderr,
                    )
        flush()

    print(f"done: {written:,} embeddings -> {emb_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
