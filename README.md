# missing

Find concepts in an LM's latent representation that have no English word.

Two scripts — run in order:

## 1. Build the English sense dataset

Streams the [kaikki.org](https://kaikki.org/) wiktextract dump and emits one JSONL row per English leaf sense.

```bash
uv run python -m missing.build_dataset
# -> data/english_senses.jsonl  (≈1.02M rows from a 10.6M-entry dump, ~4 min)
```

Each row: `{word, pos, sense_idx, tags, topics, categories, gloss, gloss_path}`. The PromptEOL string fed to the LM is built in `embed.py` as `'"{word}: {gloss}" can be summarized in one word as: "'`, so the template can be swapped without rebuilding the dataset. Alias / inflection senses (`form-of`, `alt-of`, `abbreviation`, `initialism`, `acronym`, `misspelling`, plus `Synonym of …` glosses) are dropped at build time; see the worklog below.

Smoke test: `--limit 100 --out data/smoke.jsonl`. Or read from a local gz: `--src data/raw-wiktextract-data.jsonl.gz`.

## 2. Embed each sense with Gemma-4-31B

For each row, runs one forward pass and captures the hidden state at the output of layer `round(2/3 * num_hidden_layers) = 40` of 60 (last token, bf16).

```bash
uv run python -m missing.embed
# -> data/embeddings.bin   bf16 [N, 5376] mmapped via torch.from_file
#    data/meta.json        run config
#    data/progress.txt     resume marker
```

To read the embeddings back:

```python
emb = torch.from_file(
    "data/embeddings.bin", shared=False, size=N * 5376, dtype=torch.bfloat16,
).view(N, 5376)
```

Crash-resume: rerun the same command. Smoke test: `--limit 200 --out-dir data/smoke`.

Knobs: `--layer-frac`, `--batch-size`, `--max-length`, `--no-compile`, `--model-id`.

---

## Worklog

Decisions made while building the pipeline, with the data that drove them.

### Dataset filter — drop alias / inflection senses (~42% reduction)

A 5,000-row spot-check found ~40% of English Wiktionary senses are not real definitions but cross-refs ("plural of X", "Abbreviation of Y", "Synonym of Z", ...). Their embeddings encode the literal alias string, not the meaning of the underlying word — noise for our latent-space search. We drop a sense if any of its `tags` is in `{form-of, alt-of, abbreviation, initialism, acronym, misspelling}`, or its leaf gloss starts with `Synonym of` / `Synonym for` (these slip through tagless).

| | rows |
|---|---|
| English entries in 10.6M-line kaikki dump | 1,752,263 |
| After alias / inflection filter | 1,022,153 (−41.7 %) |

We considered keeping plurals (plurality is "its own thing") but the embedding of `"dogs: plural of dog"` is a near-duplicate of the singular's vector — adds no new concept directions, and irregular plurals (`children`, `feet`) get picked up via their primary entries anyway.

### Token length — `max_length=64` covers 99.81 % of rows

Tokenized all 1.75M rows with `GemmaTokenizer`:

```
n = 1,752,263       p50=12  p90=25  p95=31  p99=45  max=216
> 32 tokens: 70,349   ( 4.01 %)
> 48 tokens: 12,989   ( 0.74 %)
> 64 tokens:  3,347   ( 0.19 %)
> 96 tokens:    411   ( 0.02 %)
> 128 tokens:    89   ( 0.01 %)
```

Median 12, p99 45. Original `max_length=128` was ~3× larger than needed; halving to 64 truncates only 3,347 rows (0.19 %) and roughly doubles forward throughput (per-batch time scales linearly with seq_len, so ~6.6 h → ~3 h on the full set). Going to 32 truncates 4 %, too aggressive.

### Prompt template — PromptEOL with explicit summarization

Vanilla last-token of `"{word}: {gloss}"` encodes "what comes next after this fragment", which is often surface continuation, not meaning. Wrapping with a PromptEOL-style suffix forces the model into "summarize this concept" mode and the trailing-token hidden state becomes a meaning-density vector. We use:

```
"{word}: {gloss}" can be summarized in one word as: "
```

The trailing `"` is the last input token; its hidden state is the model's prediction for the first token of the (quoted) one-word answer — that's the embedding we capture. Worth it especially for ambiguous entries: `Bowell: A surname from Norman.` and `what do you say: Used to remind a child to say a polite expression.` both push toward useful concept directions (Norman-surname, courtesy-prompt) instead of literal-string continuations. Tokenizer is set to `truncation_side="left"` so over-budget rows lose front-of-gloss context but keep the prompt suffix intact.

Prompt overhead: ~9 tokens (`" can be summarized in one word as: "`). p99 raw is 45 tokens, so 99 % of rows fit in `max_length=64` with the wrap.

### Embedding model & layer — Gemma-4-31B base, layer 40 of 60 (≈ 2/3 depth)

Base, not instruction-tuned, so completions are grounded in the training distribution rather than chat conventions. Layer choice is the standard ~2/3-depth heuristic — late enough to be semantically abstract, early enough to dodge the final layer's specialization for next-token prediction. Captured via a forward hook on `model.model.language_model.layers[40]` so `output_hidden_states=True` (which would force a `torch.compile` graph break) is avoided.

### Attention backend — SDPA only (no FA2/FA3)

Gemma 4 uses `global_head_dim=512`. FlashAttention 2 and the kernels-community FA3 kernel both reject `head_dim > 256` (`RuntimeError: FlashAttention forward only supports head dimension at most 256`). The Gemma 4 model docs use `attn_implementation="sdpa"` in every example for the same reason. SDPA is the default-best on H100 anyway when FA isn't available.

### Storage format — bf16 mmap via `torch.from_file`

The model runs in bf16; late-layer activations regularly exceed fp16's max (~65,504), so storing in fp16 would silently saturate to inf. bf16 has the same range as fp32. NumPy doesn't have a native bf16 dtype, so we use `torch.from_file(..., dtype=torch.bfloat16)` to mmap the raw bytes as a real bf16 tensor — read and write are both one-liners, no `uint16` ceremony. safetensors was considered but doesn't support append-style writing, which we need for crash-resume.

### Throughput is compute-bound; batch size and `kernelize` don't help

Bench results on H100 (`max_length=128`, 30 timed batches after warmup):

| batch | compile | rows/s | ms/batch | peak alloc | reserved |
|---|---|---|---|---|---|
| 16  | on  | 73.6 | 217  | 62.6 GB | 64.6 GB |
| 32  | on  | 74.1 | 432  | 62.6 GB | 66.8 GB |
| 64  | on  | 75.3 | 850  | 62.6 GB | 71.0 GB |
| 128 | on  | 76.5 | 1672 | 62.7 GB | 79.4 GB ← near-OOM |
| 16  | off | 51.6 | 310  | 64.8 GB | 65.4 GB |
| 64  | off | 53.9 | 1187 | 71.3 GB | 74.1 GB |

Per-row time is flat across batch sizes (~13 ms) — pure compute-bound, no win from larger batches. `torch.compile(mode="reduce-overhead")` gives a clean **1.43×** at every size; worth the ~60 s warmup. We also tested `kernels.kernelize(model, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)`: zero impact (73.2 vs 73.2 rows/s) because the Gemma 4 modeling code in transformers 5.8 doesn't decorate its RMSNorm / SwiGLU layers with `@use_kernel_forward_from_hub`. Subsequently dropped the `kernels` dependency.

The only real lever left for >1.43× is fp8 / int8 weight quantization (~2× extra), which would shift the very representations we're trying to study — left for future.

