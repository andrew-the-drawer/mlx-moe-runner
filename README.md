# mlx-moe-runner

Run 100B-class Mixture-of-Experts language models on a RAM-constrained Mac by streaming
experts from the SSD, with all compute on the GPU through [MLX](https://github.com/ml-explore/mlx).

gpt-oss-120b (61 GB of experts) decodes at ~29 tok/s on a 36 GB M5 Max with 14 GB of
experts pinned in unified memory and the rest read on demand from the internal NVMe. Nothing
is swapped and nothing is dequantized: expert weights stay in their 4-bit blocks from the
file to the Metal kernel.

The engine implements the algorithms measured in the
[TinyGiant study](../tinygiant/docs/study-100b-on-a-mac.md) (two-tier expert store with
`F_NOCACHE` streaming, calibrated pinning, multi-layer lookahead prefetch, cache-aware
routing bias) as a single compiled binary with an OpenAI-compatible server, installable
with Homebrew.

## Install

```sh
brew tap lantrungseo/mlx-moe-runner https://github.com/lantrungseo/mlx-moe-runner
brew install mlx-moe-runner
```

or from a checkout:

```sh
brew install --build-from-source ./Formula/mlx-moe-runner.rb
```

Dependencies (all from Homebrew): `mlx` (the C++ library), `pcre2`, and at build time
`cmake`, `pkgconf`, `nlohmann-json`, `cpp-httplib`. Apple silicon and macOS 14+ only.

To build without Homebrew's formula:

```sh
brew install mlx pcre2 cmake pkgconf nlohmann-json cpp-httplib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Usage

### 1. Convert a GGUF once

```sh
mlx-moe-runner convert ~/models/gpt-oss-120b-MXFP4.gguf ~/models/gptoss120b.mlxmoe
```

This writes a directory the runner streams from:

| file | content |
|---|---|
| `experts_NNN.bin` | layer NNN, one fixed-size record per expert, parts page-aligned |
| `base.safetensors` | attention, norms, routers, embeddings, output head (MLX quantized) |
| `config.json` | model geometry, expert layout, quantization, static expert ranking |
| `tokenizer.json` | HF-format byte-level BPE tokenizer built from the GGUF vocabulary |

Quantized ggml types are repacked into MLX's layouts without requantization where an exact
mapping exists (MXFP4 → `mxfp4`, Q8_0 and Q4_0 and Q4_K → `affine`); Q5_K/Q6_K and other
k-quants are dequantized and requantized to 8-bit affine (`--lossy-bits`). gpt-oss-120b
converts in about 70 s.

Supported architectures: `gpt-oss` (harmony chat format, attention sinks, sliding window,
YaRN) and `qwen3moe` / `qwen2moe` (ChatML, QK-norm). Other GGUF MoE models with the same
tensor naming (`ffn_{gate,up,down}_exps`, `ffn_gate_inp`) load but are untested.

### 2. Run

```sh
mlx-moe-runner run ~/models/gptoss120b.mlxmoe -p "Explain attention sinks." -n 256 --stats
mlx-moe-runner run ~/models/gptoss120b.mlxmoe --raw -p "The key insight about MoE models is" --temp 0
```

`run` applies the model's chat template by default (`--raw` for plain completion), streams
tokens to stdout with reasoning dimmed, and prints prefill/decode speed. `--stats` adds the
per-token breakdown (GPU sync, expert hit rates, SSD reads, routing swaps).

### 3. Serve

```sh
mlx-moe-runner serve ~/models/gptoss120b.mlxmoe --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 200, "stream": true, "reasoning_effort": "low"}'
```

Endpoints: `POST /v1/chat/completions` (streaming and non-streaming, `reasoning_content`
carries gpt-oss analysis / Qwen `<think>` output), `POST /v1/completions`, `GET /v1/models`,
`GET /v1/stats`, `GET /health`. Requests are served one at a time; the KV cache is reused
across requests that share a prompt prefix (multi-turn chats only prefill the new turn).

### 4. Pinning profiles (optional)

```sh
mlx-moe-runner profile ~/models/gptoss120b.mlxmoe --text corpus.txt --out profile.json
mlx-moe-runner serve ~/models/gptoss120b.mlxmoe --profile profile.json
```

By default `run` calibrates the pinned set on the prompt (8 greedy tokens with nothing
pinned) and `serve` on a generic chat prompt, backfilled from the static router ranking
computed at convert time. A profile built from real traffic gives a better pinned set.

## Tuning

| flag | default | effect |
|---|---|---|
| `--pin N` | auto | experts per layer kept in unified memory. Auto keeps the whole footprint under `--ram-fraction` (0.5) of RAM. 32/layer = 14.2 GB for gpt-oss-120b. |
| `--cold-mb` | 1536 | bounded LRU buffer for streamed experts (also caps prefill chunk size) |
| `--bias B` | 0.5 | cache-aware routing: add B to resident experts' logits before top-k. Gate weights still come from the unbiased logits. ~1% perplexity at 0.5; `--exact` for exact routing |
| `--lookahead J` | 0 | prefetch experts predicted J layers ahead from the residual stream |
| `--extra X` | 0 | extra prefetch candidates per layer (costs bandwidth) |
| `--io-threads` | 8 | parallel SSD readers |
| `--ctx` | 4096 | KV cache length |
| `--chunk` | 32 | prefill chunk (tokens per pass) |

Recommendations from the study, confirmed here: on a 36 GB Mac pin 32/layer with
`--bias 0.5` and no lookahead; on a 16 GB Mac pin 12/layer, `--cold-mb 512`,
`--lookahead 2 --bias 0.5`. Never pin more than ~40% of RAM: the OS starts swapping and
compute slows 3-5x.

## Results

gpt-oss-120b, M5 Max 36 GB, internal NVMe, greedy, 32 experts/layer pinned (14.2 GB),
1.5 GB cold buffer, page cache evicted before each run, 3 prompts x 32 tokens:

RESULTS_TABLE

For reference the TinyGiant CPU engine (fused int8 NEON kernels) reached 9.1 tok/s exact
and 16.9 tok/s with bias 0.5 on the same machine and model. gpt-oss-20b with everything
pinned decodes at ~50 tok/s and matches llama.cpp's greedy output token for token.

## How it works

See [docs/design.md](docs/design.md). In short:

1. **Two-tier expert store.** Pinned experts are loaded once into one contiguous MLX array
   per layer and matrix, so every pinned expert a token touches is served by a single
   `gather_qmm` kernel. Cold experts are read with one `preadv` + `F_NOCACHE` straight
   into MLX allocator buffers (no copies, no page cache) by a pool of I/O threads driven by
   a priority queue: demand reads first, then prefetches by predicted distance, with LRU
   eviction under a fixed budget.
2. **Per-layer routing sync.** Attention and the router run lazily on the GPU; the engine
   evaluates the router logits, selects experts on the CPU (with the optional resident
   bias), issues all of a layer's misses at once, and builds the MoE graph while the SSD
   works.
3. **Lookahead prefetch.** The routers of layers l+1..l+J, with their RMS-norm weights
   folded in, are applied to the residual at layer l in one matmul; predicted experts are
   requested at a priority equal to their distance.
4. **Everything quantized end to end.** Activations are float32; weights are consumed in
   their packed 4-bit/8-bit form by MLX's quantized kernels.

## Limitations

* One generation at a time (no batching across requests).
* Tool calls are not parsed from harmony `commentary` channels (they are returned as
  `reasoning_content`).
* Speculative decoding and the token-id prior from the study are not implemented; the
  study found neither pays for itself.
* Only GGUF inputs; MLX-community safetensors checkpoints are not read directly.

## License

MIT.
