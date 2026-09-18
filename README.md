# mlx-moe-runner

Run 100B-class Mixture-of-Experts language models on a RAM-constrained Mac by streaming
experts from the SSD, with all compute on the GPU through [MLX](https://github.com/ml-explore/mlx).

gpt-oss-120b (61 GB of experts) decodes at ~20 tok/s on a 36 GB M5 Max with 14 GB of
experts pinned in unified memory and the rest read on demand from the internal NVMe
(13.7 tok/s with exact routing; 29 tok/s when the page cache is warm). Nothing is swapped
and nothing is dequantized: expert weights stay in their 4-bit blocks from the file to the
Metal kernel.

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

### 4. Pick a configuration

```sh
mlx-moe-runner plan ~/models/gptoss120b.mlxmoe          # probes RAM and SSD, prints the best flags
mlx-moe-runner plan ~/models/gptoss120b.mlxmoe --ram 16 # what-if for another machine
```

`plan` predicts tok/s for pin budgets, cold buffer sizes, routing bias and lookahead
depth from a cost model fitted to the measurements below (per-layer sync overhead, GPU
bandwidth, measured SSD read latency and bandwidth, pinned hit-rate curves) and prints
the recommended `serve` line plus the best exact-routing alternative. Treat it as a
starting point and confirm with `bench`.

### 5. Pinning profiles (optional)

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
| `--io-split` | 1 | sub-reads per expert record (0 = one per matrix; measured neutral on the M5 Max SSD) |
| `--ctx` | 4096 | KV cache length |
| `--chunk` | 32 | prefill chunk (tokens per pass) |

Recommendations from the study, confirmed here: on a 36 GB Mac pin 32/layer with
`--bias 0.5` and no lookahead; on a 16 GB Mac pin 12/layer, `--cold-mb 512`,
`--lookahead 2 --bias 0.5`. Never pin more than ~40% of RAM: the OS starts swapping and
compute slows 3-5x.

## Results

gpt-oss-120b, M5 Max 36 GB, internal NVMe, greedy, 32 experts/layer pinned (14.2 GB),
1.5 GB cold buffer, page cache evicted before each run, 3 prompts x 32 tokens:

| config | tok/s | pinned hit | SSD reads/tok | demand wait ms/tok | TinyGiant (CPU) |
|---|---:|---:|---:|---:|---:|
| exact routing | 13.7 | 60% | 47 | 52 | 9.1 |
| exact + lookahead 1 | 14.7 | 60% | 56 | 44 | 11.1 |
| exact + lookahead 2 | 13.0 | 60% | 67 | 51 | 11.5 |
| **bias 0.5** (default) | **19.5** | 73% | 18 | 21 | 16.9 |
| bias 0.5 + lookahead 1 | 19.9 | 71% | 43 | 25 | 13.9 |
| bias 0.5 + lookahead 2 | 17.4 | 71% | 50 | 29 | 13.8 |

With a warm page cache (right after conversion, or on a machine whose SSD is not the
bottleneck) the same bias configuration measured 29 tok/s; the evicted numbers above are
the steady state. Prefill on the 120b is SSD-bound at roughly 230 MB of expert reads per
prompt token (35-45 tok/s). The per-token compute floor with everything resident is
~22 ms (36 GPU syncs, attention, output head), i.e. ~45 tok/s.

An uncached 13 MB expert read costs ~1.2 ms on this SSD regardless of how it is split
(`scripts/` has the probe), so with ~1 miss per layer the exact-routing wait is ~36 ms per
token; prefetching (lookahead) trades bandwidth for that latency and the routing bias
removes most of the misses outright, which is why bias alone wins at this pin budget.

Other models on the same machine, everything pinned: gpt-oss-20b 50 tok/s (greedy output
identical to llama.cpp for 32 tokens), Qwen3-30B-A3B 16 tok/s all pinned and 20 tok/s
with 16 of 128 experts pinned plus bias and lookahead (the first 19 greedy tokens match
llama.cpp; the rest diverge as its Q6_K tensors are requantized). Qwen's 48 layers of
small experts are bound by per-layer launch and sync overhead rather than bandwidth; that
overhead (~1 ms per layer) is the next thing to attack.

The TinyGiant column is the CPU engine (fused int8 NEON kernels) from the study on the
same machine and model.

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
* Per-layer GPU sync overhead (~0.5-1 ms per layer) caps all-resident decode below what a
  fully lazy MLX graph reaches; it is inherent to streaming (the route must reach the CPU)
  but the kernel count per layer can still be reduced.

## License

MIT.
