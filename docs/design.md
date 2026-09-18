# Design

`mlx-moe-runner` is a C++20 program linked against the MLX C++ library. It keeps the
non-expert weights and a chosen subset of experts in unified memory and streams the rest
from an expert-contiguous file layout on SSD. This document describes the pieces and the
decisions behind them; the measurements that motivated the algorithms are in the TinyGiant
study.

## Source map

| path | role |
|---|---|
| `src/gguf/` | mmap GGUF reader; ggml block decoders (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/MXFP4/F16/BF16) and exact repackers into MLX layouts |
| `src/model/` | `ModelSpec` (geometry + conventions from GGUF metadata), `ExpertLayout` (on-disk record format), `BaseWeights` (non-expert arrays, `QLinear`) |
| `src/convert/` | GGUF → model directory converter |
| `src/io/` | `ColdLoader` (priority queue, I/O threads, `F_NOCACHE` preadv, LRU budget) and `ExpertCache` (pinned banks + cold loader) |
| `src/engine/` | `Engine` (forward pass, KV cache, generation), `Router` (top-k with bias), `Lookahead`, `Sampler` |
| `src/tokenizer/` | byte-level BPE from HF `tokenizer.json` (PCRE2 pre-tokenizer), chat formats and output parsing |
| `src/server/` | cpp-httplib OpenAI-compatible API |
| `tests/` | unit tests for repackers, tokenizer, loader, router, layout |

## Model directory

`config.json` holds the `ModelSpec`, the `ExpertLayout`, the quantization of each base
tensor role and a static expert ranking. `base.safetensors` holds every non-expert tensor
as MLX arrays: quantized matrices as `<name>.weight` (packed `uint32`), `.scales`
(`float16`, or `uint8` e8m0 for mxfp4) and `.biases`; norms, biases, routers and sinks as
`float32`.

Each `experts_NNN.bin` is `n_experts` records of `record_bytes`. A record is a sequence of
parts (`gate.weight`, `gate.scales`, `gate.biases`, `up.*`, `down.*`), each starting on a
4096-byte boundary; the record itself is a multiple of 16 KiB. Alignment matters because
uncached reads on macOS are only truly uncached when offsets and lengths are page aligned,
and because the parts are read straight into separate MLX buffers.

### Exact repacking

MLX's affine format is `w = scale * q + bias` with unsigned `q`, `float16` scale/bias per
group of 32, and elements packed little-endian (nibble *i* of the row is element *i*).
MLX's `mxfp4` format is an `e8m0` exponent per 32 values and e2m1 nibbles in the same
element order. ggml's blocks map onto these without loss of information:

| ggml | MLX | mapping |
|---|---|---|
| MXFP4 (17 B / 32) | mxfp4 g32 | same exponent; low nibbles are elements 0-15, high nibbles 16-31 |
| Q8_0 (34 B / 32) | affine 8-bit g32 | `q_u8 = q_s8 + 128`, `scale = d`, `bias = -128 d` |
| Q4_0 (18 B / 32) | affine 4-bit g32 | `scale = d`, `bias = -8 d` |
| Q4_K (144 B / 256) | affine 4-bit g32 | per sub-block `scale = d * sc`, `bias = -dmin * m` (6-bit sub-block scales become f16, the only rounding) |

Q5_K and Q6_K use 16- or 32-value sub-blocks with signed values that have no exact affine
equivalent; they are dequantized and requantized with MLX at 8 bits (`--lossy-bits`).

`mx::dequantize` computes in `float16` and is only approximate; the quantized matmul
kernels with `float32` activations are exact (verified in `tests/test_quants.cpp` by
multiplying with an identity matrix). Activations therefore stay `float32` throughout.

## Engine

### Forward pass

`forward_rows(tokens, start_pos)` runs *k* tokens through the model in one pass (prefill
chunks and single-token decode share the code):

1. embeddings: rows of the quantized embedding table, dequantized on the GPU;
2. per layer: RMS-norm, fused Q/K/V projections, optional QK-norm, RoPE (`fast::rope` with
   per-dimension periods, YaRN-scaled for gpt-oss, and the YaRN `mscale` applied to q and
   k), KV write (`slice_update` into preallocated `[1, H_kv, ctx, d]` arrays), fused
   `scaled_dot_product_attention` with attention sinks and a causal + sliding-window mask
   for k > 1 (a K/V slice for k = 1), output projection;
3. router logits (and lookahead logits) are evaluated: **the one GPU sync per layer**;
4. routing on the CPU: top-k, cache-aware bias, per-row gate weights from the unbiased
   logits; the union of experts over the k rows;
5. lookahead prefetch requests; retargeting of the next layer's queued prefetches;
6. MoE: pinned pairs (row, slot) go through three `gather_qmm` calls over the layer's
   stacked bank (sorted by slot); cold experts are fetched (all misses issued together,
   then blocking waits) and multiplied one by one; per-expert biases and the gpt-oss
   activation `min(g,7) * sigmoid(1.702 g) * (clip(u,±7) + 1)` are applied on the GPU;
   contributions are accumulated with `scatter_add_axis`;
7. `async_eval` so the GPU starts the MoE while the CPU builds the next layer;
8. final norm and output head, evaluated to host logits (last row only for prefill).

### Why one sync per layer

Routing decides which experts to read from disk, so the token's route must reach the CPU
before the expert can be fetched; this is inherent to streaming. The cost is the GPU
idling while the CPU routes and builds the next graph (~0.1-0.2 ms) plus command-buffer
completion latency. The stacked pinned bank matters here: it turned ~60 kernel launches
per layer (per-expert matmuls and activations) into ~15, cutting MoE-side overhead from
13 ms to 4 ms per token on gpt-oss-20b.

### Pinned banks

`ExpertCache::pin` reads each chosen record with the same aligned uncached `preadv` as the
cold path and copies its parts into `[n_pinned, ...]` arrays allocated once per layer and
part. `slot_of[e]` maps an expert to its slot. The bank plus the base weights plus the cold
budget are wired with `mx::set_wired_limit` so the OS does not page them.

### Cold loader

A priority queue of `(prio, seq, key, group)`; priority 0 is a demand read, `10 + j` a
lookahead prefetch for a layer *j* ahead. The first job of a record to be dispatched does
admission control: evict LRU entries until the record fits the budget (entries used by the
layer currently computing are protected) and allocate one MLX buffer per part; the read is
one `preadv` scattering the record into those buffers.

A record can optionally be split into one sub-read per matrix (`--io-split 0`) so several
threads work on one expert. Measured on the M5 Max internal SSD with the page cache
evicted, this does not help: a 13 MB uncached read takes ~1.2 ms whether it is issued as
1, 3 or 8 requests, and even 8 independent records in flight only reach 13 GB/s
aggregate. Per-expert latency is therefore a floor (~1.2 ms per miss, serialized per
layer), which is why prefetching and the routing bias are the levers that matter. Prefetches
whose generation (forward pass) has passed are dropped when they reach the head of the
queue; `retarget` drops queued prefetches for a layer that a closer prediction no longer
wants. Demand reads that do not fit the budget run anyway (`n_overflow`), so prefill chunks
larger than the buffer still complete.

Buffers are MLX allocator buffers, so a freed expert returns to MLX's buffer cache and the
next read of the same size reuses it without touching the system allocator. The cache limit
is set to 1 GB.

### Prefill

Prompts are processed in chunks of `--chunk` tokens; each chunk's MoE loads each needed
expert once for all rows. The chunk is capped so that a layer's worst-case cold footprint
(`min(E, chunk * top_k)` experts, minus the pinned fraction) stays under 80% of the cold
budget. The KV prefix shared with the previous request is reused, so a multi-turn chat
prefills only its new tokens.

### Calibration and pinning

`Engine::calibrate` runs the prompt plus a few greedy tokens with nothing pinned while the
cache counts accesses; `pin` orders experts by count, backfills from the static ranking
(router weights applied to random inputs at convert time), and loads the top *n* per layer.
`profile` produces the same ranking from a text corpus for servers. The auto pin budget is
`ram_fraction * RAM - base - cold budget - KV - 1 GB` divided over layers.

## Tokenizer and chat formats

The converter writes an HF-style `tokenizer.json` from the GGUF vocabulary (byte-level
BPE, merges, added/special tokens, the pre-tokenizer regex chosen from
`tokenizer.ggml.pre`: o200k for gpt-oss, Qwen2 for Qwen). The runtime tokenizer matches
added tokens literally with a PCRE2 alternation, splits the rest with the pre-tokenizer
regex (PCRE2 with UTF/UCP, JIT), then applies rank-ordered merges. A streaming decoder
withholds incomplete UTF-8 sequences.

Chat formats are built in rather than interpreted from Jinja: harmony (gpt-oss: system
message with reasoning effort, developer instructions, `analysis`/`final` channels; stops
on `<|return|>` and `<|call|>`) and ChatML (Qwen: `<think>` blocks, optional
`enable_thinking=false`). `ChatOutputParser` splits streamed tokens into `content` and
`reasoning_content`.

## Server

cpp-httplib, single generation lock. Streaming uses a chunked SSE content provider;
non-streaming collects the text. Sampling parameters, `stop` strings, `reasoning_effort`
and `chat_template_kwargs.enable_thinking` are honored. `/v1/stats` exposes the engine
counters (hit rates, bytes read, demand wait, router swaps, MLX memory).

## Not implemented

Self-speculative decoding and the token-id prior from the study (both measured as net
losses there), tool-call parsing, batched requests, MLX-community safetensors as input.
