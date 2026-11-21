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
| `src/engine/` | `Engine` (forward pass, generation, prefix reuse), `KVBuffer` (per-layer K/V, rings for sliding-window layers), `KVStore` (session store: RAM and SSD tiers), `Router` (top-k with bias), `Lookahead`, `Sampler` |
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
   k), KV write into the layer's `KVBuffer` (see [KV cache](#kv-cache)), fused
   `scaled_dot_product_attention` (SDPA) with attention sinks over the positions the rows
   can see (all of them for full attention, the last `window` for sliding-window layers),
   with a causal + sliding-window mask for k > 1, output projection;
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
budget. The prompt prefix already in the active KV cache, in the
[prompt checkpoint](#prompt-checkpoint) or in the [session store](#session-store) is
reused, so a multi-turn chat prefills only its new tokens. The last prompt token is always
run again for its logits.

### Calibration and pinning

`Engine::calibrate` runs the prompt plus a few greedy tokens with nothing pinned while the
cache counts accesses; `pin` orders experts by count, backfills from the static ranking
(router weights applied to random inputs at convert time), and loads the top *n* per layer.
`profile` produces the same ranking from a text corpus for servers. The auto pin budget is
`ram_fraction * RAM - base - cold budget - KV - session RAM tier - 1 GB` divided over
layers.

## KV cache

### The idea: never prefill the same tokens twice, without spending memory on it

Two costs pull in opposite directions:

- **Prefilling is expensive.** On a streamed model every prompt token costs expert reads
  from SSD: ~230 MB per token on gpt-oss-120b, 35-45 tok/s. An 8K-token conversation
  takes ~200 s to prefill again.
- **Memory is expensive too.** KV competes with pinned experts for unified memory (1 GB
  is ~77 experts on the 120b), and fewer pinned experts slow down *every* token of *every*
  request.

KV that is already computed is worth keeping, but only the small part that is useful
right now belongs in memory; the rest goes to SSD, where it costs nothing but disk space
and reads back thousands of times faster than prefilling it again (72 KB per token on the
120b, versus 230 MB of expert reads).

So the engine keeps a few saved states, each in the cheapest place that still pays off,
and for every request uses the cheapest one that can serve it:

| source | what it holds | where | cost to reuse (8K tokens, 120b) | when it can serve |
|---|---|---|---|---|
| live cache | the active conversation | memory (always) | free | the prompt extends the cached tokens, or rolls back a short way (see [rollback](#rollback)) |
| prompt checkpoint | sliding-window state at the end of the last prompt | memory, 28 MB | `memcpy` of 28 MB | a follow-up turn or a regenerate of the active conversation |
| session store, RAM | conversations switched away from | memory, `--kv-ram-mb` | `memcpy` of ~0.63 GB | a conversation comes back (memory permitting) |
| session store, SSD | the same, overflow and long term | SSD, `--kv-disk-gb` | ~0.1 s read | a conversation comes back, even after a restart |
| none | | | full prefill, ~200 s | anything else |

`prefill()` checks them in that order: the live cache, then the checkpoint, then the
store (longest matching prefix wins); whichever reuses the most tokens is used, and only
the rest is prefilled. Saving is just as frugal: the checkpoint is a fixed 28 MB copy
taken after every prompt, and a conversation is written to the store only when a request
abandons it. Continuing a chat or regenerating writes nothing.

The rest of this section covers each piece: where memory goes, the per-layer buffers and
why they limit rollback, the checkpoint, and the session store.

### Memory tiers

Apple silicon has one pool of unified memory shared by CPU and GPU, so there is no
"offload to host RAM" tier: the only tier below memory is the SSD. What lives where:

| tier | contents | bound |
|---|---|---|
| unified memory, wired | base weights, pinned expert banks, cold expert LRU, active KV cache | `mx::set_wired_limit(base + pinned + cold + KV + 512 MB)` |
| unified memory | session store RAM tier (idle conversations) | `--kv-ram-mb` |
| MLX buffer cache | freed buffers, reused by the next allocation of the same size | `set_cache_limit(1 GB)` |
| SSD | expert records (`F_NOCACHE` reads), session store SSD tier | `--kv-disk-gb` |

Every byte of KV in memory is a byte not available for pinned experts: on the 120b 1 GB
is ~77 pinned experts, about 2 per layer, and every request pays for the extra misses.
The design therefore keeps the active cache as small as exactness allows and moves idle
conversations to the SSD rather than keeping them in memory.

The active conversation's KV cannot be offloaded: SDPA reads every cached position of
every layer for every token (2.4 GB per token at 32K context on the 120b, even with the
rings below), from the same SSD the expert streaming already saturates. Only
sparse/approximate attention could change that, which is out of scope for an exact engine.

### KVBuffer

Each layer has a `KVBuffer` holding K and V as float32 `[1, H_kv, slots, d]`, allocated
once. Full-attention layers have `slots = ctx`. Sliding-window layers (gpt-oss: every other
layer, window 128) keep a ring of `window + slack` slots, `slack = max(chunk, 256)`;
position p lives in slot `p % slots`. A chunk that crosses the end of the ring is written
with two `slice_update`s; reads of `[first, end)` return positions in order (two slices
and a concatenate when they wrap), so SDPA sees the same keys in the same order as with a
linear buffer. A ring is used only if it can wrap (`window + slack < ctx`).

On gpt-oss-120b (36 layers, 8 KV heads x 64) this halves the cache: 0.36 GB instead of
0.60 GB at 4K context, 2.5 GB instead of 4.8 GB at 32K, 9.7 GB instead of 19.3 GB at
128K (the totals include the prompt-checkpoint copy of the rings, below). gpt-oss-20b:
0.24 GB instead of 0.40 GB at 4K. Qwen3 has no sliding window and is unchanged.

The slack also bounds the rows written in one pass (a prefill chunk), so the oldest row
of a chunk still sees its whole window after the chunk overwrites the oldest slots.

### Rollback

A new request's tokens usually match the cached ones up to some position m and then
differ. *Rolling back* means keeping positions `[0, m)` and recomputing from m. It happens
on every request: a regenerate rolls back 1 token (the last prompt token is always run
again for its logits), a chat follow-up rolls back the whole previous answer (harmony
drops its analysis channel, Qwen3 its `<think>` block), an edited message rolls back to
the edit.

For a full-attention layer this is free: position p lives in row p and is never
overwritten, so rolling back is setting the length to m.

A ring forgets. With a toy window of 4 (a token sees itself and the 3 before it) and 6
slots, after writing positions 0-9:

```
slot:   0  1  2  3  4  5
holds:  6  7  8  9  4  5        positions 0-3 are gone; valid range [4, 10), lo = 4
```

- Roll back to m = 8: position 8 needs 5, 6, 7. All present, so it works.
- Roll back to m = 6: position 6 needs 3, 4, 5. Position 3 is gone, so it can't.

The rule is: continuing at m needs `[max(0, m - window + 1), m)`, so `m - window + 1 >= lo`,
where `lo` is the ring's lowest valid position. A fresh ring has `lo = end - slots`, so it
can roll back `slots - window + 1 = slack + 1` positions: 257 with the real numbers.

`lo` must be tracked, not recomputed from `end`. Roll back from the state above to m = 7
and write one token 7':

```
slot:   0  1   2  3  4  5
holds:  6  7'  8  9  4  5       end = 8, but valid is still [4, 8): lo stays 4
```

`end - slots` would give `lo = 2` and allow rolling back to m = 5, which needs positions 2
and 3. Those were overwritten before the first rollback. So each buffer keeps
`lo = max(lo, end - slots)` across writes. A rollback lowers `end`, never `lo`; only a
restart at position 0 resets it. A write that would need positions the ring no longer has
throws instead of attending to stale keys.

Regenerates and short edits fit within the slack. A chat follow-up rolls back hundreds or
thousands of tokens, past the slack, which is what the checkpoint is for.

### Prompt checkpoint

A follow-up turn diverges from the cached tokens right after the previous prompt, so the
rollback is the length of the previous answer, usually more than the slack. At the end of
every prefill the engine therefore copies the rings (28 MB on the 120b) and records the
position P and the rings' `lo`. Nothing is computed; it is a snapshot of the sliding-window
state at the end of the prompt. Full-attention layers are not copied: generation only
appends after P, so their rows `[0, P)` stay intact.

Example on the 120b (window 128, 384 slots), prompt of 1,000 tokens, answer of 500:

```
after the prompt:  end = 1000, lo = 616   -> checkpoint taken (P = 1000, lo = 616)
after the answer:  end = 1500, lo = 1116
follow-up turn, matches up to m = 1000:
  live rings:  1000 - 127 = 873 >= 1116?  no  (positions 873-999 were overwritten)
  checkpoint:  873 >= 616?                yes -> copy the rings back, set end = 1000, lo = 616,
                                                 keep the full layers, prefill from 1000
```

Regenerating the same prompt (m = P - 1) works the same way. The checkpoint can serve any
m with `P - 257 <= m <= P` (less if `lo` was raised by an earlier rollback, which is why
`lo` is recorded rather than assumed to be `P - 384`). It is dropped when a request rolls
back below P (those positions are about to be rewritten; a new checkpoint is taken at the
end of that prefill), when a session is restored from the store, and on `reset_kv`.

### Session store

`KVStore` keeps snapshots of conversations the engine switched away from, in a RAM tier
(`--kv-ram-mb`, default 0) and an SSD tier (`--kv-disk-gb`, default 0 = off; directory
`--kv-dir`, default `<model-dir>/kv-cache`). Both are LRU. A snapshot lives in exactly one
tier: new snapshots go to RAM if they fit, RAM entries move to SSD when RAM is full (or
are dropped without an SSD tier), and an entry leaves the store when it is restored into
the active cache (it is saved again if the engine switches away again).

A snapshot is the state after n tokens: for full-attention layers rows `[0, n)`, for rings
the whole ring plus its `lo`. Ring models are saved at the prompt checkpoint (what a
returning conversation continues from), others at the end of the generated tokens. The RAM
blob and the file body share one layout: per layer K then V, each a section of `H_kv`
chunks of `rows * d` floats, sections aligned to 16 KiB. A file adds a 16 KiB-aligned
header (magic, JSON with the fingerprint, ring `lo` and the tokens). Snapshots are copied
straight between the active buffers and the blob or file (`memcpy`, or `preadv`/`pwritev`
with one iovec per head and `F_NOCACHE`), so a save or restore needs no extra memory the
size of the conversation. Files are written to a temporary name and renamed.

Cost on the 120b: 72 KB per token (half the layers are rings) plus 28 MB per snapshot,
so an 8K-token conversation is ~0.63 GB, read back in ~0.1 s, versus ~200 s to prefill
it again at 230 MB of expert reads per token.

A snapshot is taken only when a request abandons the active conversation, i.e. keeps less
than its last prompt minus one token (another conversation, or an edit before the end of
the previous prompt), or when a store hit replaces the active cache. Continuing turns and
regenerations write nothing. Sessions shorter than 16 tokens are not saved.

Lookup: for each entry, the common prefix with the request (capped at the prompt length
minus one and the context), accepted if the rings can continue from it
(`m >= lo + window - 1`); the longest wins, RAM before SSD. The store is used when it
beats what the active cache and the checkpoint offer; the active conversation is saved
first, with the entry about to be restored protected from eviction. A new snapshot drops
older entries it makes redundant (a token prefix of it whose reusable range it covers).

Restarts: the SSD tier is re-indexed from the file headers at startup, in modification
time order. The fingerprint is a hash of `config.json` (weights, quantization, spec) plus
the KV head geometry and the ring sizes; files that do not match (other weights, or a
`--chunk` that changes the ring size) are deleted. Full-attention slot counts are not part
of it, so snapshots survive a different `--ctx` as long as they fit. Routing bias is not
part of it either: as with in-memory prefix reuse, KV computed with cache-aware routing is
reused as is.

`/v1/stats` reports `kv` (active length and bytes, checkpoint position, store tiers,
hits, bytes moved, times), and each completion's usage reports `prompt_tokens_cached` and
`prompt_cache_source` (`live`, `ckpt`, `ram`, `disk`, `none`).

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
For the KV cache: float16/quantized KV (it would halve memory again but break the float32
convention, and SDPA needs matching dtypes), offloading the active conversation's KV
(see [Memory tiers](#memory-tiers)), asynchronous snapshot writes (a save happens
synchronously before the next prefill), and letting the KV cache grow into the cold expert
budget instead of being allocated at `--ctx`.
