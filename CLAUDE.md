# mlx-moe-runner

C++20 + MLX (C++ library from Homebrew). Build and test:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build
```

Layout: `src/gguf` (GGUF reader, ggml block decoders, exact repackers), `src/model` (spec,
on-disk layout, base weights), `src/convert` (GGUF -> model dir), `src/io` (cold loader,
expert cache with pinned banks), `src/engine` (forward pass, router, lookahead, sampler),
`src/tokenizer`, `src/server`, `tests/`. Design notes in `docs/design.md`.

Conventions: activations are float32 (quantized kernels with f32 input are exact; `mx::dequantize`
rounds in f16 and is only for diagnostics). One GPU sync per layer is intentional (routing
must reach the CPU to fetch experts). Benchmarks that claim SSD numbers must evict the page
cache first (`scripts/evict_page_cache.py`).

Formula lives in `Formula/`; Homebrew only installs formulae from a tap (`brew tap-new` for
local testing, see README). `scripts/release.sh vX.Y.Z` updates url/sha256.
