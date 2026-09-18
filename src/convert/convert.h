#pragma once
// GGUF -> converted model directory (see model/layout.h).
#include <string>
#include <vector>

namespace mmr {

struct ConvertOptions {
  std::string gguf_path;
  std::string out_dir;
  std::vector<int> layers;      // empty = all
  int lossy_bits = 8;           // bits used when a ggml type has no exact MLX mapping
  int lossy_group = 32;
  int threads = 0;              // 0 = performance cores
  bool skip_experts = false;    // only write config/base/tokenizer (debugging)
  bool verify = true;           // dequantize-compare the first expert of each layer
};

int run_convert(const ConvertOptions& opts);

} // namespace mmr
