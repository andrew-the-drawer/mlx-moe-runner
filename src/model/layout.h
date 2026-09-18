#pragma once
// On-disk layout of a converted model directory:
//
//   config.json         spec + expert layout + base tensor quantization
//   base.safetensors    every non-expert tensor (attention, norms, routers, head)
//   experts_NNN.bin     layer NNN: n_experts fixed-size records, one expert each
//   tokenizer.json      HF-format byte-level BPE tokenizer
//   profile.json        (optional) per-layer expert usage ranking for pinning
//
// An expert record is a sequence of parts (gate/up/down weights, scales,
// biases), each part page-aligned so that one preadv with F_NOCACHE can
// scatter a record straight into MLX buffers without copies.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace mmr {

struct QuantDesc {
  std::string mode = "none";  // "affine", "mxfp4", or "none" (dense weight)
  int bits = 0;
  int group = 0;
  bool has_biases = false;
  std::string dtype = "float16"; // dense dtype when mode == "none"

  bool quantized() const { return mode != "none"; }
  nlohmann::json to_json() const;
  static QuantDesc from_json(const nlohmann::json& j);
};

struct PartDesc {
  std::string name;          // e.g. "gate.weight", "gate.scales", "gate.biases"
  std::vector<int> shape;
  std::string dtype;         // "uint32", "uint8", "float16", "bfloat16", "float32"
  uint64_t offset = 0;       // within the record
  uint64_t bytes = 0;        // exact payload bytes
  uint64_t padded = 0;       // bytes reserved in the record (multiple of align)
};

struct ExpertLayout {
  int n_layers = 0;
  int n_experts = 0;
  uint64_t align = 4096;
  uint64_t record_bytes = 0; // per expert, multiple of 16384
  std::vector<PartDesc> parts;
  std::map<std::string, QuantDesc> quant; // "gate", "up", "down"
  std::map<int, std::string> files;       // layer -> file name

  const PartDesc* part(const std::string& name) const;
  std::string file_for(int layer) const;
  uint64_t expert_offset(int expert) const { return record_bytes * (uint64_t)expert; }
  uint64_t payload_bytes() const;
  nlohmann::json to_json() const;
  static ExpertLayout from_json(const nlohmann::json& j);
};

size_t dtype_size(const std::string& dtype);

} // namespace mmr
