#pragma once
// Non-expert weights of a converted model, resident in MLX arrays.
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <nlohmann/json_fwd.hpp>

#include "model/layout.h"
#include "model/spec.h"

namespace mmr {

namespace mx = mlx::core;

// A linear map y = x W^T + bias with W either dense or MLX-quantized.
struct QLinear {
  QuantDesc q;
  int out_dim = 0;
  int in_dim = 0;
  std::optional<mx::array> w;      // dense [out, in] or packed uint32 [out, in*bits/32]
  std::optional<mx::array> scales;
  std::optional<mx::array> biases; // affine quantization biases (not the additive bias)
  std::optional<mx::array> bias;   // additive bias [out]

  bool valid() const { return w.has_value(); }
  uint64_t nbytes() const;
  // x [..., in] -> [..., out]
  mx::array operator()(const mx::array& x) const;
  // Rows of W dequantized to float32: [n, in]
  mx::array rows(const mx::array& ids) const;
};

// Builds a QLinear from arrays in a map under `base_name` (.weight/.scales/.biases/.bias).
QLinear make_qlinear(const std::unordered_map<std::string, mx::array>& arrays, const std::string& base_name,
                     const QuantDesc& q, bool with_bias);

struct LayerWeights {
  mx::array attn_norm;
  QLinear q, k, v, o;
  std::optional<mx::array> q_norm, k_norm;  // [head_dim]
  std::optional<mx::array> sinks;           // [n_heads]
  mx::array ffn_norm;
  mx::array router_w;                       // [E, D] float32
  std::optional<mx::array> router_b;        // [E]
  std::optional<mx::array> gate_bias, up_bias, down_bias; // [E, I], [E, I], [E, D]
  LayerWeights() : attn_norm(mx::array(0.f)), ffn_norm(mx::array(0.f)), router_w(mx::array(0.f)) {}
};

struct BaseWeights {
  QLinear embed;   // table [vocab, D]
  QLinear output;  // [vocab, D]
  mx::array output_norm;
  std::vector<LayerWeights> layers;
  uint64_t nbytes = 0;

  BaseWeights() : output_norm(mx::array(0.f)) {}
  static BaseWeights load(const std::string& dir, const ModelSpec& spec, const nlohmann::json& config);
  // [k, D] float32 embeddings for token ids.
  mx::array embed_tokens(const std::vector<int>& ids) const;
};

} // namespace mmr
