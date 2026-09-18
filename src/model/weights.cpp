#include "model/weights.h"

#include <stdexcept>

#include <nlohmann/json.hpp>

#include "util/log.h"

namespace mmr {

uint64_t QLinear::nbytes() const {
  uint64_t n = 0;
  if (w) n += w->nbytes();
  if (scales) n += scales->nbytes();
  if (biases) n += biases->nbytes();
  if (bias) n += bias->nbytes();
  return n;
}

mx::array QLinear::operator()(const mx::array& x) const {
  mx::array y = x;
  if (q.quantized()) {
    y = mx::quantized_matmul(x, *w, *scales, biases, true, q.group, q.bits, q.mode);
  } else {
    y = mx::matmul(x, mx::transpose(mx::astype(*w, x.dtype())));
  }
  if (bias) y = mx::add(y, mx::astype(*bias, y.dtype()));
  return y;
}

mx::array QLinear::rows(const mx::array& ids) const {
  if (q.quantized()) {
    auto wr = mx::take(*w, ids, 0);
    auto sr = mx::take(*scales, ids, 0);
    std::optional<mx::array> br;
    if (biases) br = mx::take(*biases, ids, 0);
    return mx::dequantize(wr, sr, br, q.group, q.bits, q.mode, std::nullopt, mx::float32);
  }
  return mx::astype(mx::take(*w, ids, 0), mx::float32);
}

QLinear make_qlinear(const std::unordered_map<std::string, mx::array>& arrays, const std::string& base_name,
                     const QuantDesc& q, bool with_bias) {
  QLinear l;
  l.q = q;
  auto it = arrays.find(base_name + ".weight");
  if (it == arrays.end()) throw std::runtime_error("missing tensor " + base_name + ".weight");
  l.w = it->second;
  l.out_dim = l.w->shape(0);
  if (q.quantized()) {
    auto s = arrays.find(base_name + ".scales");
    if (s == arrays.end()) throw std::runtime_error("missing tensor " + base_name + ".scales");
    l.scales = s->second;
    auto b = arrays.find(base_name + ".biases");
    if (b != arrays.end()) l.biases = b->second;
    l.in_dim = l.w->shape(1) * 32 / q.bits;
  } else {
    l.in_dim = l.w->shape(1);
  }
  if (with_bias) {
    auto b = arrays.find(base_name + ".bias");
    if (b != arrays.end()) l.bias = b->second;
  }
  return l;
}

BaseWeights BaseWeights::load(const std::string& dir, const ModelSpec& spec, const nlohmann::json& config) {
  auto [arrays, meta] = mx::load_safetensors(dir + "/base.safetensors");
  std::map<std::string, QuantDesc> bq;
  if (config.contains("base_quant"))
    for (auto& [k, v] : config["base_quant"].items()) bq[k] = QuantDesc::from_json(v);
  // base_quant is keyed by tensor name ("blk.3.attn_v"); older files used roles ("attn_v").
  auto quant_of = [&](const std::string& name) {
    auto it = bq.find(name);
    if (it != bq.end()) return it->second;
    std::string role = name;
    if (role.rfind("blk.", 0) == 0) role = role.substr(role.find('.', 4) + 1);
    it = bq.find(role);
    if (it == bq.end()) throw std::runtime_error("config.json: no base_quant entry for " + name);
    return it->second;
  };
  auto get = [&](const std::string& name) -> mx::array {
    auto it = arrays.find(name);
    if (it == arrays.end()) throw std::runtime_error("base.safetensors: missing " + name);
    return it->second;
  };
  auto maybe = [&](const std::string& name) -> std::optional<mx::array> {
    auto it = arrays.find(name);
    if (it == arrays.end()) return std::nullopt;
    return it->second;
  };

  BaseWeights b;
  b.embed = make_qlinear(arrays, "token_embd", quant_of("token_embd"), false);
  b.output = make_qlinear(arrays, "output", quant_of("output"), false);
  b.output_norm = get("output_norm.weight");
  b.layers.reserve(spec.n_layers);
  for (int i = 0; i < spec.n_layers; i++) {
    std::string p = "blk." + std::to_string(i) + ".";
    LayerWeights L;
    L.attn_norm = get(p + "attn_norm.weight");
    L.q = make_qlinear(arrays, p + "attn_q", quant_of(p + "attn_q"), spec.attn_bias);
    L.k = make_qlinear(arrays, p + "attn_k", quant_of(p + "attn_k"), spec.attn_bias);
    L.v = make_qlinear(arrays, p + "attn_v", quant_of(p + "attn_v"), spec.attn_bias);
    L.o = make_qlinear(arrays, p + "attn_output", quant_of(p + "attn_output"), spec.attn_bias);
    if (spec.qk_norm) {
      L.q_norm = get(p + "attn_q_norm.weight");
      L.k_norm = get(p + "attn_k_norm.weight");
    }
    if (spec.attn_sinks) L.sinks = get(p + "attn_sinks.weight");
    L.ffn_norm = get(p + spec.ffn_norm_name + ".weight");
    L.router_w = get(p + "ffn_gate_inp.weight");
    if (spec.router_bias) L.router_b = maybe(p + "ffn_gate_inp.bias");
    if (spec.expert_bias) {
      L.gate_bias = maybe(p + "ffn_gate_exps.bias");
      L.up_bias = maybe(p + "ffn_up_exps.bias");
      L.down_bias = maybe(p + "ffn_down_exps.bias");
    }
    b.layers.push_back(std::move(L));
  }
  std::vector<mx::array> all;
  for (auto& [k, a] : arrays) {
    all.push_back(a);
    b.nbytes += a.nbytes();
  }
  mx::eval(all);
  return b;
}

mx::array BaseWeights::embed_tokens(const std::vector<int>& ids) const {
  mx::array idx(ids.data(), {(int)ids.size()}, mx::int32);
  return embed.rows(idx);
}

} // namespace mmr
