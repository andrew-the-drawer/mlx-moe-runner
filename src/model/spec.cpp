#include "model/spec.h"

#include <cmath>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "gguf/gguf_reader.h"

using json = nlohmann::json;

namespace mmr {

float ModelSpec::rope_mscale() const {
  if (!uses_yarn()) return 1.0f;
  return 0.1f * logf(rope_yarn_factor) + 1.0f;
}

std::vector<float> ModelSpec::rope_freqs() const {
  int d = rope_dims > 0 ? rope_dims : head_dim;
  std::vector<float> freqs(d / 2);
  for (int i = 0; i < d / 2; i++) {
    double inv = 1.0 / pow((double)rope_base, (2.0 * i) / d);
    if (uses_yarn()) {
      double factor = rope_yarn_factor, orig = rope_orig_ctx;
      auto corr_dim = [&](double n_rot) {
        return d * log(orig / (n_rot * 2 * M_PI)) / (2 * log((double)rope_base));
      };
      double low = std::max(floor(corr_dim(yarn_beta_fast)), 0.0);
      double high = std::min(ceil(corr_dim(yarn_beta_slow)), (double)d - 1);
      double ramp = (i - low) / std::max(high - low, 0.001);
      ramp = std::min(std::max(ramp, 0.0), 1.0);
      double extrap = 1.0 - ramp;
      inv = (inv / factor) * (1 - extrap) + inv * extrap;
    }
    freqs[i] = (float)(1.0 / inv);
  }
  return freqs;
}

ModelSpec ModelSpec::from_gguf(const gguf::Reader& r) {
  ModelSpec s;
  s.arch = r.architecture();
  s.name = r.get_string("general.name", s.arch);
  auto g = [&](const std::string& k) { return s.arch + "." + k; };
  s.n_layers = (int)r.get_int(g("block_count"), 0);
  s.embed_dim = (int)r.get_int(g("embedding_length"), 0);
  s.n_heads = (int)r.get_int(g("attention.head_count"), 0);
  s.n_kv_heads = (int)r.get_int(g("attention.head_count_kv"), s.n_heads);
  s.head_dim = (int)r.get_int(g("attention.key_length"), s.n_heads ? s.embed_dim / s.n_heads : 0);
  s.n_experts = (int)r.get_int(g("expert_count"), 0);
  s.n_experts_used = (int)r.get_int(g("expert_used_count"), 0);
  s.expert_inter = (int)r.get_int(g("expert_feed_forward_length"), r.get_int(g("feed_forward_length"), 0));
  s.context_length = (int)r.get_int(g("context_length"), 4096);
  s.rms_eps = (float)r.get_float(g("attention.layer_norm_rms_epsilon"), 1e-6);
  s.rope_base = (float)r.get_float(g("rope.freq_base"), 10000.0);
  s.rope_dims = (int)r.get_int(g("rope.dimension_count"), s.head_dim);
  if (r.get_string(g("rope.scaling.type")) == "yarn") {
    s.rope_yarn_factor = (float)r.get_float(g("rope.scaling.factor"), 0.0);
    s.rope_orig_ctx = (int)r.get_int(g("rope.scaling.original_context_length"), 0);
    s.yarn_beta_fast = (float)r.get_float(g("rope.scaling.yarn_beta_fast"), 32.0);
    s.yarn_beta_slow = (float)r.get_float(g("rope.scaling.yarn_beta_slow"), 1.0);
  }
  s.sliding_window = (int)r.get_int(g("attention.sliding_window"), 0);
  s.qk_norm = r.has_tensor("blk.0.attn_q_norm.weight");
  s.attn_bias = r.has_tensor("blk.0.attn_q.bias");
  s.attn_sinks = r.has_tensor("blk.0.attn_sinks.weight");
  s.router_bias = r.has_tensor("blk.0.ffn_gate_inp.bias");
  s.expert_bias = r.has_tensor("blk.0.ffn_gate_exps.bias");
  const auto* embd = r.tensor("token_embd.weight");
  if (!embd) throw std::runtime_error("GGUF: token_embd.weight missing");
  s.vocab_size = (int)embd->ne(1);

  if (s.arch == "gpt-oss") {
    s.activation = "gptoss";
    s.swa_every = 2;
    s.chat_format = "harmony";
  } else if (s.arch == "qwen3moe" || s.arch == "qwen2moe") {
    s.chat_format = "chatml";
    s.norm_topk_prob = s.arch == "qwen3moe";
    if (s.sliding_window) s.swa_every = (int)r.get_int(g("attention.sliding_window_pattern"), 0);
  } else {
    if (s.sliding_window) s.swa_every = (int)r.get_int(g("attention.sliding_window_pattern"), 0);
  }
  if (!r.has_tensor("blk.0.ffn_norm.weight")) {
    if (r.has_tensor("blk.0.post_attention_norm.weight")) s.ffn_norm_name = "post_attention_norm";
    else if (r.has_tensor("blk.0.attn_post_norm.weight")) s.ffn_norm_name = "attn_post_norm";
  }
  if (s.n_experts == 0 || s.n_experts_used == 0)
    throw std::runtime_error("Not a mixture-of-experts model (expert_count missing)");

  s.bos_id = (int)r.get_int("tokenizer.ggml.bos_token_id", -1);
  s.eos_id = (int)r.get_int("tokenizer.ggml.eos_token_id", -1);
  s.pad_id = (int)r.get_int("tokenizer.ggml.padding_token_id", -1);
  s.add_bos = r.get_int("tokenizer.ggml.add_bos_token", 0) != 0;
  if (s.eos_id >= 0) s.stop_ids.push_back(s.eos_id);
  return s;
}

json ModelSpec::to_json() const {
  json j;
  j["arch"] = arch;
  j["name"] = name;
  j["n_layers"] = n_layers;
  j["embed_dim"] = embed_dim;
  j["n_heads"] = n_heads;
  j["n_kv_heads"] = n_kv_heads;
  j["head_dim"] = head_dim;
  j["n_experts"] = n_experts;
  j["n_experts_used"] = n_experts_used;
  j["expert_inter"] = expert_inter;
  j["vocab_size"] = vocab_size;
  j["context_length"] = context_length;
  j["rms_eps"] = rms_eps;
  j["rope_base"] = rope_base;
  j["rope_dims"] = rope_dims;
  j["rope_yarn_factor"] = rope_yarn_factor;
  j["rope_orig_ctx"] = rope_orig_ctx;
  j["yarn_beta_fast"] = yarn_beta_fast;
  j["yarn_beta_slow"] = yarn_beta_slow;
  j["sliding_window"] = sliding_window;
  j["swa_every"] = swa_every;
  j["qk_norm"] = qk_norm;
  j["attn_bias"] = attn_bias;
  j["attn_sinks"] = attn_sinks;
  j["router_bias"] = router_bias;
  j["expert_bias"] = expert_bias;
  j["norm_topk_prob"] = norm_topk_prob;
  j["activation"] = activation;
  j["ffn_norm_name"] = ffn_norm_name;
  j["chat_format"] = chat_format;
  j["bos_id"] = bos_id;
  j["eos_id"] = eos_id;
  j["pad_id"] = pad_id;
  j["add_bos"] = add_bos;
  j["stop_ids"] = stop_ids;
  return j;
}

ModelSpec ModelSpec::from_json(const json& j) {
  ModelSpec s;
  s.arch = j.value("arch", "");
  s.name = j.value("name", "");
  s.n_layers = j.value("n_layers", 0);
  s.embed_dim = j.value("embed_dim", 0);
  s.n_heads = j.value("n_heads", 0);
  s.n_kv_heads = j.value("n_kv_heads", 0);
  s.head_dim = j.value("head_dim", 0);
  s.n_experts = j.value("n_experts", 0);
  s.n_experts_used = j.value("n_experts_used", 0);
  s.expert_inter = j.value("expert_inter", 0);
  s.vocab_size = j.value("vocab_size", 0);
  s.context_length = j.value("context_length", 4096);
  s.rms_eps = j.value("rms_eps", 1e-6f);
  s.rope_base = j.value("rope_base", 10000.f);
  s.rope_dims = j.value("rope_dims", s.head_dim);
  s.rope_yarn_factor = j.value("rope_yarn_factor", 0.f);
  s.rope_orig_ctx = j.value("rope_orig_ctx", 0);
  s.yarn_beta_fast = j.value("yarn_beta_fast", 32.f);
  s.yarn_beta_slow = j.value("yarn_beta_slow", 1.f);
  s.sliding_window = j.value("sliding_window", 0);
  s.swa_every = j.value("swa_every", 0);
  s.qk_norm = j.value("qk_norm", false);
  s.attn_bias = j.value("attn_bias", false);
  s.attn_sinks = j.value("attn_sinks", false);
  s.router_bias = j.value("router_bias", false);
  s.expert_bias = j.value("expert_bias", false);
  s.norm_topk_prob = j.value("norm_topk_prob", true);
  s.activation = j.value("activation", "silu");
  s.ffn_norm_name = j.value("ffn_norm_name", "ffn_norm");
  s.chat_format = j.value("chat_format", "none");
  s.bos_id = j.value("bos_id", -1);
  s.eos_id = j.value("eos_id", -1);
  s.pad_id = j.value("pad_id", -1);
  s.add_bos = j.value("add_bos", false);
  s.stop_ids = j.value("stop_ids", std::vector<int>{});
  return s;
}

} // namespace mmr
