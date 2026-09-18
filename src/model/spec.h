#pragma once
// Model geometry and conventions. Built from GGUF metadata at convert time and
// stored in config.json of the converted model directory.
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace mmr::gguf { class Reader; }

namespace mmr {

struct ModelSpec {
  std::string arch;            // "gpt-oss", "qwen3moe", ...
  std::string name;
  int n_layers = 0;
  int embed_dim = 0;
  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int n_experts = 0;
  int n_experts_used = 0;
  int expert_inter = 0;
  int vocab_size = 0;
  int context_length = 0;
  float rms_eps = 1e-6f;
  float rope_base = 10000.f;
  int rope_dims = 0;           // rotary dims (== head_dim unless partial rotary)
  float rope_yarn_factor = 0.f;
  int rope_orig_ctx = 0;
  float yarn_beta_fast = 32.f;
  float yarn_beta_slow = 1.f;
  int sliding_window = 0;
  int swa_every = 0;           // layer i uses the window iff swa_every > 0 && i % swa_every == 0
  bool qk_norm = false;
  bool attn_bias = false;
  bool attn_sinks = false;
  bool router_bias = false;
  bool expert_bias = false;
  bool norm_topk_prob = true;  // renormalize the selected experts' weights (false: qwen2moe-style raw softmax probs)
  std::string activation = "silu";        // "silu" (SwiGLU) or "gptoss"
  std::string ffn_norm_name = "ffn_norm";
  std::string chat_format = "none";       // "harmony", "chatml", "none"

  // Tokenizer conventions
  int bos_id = -1;
  int eos_id = -1;
  int pad_id = -1;
  bool add_bos = false;
  std::vector<int> stop_ids;

  int gqa_ratio() const { return n_kv_heads > 0 ? n_heads / n_kv_heads : 1; }
  bool is_swa_layer(int i) const { return sliding_window > 0 && swa_every > 0 && i % swa_every == 0; }
  bool uses_yarn() const { return rope_yarn_factor > 0.f && rope_orig_ctx > 0; }
  // YaRN attention scale applied to q and k (1.0 without YaRN).
  float rope_mscale() const;
  // Per-dimension rope periods (what MLX's fast::rope takes as `freqs`).
  std::vector<float> rope_freqs() const;

  static ModelSpec from_gguf(const gguf::Reader& reader);
  nlohmann::json to_json() const;
  static ModelSpec from_json(const nlohmann::json& j);
};

} // namespace mmr
