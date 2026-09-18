#pragma once
// The inference engine: MLX for all compute, experts streamed from SSD.
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

#include "engine/lookahead.h"
#include "engine/router.h"
#include "engine/sampler.h"
#include "io/expert_cache.h"
#include "model/layout.h"
#include "model/spec.h"
#include "model/weights.h"
#include "tokenizer/bpe_tokenizer.h"

namespace mmr {

namespace mx = mlx::core;

struct EngineOptions {
  std::string model_dir;
  uint64_t cold_budget_bytes = (uint64_t)1536 << 20;
  int io_threads = 8;
  int io_split = 0;             // sub-reads per expert record (0 = one per matrix, 1 = whole record)
  bool nocache = true;
  int kv_max = 4096;
  int prefill_chunk = 32;
  RoutingConfig routing;
  int lookahead_depth = 0;
  int lookahead_extra = 0;
  bool wire = true;             // keep base + pinned weights resident via MLX's wired limit
  bool verbose = true;
};

struct EngineStats {
  double attn_s = 0, route_s = 0, prefetch_s = 0, gather_s = 0, moe_s = 0, head_s = 0, sync_s = 0;
  uint64_t rows = 0, passes = 0;
  CacheStats cache;
  RouterStats router;
};

struct GenerateResult {
  int n_prompt = 0;
  int n_cached = 0;
  int n_generated = 0;
  double prefill_s = 0;
  double decode_s = 0;
  std::string finish_reason = "length";
};

class Engine {
 public:
  explicit Engine(const EngineOptions& opts);
  ~Engine();

  const ModelSpec& spec() const { return spec_; }
  const ExpertLayout& layout() const { return layout_; }
  const BpeTokenizer& tokenizer() const { return *tokenizer_; }
  ExpertCache& cache() { return *cache_; }
  EngineOptions& options() { return opts_; }
  uint64_t base_bytes() const { return base_.nbytes; }
  uint64_t expert_bytes() const { return layout_.payload_bytes(); }

  // --- pinning ---------------------------------------------------------------
  const std::vector<std::vector<int>>& static_ranking() const { return static_ranking_; }
  // Runs the prompt plus n greedy tokens with nothing pinned to gather usage counts.
  void calibrate(const std::vector<int>& prompt, int n_tokens);
  // Pins `per_layer` experts per layer: usage counts first, then `ranking` (or the
  // static ranking) as backfill. Returns the number pinned.
  int pin(int per_layer, const std::vector<std::vector<int>>* ranking = nullptr);
  int pin_per_layer(const std::vector<int>& per_layer, const std::vector<std::vector<int>>* ranking = nullptr);
  // Suggests pins per layer so the whole model footprint stays within `ram_fraction` of RAM.
  int suggest_pins(double ram_fraction) const;

  // --- forward -----------------------------------------------------------------
  // Runs one short prefill + decode so MLX compiles its kernels before real traffic.
  void warmup();
  void reset_kv();
  int kv_len() const { return kv_len_; }
  // Runs k tokens at positions start_pos.. through the model. Returns evaluated
  // float32 logits [k or 1, vocab] (last row only when last_only).
  mx::array forward_rows(const std::vector<int>& tokens, int start_pos, bool last_only = false);
  // Prefill with prefix reuse; returns logits for the last prompt token.
  mx::array prefill(const std::vector<int>& tokens, int* n_cached = nullptr);

  // --- generation ---------------------------------------------------------------
  // on_token returns false to stop early.
  GenerateResult generate(const std::vector<int>& prompt, int max_tokens, const SamplingParams& sampling,
                          const std::vector<int>& stop_ids, const std::function<bool(int)>& on_token);

  EngineStats stats() const;
  void reset_stats();
  nlohmann::json stats_json() const;

 private:
  struct KV {
    mx::array k;
    mx::array v;
  };
  void attention(int layer, mx::array& X, int start_pos, int k);
  void moe(int layer, const mx::array& N, mx::array& X, int k, const std::vector<int>& experts,
           const std::vector<std::vector<std::pair<int, float>>>& rows_of);
  void apply_wired_limit();

  EngineOptions opts_;
  nlohmann::json config_;
  ModelSpec spec_;
  ExpertLayout layout_;
  BaseWeights base_;
  std::unique_ptr<BpeTokenizer> tokenizer_;
  std::unique_ptr<ExpertCache> cache_;
  std::unique_ptr<Router> router_;
  std::unique_ptr<Lookahead> lookahead_;
  std::vector<std::vector<int>> static_ranking_;
  std::vector<KV> kv_;
  std::vector<int> kv_tokens_;
  int kv_len_ = 0;
  mx::array rope_freqs_;
  float rope_mscale_ = 1.f;
  EngineStats timers_;
  uint64_t wired_bytes_ = 0;
};

} // namespace mmr
