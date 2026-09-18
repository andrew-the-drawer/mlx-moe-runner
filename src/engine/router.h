#pragma once
// Top-k expert routing with cache-aware selection (CPU, on evaluated logits).
// Selection may be biased toward resident experts; gate weights always come
// from the unbiased logits (the bias only changes *which* experts run).
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace mmr {

struct RoutingConfig {
  float bias = 0.0f;              // added to resident experts' logits before top-k
  std::optional<float> margin;    // only swap when the logit gap is below this
  std::optional<float> tau;       // never displace an expert whose probability exceeds this
  int protect_top = 0;            // never displace the top-n unbiased experts
};

struct RouterStats {
  uint64_t slots = 0;
  uint64_t swapped = 0;
  double displaced_mass = 0;
  uint64_t restricted_fallback = 0;
};

class Router {
 public:
  Router(int n_experts, int k, bool norm_topk_prob = true);

  // logits: [n_experts]; resident: [n_experts] bytes or nullptr.
  // Fills idx (descending unbiased logit) and weights.
  void select(const float* logits, const uint8_t* resident, const RoutingConfig& cfg, bool restrict,
              std::vector<int>& idx, std::vector<float>& weights);
  // Plain top-n indices by descending value.
  static void topn(const float* v, int n_values, int n, std::vector<int>& out);

  const RouterStats& stats() const { return stats_; }
  void reset_stats() { stats_ = RouterStats{}; }
  int k() const { return k_; }

 private:
  void weights_for(const float* logits, const std::vector<int>& idx, std::vector<float>& w) const;

  int n_experts_;
  int k_;
  bool norm_topk_prob_;
  RouterStats stats_;
  std::vector<int> scratch_;
};

} // namespace mmr
