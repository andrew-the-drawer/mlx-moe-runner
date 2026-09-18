#include "engine/router.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mmr {

Router::Router(int n_experts, int k, bool norm_topk_prob)
    : n_experts_(n_experts), k_(k), norm_topk_prob_(norm_topk_prob) {}

void Router::topn(const float* v, int n_values, int n, std::vector<int>& out) {
  out.resize(n_values);
  for (int i = 0; i < n_values; i++) out[i] = i;
  n = std::min(n, n_values);
  std::partial_sort(out.begin(), out.begin() + n, out.end(), [&](int a, int b) { return v[a] > v[b]; });
  out.resize(n);
}

void Router::weights_for(const float* logits, const std::vector<int>& idx, std::vector<float>& w) const {
  w.resize(idx.size());
  if (norm_topk_prob_) {
    float m = -std::numeric_limits<float>::infinity();
    for (int i : idx) m = std::max(m, logits[i]);
    float sum = 0;
    for (size_t j = 0; j < idx.size(); j++) {
      w[j] = expf(logits[idx[j]] - m);
      sum += w[j];
    }
    for (auto& x : w) x /= sum;
  } else {
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_experts_; i++) m = std::max(m, logits[i]);
    double sum = 0;
    for (int i = 0; i < n_experts_; i++) sum += exp((double)(logits[i] - m));
    for (size_t j = 0; j < idx.size(); j++) w[j] = (float)(exp((double)(logits[idx[j]] - m)) / sum);
  }
}

void Router::select(const float* logits, const uint8_t* resident, const RoutingConfig& cfg, bool restrict,
                    std::vector<int>& idx, std::vector<float>& weights) {
  const int k = k_;
  if (restrict) {
    int n_res = 0;
    if (resident)
      for (int i = 0; i < n_experts_; i++) n_res += resident[i] ? 1 : 0;
    if (!resident || n_res < k) {
      stats_.restricted_fallback++;
      topn(logits, n_experts_, k, idx);
    } else {
      std::vector<float> masked(n_experts_);
      for (int i = 0; i < n_experts_; i++)
        masked[i] = resident[i] ? logits[i] : -std::numeric_limits<float>::infinity();
      topn(masked.data(), n_experts_, k, idx);
    }
    weights_for(logits, idx, weights);
    return;
  }

  topn(logits, n_experts_, k, idx);
  if (cfg.bias == 0.0f || resident == nullptr) {
    weights_for(logits, idx, weights);
    return;
  }

  std::vector<float> biased(n_experts_);
  for (int i = 0; i < n_experts_; i++) biased[i] = logits[i] + (resident[i] ? cfg.bias : 0.f);
  std::vector<int> b_idx;
  topn(biased.data(), n_experts_, k, b_idx);
  stats_.slots += k;

  std::vector<int> removed, added; // both in descending unbiased-logit order
  for (int e : idx)
    if (std::find(b_idx.begin(), b_idx.end(), e) == b_idx.end()) removed.push_back(e);
  for (int e : b_idx)
    if (std::find(idx.begin(), idx.end(), e) == idx.end()) added.push_back(e);
  if (removed.empty()) {
    weights_for(logits, idx, weights);
    return;
  }
  std::sort(added.begin(), added.end(), [&](int a, int b) { return logits[a] > logits[b]; });

  std::vector<float> probs;
  if (cfg.tau) {
    probs.resize(n_experts_);
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_experts_; i++) m = std::max(m, logits[i]);
    double sum = 0;
    for (int i = 0; i < n_experts_; i++) {
      probs[i] = expf(logits[i] - m);
      sum += probs[i];
    }
    for (auto& p : probs) p = (float)(p / sum);
  }
  std::vector<int> keep = idx;
  // Pair the weakest removed expert with the strongest added one, and so on.
  size_t n_pairs = std::min(removed.size(), added.size());
  for (size_t j = 0; j < n_pairs; j++) {
    int rem = removed[removed.size() - 1 - j];
    int add = added[j];
    bool is_protected = false;
    for (int t = 0; t < cfg.protect_top && t < (int)idx.size(); t++)
      if (idx[t] == rem) is_protected = true;
    if (is_protected) continue;
    if (cfg.margin && logits[rem] - logits[add] > *cfg.margin) continue;
    if (cfg.tau && probs[rem] > *cfg.tau) continue;
    auto it = std::find(keep.begin(), keep.end(), rem);
    *it = add;
    stats_.swapped++;
    if (cfg.tau) stats_.displaced_mass += probs[rem] - probs[add];
  }
  std::sort(keep.begin(), keep.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  idx = keep;
  weights_for(logits, idx, weights);
}

} // namespace mmr
