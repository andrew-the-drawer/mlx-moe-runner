#include <cmath>
#include <cstdio>
#include <vector>

#include "engine/router.h"

using namespace mmr;

int main() {
  int failures = 0;
  Router r(8, 2, true);
  std::vector<float> logits = {0.1f, 3.0f, 2.9f, 0.5f, 2.5f, -1.f, 0.f, 1.f};
  std::vector<int> idx;
  std::vector<float> w;
  RoutingConfig cfg;
  r.select(logits.data(), nullptr, cfg, false, idx, w);
  if (idx != std::vector<int>{1, 2}) { printf("FAIL: plain top-k\n"); failures++; }
  if (fabsf(w[0] + w[1] - 1.f) > 1e-5f) { printf("FAIL: weights normalized\n"); failures++; }
  // Bias: expert 4 (2.5) resident, expert 2 (2.9) not. bias 0.5 -> 3.0 > 2.9, so 4 replaces 2.
  std::vector<uint8_t> resident = {0, 0, 0, 0, 1, 0, 0, 0};
  cfg.bias = 0.5f;
  r.select(logits.data(), resident.data(), cfg, false, idx, w);
  if (idx != std::vector<int>{1, 4}) { printf("FAIL: biased swap (got %d %d)\n", idx[0], idx[1]); failures++; }
  // Weights come from the unbiased logits: softmax(3.0, 2.5)
  float e0 = expf(3.0f), e1 = expf(2.5f);
  if (fabsf(w[0] - e0 / (e0 + e1)) > 1e-5f) { printf("FAIL: unbiased weights after swap\n"); failures++; }
  if (r.stats().swapped != 1) { printf("FAIL: swap stats\n"); failures++; }
  // Margin guard blocks the swap when the gap (0.4) exceeds the margin.
  cfg.margin = 0.1f;
  r.select(logits.data(), resident.data(), cfg, false, idx, w);
  if (idx != std::vector<int>{1, 2}) { printf("FAIL: margin guard\n"); failures++; }
  cfg.margin.reset();
  // protect_top keeps the top-1; here the swap target is the 2nd slot, so it still swaps.
  cfg.protect_top = 1;
  r.select(logits.data(), resident.data(), cfg, false, idx, w);
  if (idx != std::vector<int>{1, 4}) { printf("FAIL: protect_top\n"); failures++; }
  cfg.protect_top = 2;
  r.select(logits.data(), resident.data(), cfg, false, idx, w);
  if (idx != std::vector<int>{1, 2}) { printf("FAIL: protect_top 2\n"); failures++; }
  // Restricted routing picks the best resident experts.
  std::vector<uint8_t> res2 = {1, 0, 0, 1, 1, 0, 0, 0};
  r.select(logits.data(), res2.data(), RoutingConfig{}, true, idx, w);
  if (idx != std::vector<int>{4, 3}) { printf("FAIL: restricted\n"); failures++; }
  // qwen2moe-style raw probabilities
  Router r2(8, 2, false);
  r2.select(logits.data(), nullptr, RoutingConfig{}, false, idx, w);
  if (w[0] + w[1] >= 1.f) { printf("FAIL: raw probs should not sum to 1\n"); failures++; }
  printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
