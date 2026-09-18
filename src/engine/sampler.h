#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace mmr {

struct SamplingParams {
  float temperature = 0.7f;
  float top_p = 0.9f;
  int top_k = 0;        // 0 = disabled
  float min_p = 0.0f;   // 0 = disabled
  uint64_t seed = 42;
};

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p);
  // Picks a token from raw logits [vocab].
  int sample(const float* logits, int vocab);
  // Probability distribution after temperature/top-k/top-p/min-p (for speculative decoding).
  void distribution(const float* logits, int vocab, std::vector<float>& probs);
  const SamplingParams& params() const { return params_; }

 private:
  SamplingParams params_;
  std::mt19937_64 rng_;
  std::vector<int> order_;
  std::vector<float> probs_;
};

} // namespace mmr
