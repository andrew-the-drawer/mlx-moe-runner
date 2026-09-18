#include "engine/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mmr {

Sampler::Sampler(const SamplingParams& p) : params_(p), rng_(p.seed) {}

void Sampler::distribution(const float* logits, int vocab, std::vector<float>& probs) {
  probs.assign(vocab, 0.f);
  float temp = params_.temperature;
  if (temp <= 0.f) {
    int best = (int)(std::max_element(logits, logits + vocab) - logits);
    probs[best] = 1.f;
    return;
  }
  order_.resize(vocab);
  for (int i = 0; i < vocab; i++) order_[i] = i;
  int keep = vocab;
  if (params_.top_k > 0 && params_.top_k < vocab) {
    keep = params_.top_k;
    std::partial_sort(order_.begin(), order_.begin() + keep, order_.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
  } else {
    std::sort(order_.begin(), order_.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  }
  float m = logits[order_[0]];
  double sum = 0;
  probs_.resize(keep);
  for (int i = 0; i < keep; i++) {
    probs_[i] = expf((logits[order_[i]] - m) / temp);
    sum += probs_[i];
  }
  for (int i = 0; i < keep; i++) probs_[i] = (float)(probs_[i] / sum);
  if (params_.min_p > 0.f) {
    float thresh = params_.min_p * probs_[0];
    int n = 0;
    while (n < keep && probs_[n] >= thresh) n++;
    keep = std::max(n, 1);
  }
  if (params_.top_p < 1.f) {
    double cum = 0;
    int n = 0;
    while (n < keep) {
      cum += probs_[n];
      n++;
      if (cum >= params_.top_p) break;
    }
    keep = std::max(n, 1);
  }
  double s2 = 0;
  for (int i = 0; i < keep; i++) s2 += probs_[i];
  for (int i = 0; i < keep; i++) probs[order_[i]] = (float)(probs_[i] / s2);
}

int Sampler::sample(const float* logits, int vocab) {
  if (params_.temperature <= 0.f)
    return (int)(std::max_element(logits, logits + vocab) - logits);
  std::vector<float> probs;
  distribution(logits, vocab, probs);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng_), cum = 0;
  int last = 0;
  for (int i = 0; i < vocab; i++) {
    if (probs[i] <= 0.f) continue;
    cum += probs[i];
    last = i;
    if (r < cum) return i;
  }
  return last;
}

} // namespace mmr
