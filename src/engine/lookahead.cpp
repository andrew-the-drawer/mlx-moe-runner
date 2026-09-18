#include "engine/lookahead.h"

#include <algorithm>

#include "engine/router.h"

namespace mmr {

Lookahead::Lookahead(const BaseWeights& base, const ModelSpec& spec) : spec_(spec), folded_(mx::array(0.f)) {
  std::vector<mx::array> folded, biases;
  for (auto& L : base.layers) {
    folded.push_back(mx::multiply(mx::astype(L.router_w, mx::float32), mx::expand_dims(mx::astype(L.ffn_norm, mx::float32), 0)));
    if (L.router_b) biases.push_back(mx::astype(*L.router_b, mx::float32));
  }
  folded_ = mx::stack(folded); // [L, E, D]
  if (biases.size() == folded.size()) biases_ = mx::stack(biases);
  mx::eval(folded_);
  if (biases_) mx::eval(*biases_);
}

mx::array Lookahead::logits(int l0, int l1, const mx::array& X) const {
  int J = l1 - l0;
  int E = spec_.n_experts, D = spec_.embed_dim;
  auto Xn = mx::fast::rms_norm(X, std::nullopt, spec_.rms_eps);          // [k, D]
  auto W = mx::reshape(mx::slice(folded_, {l0, 0, 0}, {l1, E, D}), {J * E, D});
  auto lg = mx::reshape(mx::matmul(Xn, mx::transpose(W)), {X.shape(0), J, E});
  if (biases_) lg = mx::add(lg, mx::expand_dims(mx::slice(*biases_, {l0, 0}, {l1, E}), 0));
  return lg;
}

void Lookahead::candidates(const float* logits, int k, int J, int n, std::vector<std::vector<int>>& out) const {
  int E = spec_.n_experts;
  out.assign(J, {});
  std::vector<uint8_t> seen(E);
  std::vector<int> top;
  for (int j = 0; j < J; j++) {
    std::fill(seen.begin(), seen.end(), 0);
    for (int r = 0; r < k; r++) {
      Router::topn(logits + ((size_t)r * J + j) * E, E, n, top);
      for (int e : top)
        if (!seen[e]) {
          seen[e] = 1;
          out[j].push_back(e);
        }
    }
  }
}

} // namespace mmr
