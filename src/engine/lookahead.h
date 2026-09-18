#pragma once
// Multi-layer lookahead prefetch: apply the routers of layers l+1..l+J to the
// residual stream at layer l (each layer adds a small update, so the residual
// is a good proxy for later layers' inputs). The RMS-norm weight is folded
// into each router so all targets are one matmul against the normalized
// residual.
#include <vector>

#include <mlx/mlx.h>

#include "model/spec.h"
#include "model/weights.h"

namespace mmr {

namespace mx = mlx::core;

class Lookahead {
 public:
  Lookahead(const BaseWeights& base, const ModelSpec& spec);

  // Lazy logits [k, J, E] for target layers l0..l1-1 given residual X [k, D].
  mx::array logits(int l0, int l1, const mx::array& X) const;

  // CPU: from evaluated logits [k, J, E], the union over rows of the top-n per
  // target layer. out[j] lists experts for layer l0 + j.
  void candidates(const float* logits, int k, int J, int n, std::vector<std::vector<int>>& out) const;

 private:
  ModelSpec spec_;
  mx::array folded_;                  // [L, E, D]
  std::optional<mx::array> biases_;   // [L, E]
};

} // namespace mmr
