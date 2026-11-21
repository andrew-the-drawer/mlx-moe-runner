#pragma once
// One layer's K/V cache. Full-attention layers keep every position (slots = ctx);
// sliding-window layers keep a ring of `window + slack` slots, position p living in
// slot p % slots. Reads always return positions in order, so attention sees the same
// keys in the same order either way.
#include <cstdint>
#include <utility>

#include <mlx/mlx.h>

namespace mmr {

namespace mx = mlx::core;
struct ModelSpec;

// Minimum ring slack beyond the window: lets a sequence roll back this many tokens
// (regenerate, edited stop strings) without recomputing, at ~4 KiB per token per layer.
constexpr int kRingSlackMin = 256;

// Slots a layer needs. `slack` must be >= the largest number of rows written at once.
int kv_slots(const ModelSpec& spec, int layer, int ctx, int slack);
// Bytes of the whole K/V cache (all layers, K and V, float32, plus the engine's
// prompt-checkpoint copy of each ring).
uint64_t kv_cache_bytes(const ModelSpec& spec, int ctx, int slack);

// A ring's valid positions are [lo, end): every write of rows up to `end` overwrites the
// slots of positions below end - slots, so lo = max over its history of (end - slots).
// Rollbacks do not lower it; only restarting at position 0 does. Continuing at m needs
// positions [max(0, m - window + 1), m).
inline bool kv_ring_ok(int window, int lo, int m) {
  return window <= 0 || m <= 0 || (m - window + 1 > 0 ? m - window + 1 : 0) >= lo;
}
// Smallest m a ring state with lowest valid position `lo` can continue from.
inline int kv_ring_min(int window, int lo) { return window <= 0 || lo <= 0 ? 0 : lo + window - 1; }
// Rows to copy to restore the state after `n` positions, truncated to m.
inline int kv_rows(int slots, int window, int n, int m) { return window <= 0 ? m : (n < slots ? n : slots); }

class KVBuffer {
 public:
  KVBuffer(int n_kv_heads, int head_dim, int slots, int window);

  int slots() const { return slots_; }
  int window() const { return window_; }
  bool ring() const { return window_ > 0; }
  uint64_t bytes() const { return 2ull * n_kv_heads_ * slots_ * head_dim_ * sizeof(float); }

  // Writes k, v ([1, H_kv, n, head_dim]) at positions start..start+n-1.
  void write(const mx::array& k, const mx::array& v, int start);
  // K and V for positions [first, end) in order ([1, H_kv, end - first, head_dim]).
  std::pair<mx::array, mx::array> read(int first, int end) const;

  // Whether the next write can start at position m (m <= end, and the ring still has
  // every position m attends to).
  bool holds(int m) const { return m <= end_ && kv_ring_ok(window_, lo_, m); }
  int end() const { return end_; }
  int lo() const { return lo_; }
  // After a reset or after loading a snapshot into the arrays.
  void set_state(int end, int lo) {
    end_ = end;
    lo_ = lo;
  }

  // Evaluates the arrays and returns their host pointers ([H_kv, slots, head_dim],
  // row-major) for snapshot I/O. Call with no GPU work pending on this buffer.
  float* k_data();
  float* v_data();

 private:
  void materialize();

  int n_kv_heads_, head_dim_, slots_, window_;
  int end_ = 0, lo_ = 0;
  mx::array k_, v_;
};

} // namespace mmr
