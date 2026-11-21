#include "engine/kv_buffer.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "model/spec.h"

namespace mmr {

int kv_slots(const ModelSpec& spec, int layer, int ctx, int slack) {
  if (!spec.is_swa_layer(layer)) return ctx;
  return std::min(ctx, spec.sliding_window + slack);
}

uint64_t kv_cache_bytes(const ModelSpec& spec, int ctx, int slack) {
  uint64_t total = 0;
  for (int l = 0; l < spec.n_layers; l++) {
    int slots = kv_slots(spec, l, ctx, slack);
    uint64_t b = 2ull * spec.n_kv_heads * slots * spec.head_dim * sizeof(float);
    total += slots < ctx ? 2 * b : b; // a ring also has a prompt-checkpoint copy
  }
  return total;
}

KVBuffer::KVBuffer(int n_kv_heads, int head_dim, int slots, int window)
    : n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      slots_(slots),
      window_(window),
      k_(mx::zeros({1, n_kv_heads, slots, head_dim}, mx::float32)),
      v_(mx::zeros({1, n_kv_heads, slots, head_dim}, mx::float32)) {
  materialize();
}

void KVBuffer::write(const mx::array& k, const mx::array& v, int start) {
  int n = k.shape(2);
  // Rows written in one pass overwrite the oldest slots; the oldest row of the pass
  // must still see `window - 1` earlier positions after the write.
  if (ring() && n > slots_ - window_ + 1)
    throw std::runtime_error("KV ring of " + std::to_string(slots_) + " slots cannot take " + std::to_string(n) +
                             " rows at once (window " + std::to_string(window_) + ")");
  if (n > slots_) throw std::runtime_error("KV write larger than the buffer");
  if (!holds(start))
    throw std::runtime_error("KV write at " + std::to_string(start) + " needs positions this buffer no longer has (end " +
                             std::to_string(end_) + ", lowest valid " + std::to_string(lo_) + ")");
  end_ = start + n;
  lo_ = std::max(start == 0 ? 0 : lo_, end_ - slots_); // a write at 0 starts a new history
  int H = n_kv_heads_, hd = head_dim_;
  int s = start % slots_;
  if (s + n <= slots_) {
    k_ = mx::slice_update(k_, k, {0, 0, s, 0}, {1, H, s + n, hd});
    v_ = mx::slice_update(v_, v, {0, 0, s, 0}, {1, H, s + n, hd});
    return;
  }
  int n1 = slots_ - s;
  k_ = mx::slice_update(k_, mx::slice(k, {0, 0, 0, 0}, {1, H, n1, hd}), {0, 0, s, 0}, {1, H, slots_, hd});
  v_ = mx::slice_update(v_, mx::slice(v, {0, 0, 0, 0}, {1, H, n1, hd}), {0, 0, s, 0}, {1, H, slots_, hd});
  k_ = mx::slice_update(k_, mx::slice(k, {0, 0, n1, 0}, {1, H, n, hd}), {0, 0, 0, 0}, {1, H, n - n1, hd});
  v_ = mx::slice_update(v_, mx::slice(v, {0, 0, n1, 0}, {1, H, n, hd}), {0, 0, 0, 0}, {1, H, n - n1, hd});
}

std::pair<mx::array, mx::array> KVBuffer::read(int first, int end) const {
  int c = end - first;
  if (c <= 0 || c > slots_) throw std::runtime_error("KV read of " + std::to_string(c) + " positions out of range");
  int H = n_kv_heads_, hd = head_dim_;
  int s = first % slots_;
  if (s + c <= slots_)
    return {mx::slice(k_, {0, 0, s, 0}, {1, H, s + c, hd}), mx::slice(v_, {0, 0, s, 0}, {1, H, s + c, hd})};
  int r = s + c - slots_;
  auto k = mx::concatenate({mx::slice(k_, {0, 0, s, 0}, {1, H, slots_, hd}), mx::slice(k_, {0, 0, 0, 0}, {1, H, r, hd})}, 2);
  auto v = mx::concatenate({mx::slice(v_, {0, 0, s, 0}, {1, H, slots_, hd}), mx::slice(v_, {0, 0, 0, 0}, {1, H, r, hd})}, 2);
  return {k, v};
}

void KVBuffer::materialize() {
  mx::eval(k_, v_);
  auto dense = [](const mx::array& a) { return a.flags().row_contiguous && a.data_size() == a.size(); };
  if (!dense(k_)) k_ = mx::contiguous(k_);
  if (!dense(v_)) v_ = mx::contiguous(v_);
  mx::eval(k_, v_);
  if (!dense(k_) || !dense(v_)) throw std::runtime_error("KV buffer is not dense after materialize");
}

float* KVBuffer::k_data() {
  materialize();
  return k_.data<float>();
}

float* KVBuffer::v_data() {
  materialize();
  return v_.data<float>();
}

} // namespace mmr
