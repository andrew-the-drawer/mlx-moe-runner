#include "io/expert_cache.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "util/log.h"
#include "util/mlx_util.h"

#include <cstring>

namespace mmr {

ExpertCache::ExpertCache(const std::string& dir, const ExpertLayout& layout, uint64_t cold_budget_bytes,
                         int io_threads, bool nocache, int split)
    : dir_(dir), layout_(layout), loader_(dir, layout, cold_budget_bytes, io_threads, nocache, split),
      pinned_mask_((size_t)layout.n_layers * layout.n_experts, 0),
      banks_(layout.n_layers),
      counts_((size_t)layout.n_layers * layout.n_experts, 0) {
  auto find = [&](const std::string& name) {
    for (size_t i = 0; i < layout.parts.size(); i++)
      if (layout.parts[i].name == name) return (int)i;
    return -1;
  };
  for (auto* m : {&gate_, &up_, &down_}) {
    const char* n = m == &gate_ ? "gate" : (m == &up_ ? "up" : "down");
    m->w = find(std::string(n) + ".weight");
    m->scales = find(std::string(n) + ".scales");
    m->biases = find(std::string(n) + ".biases");
    if (m->w < 0) throw std::runtime_error(std::string("expert layout lacks ") + n + ".weight");
  }
}

const MatrixParts& ExpertCache::parts(const std::string& matrix) const {
  if (matrix == "gate") return gate_;
  if (matrix == "up") return up_;
  return down_;
}

void ExpertCache::new_token() { loader_.new_token(); }
void ExpertCache::begin_layer(int layer) { loader_.begin_layer(layer); }

std::vector<ExpertViews> ExpertCache::get_cold(int layer, const std::vector<int>& experts) {
  std::vector<ExpertKey> missing;
  for (int e : experts)
    if (!loader_.has({layer, e})) missing.push_back({layer, e});
  if (missing.size() > 1) loader_.request_many(missing, PRIO_DEMAND);
  std::vector<ExpertViews> out;
  out.reserve(experts.size());
  for (int e : experts) {
    size_t i = idx(layer, e);
    counts_[i]++;
    stats_.accesses++;
    if (loader_.has({layer, e})) stats_.cold_hits++;
    else stats_.misses++;
    out.push_back(loader_.get({layer, e}));
  }
  return out;
}

void ExpertCache::note_pinned(int layer, int e) {
  counts_[idx(layer, e)]++;
  stats_.accesses++;
  stats_.pinned_hits++;
}

void ExpertCache::prefetch_many(int layer, const std::vector<int>& experts, int prio) {
  std::vector<ExpertKey> keys;
  for (int e : experts)
    if (!pinned_mask_[idx(layer, e)] && !loader_.has({layer, e})) keys.push_back({layer, e});
  if (!keys.empty()) loader_.request_many(keys, prio);
}

void ExpertCache::retarget(int layer, const std::set<int>& keep) { loader_.retarget(layer, keep); }

void ExpertCache::resident_mask(int layer, std::vector<uint8_t>& out) const {
  out.resize(layout_.n_experts);
  const auto& ready = loader_.ready_mask();
  for (int e = 0; e < layout_.n_experts; e++) {
    size_t i = idx(layer, e);
    out[e] = pinned_mask_[i] || ready[i].load(std::memory_order_acquire);
  }
}

int ExpertCache::pin(const std::vector<std::vector<int>>& ranking, const std::vector<int>& pins_per_layer,
                     int threads) {
  // Decide the pinned set per layer.
  std::vector<std::vector<int>> chosen(layout_.n_layers);
  for (int l = 0; l < layout_.n_layers; l++) {
    int n = l < (int)pins_per_layer.size() ? pins_per_layer[l] : 0;
    n = std::min(n, layout_.n_experts);
    if (n <= 0) continue;
    if (banks_[l].n > 0) throw std::runtime_error("layer " + std::to_string(l) + " is already pinned");
    const auto& order = l < (int)ranking.size() ? ranking[l] : std::vector<int>{};
    for (int e : order) {
      if ((int)chosen[l].size() >= n) break;
      if (std::find(chosen[l].begin(), chosen[l].end(), e) == chosen[l].end()) chosen[l].push_back(e);
    }
    for (int e = 0; (int)chosen[l].size() < n && e < layout_.n_experts; e++)
      if (std::find(chosen[l].begin(), chosen[l].end(), e) == chosen[l].end()) chosen[l].push_back(e);
  }

  // Allocate the stacked bank buffers.
  struct Slot { int layer, slot, expert; };
  std::vector<Slot> jobs;
  std::vector<std::vector<mx::allocator::Buffer>> bufs(layout_.n_layers);
  for (int l = 0; l < layout_.n_layers; l++) {
    int n = (int)chosen[l].size();
    if (n == 0) continue;
    auto& bank = banks_[l];
    bank.n = n;
    bank.slot_of.assign(layout_.n_experts, -1);
    bank.expert_of = chosen[l];
    for (int si = 0; si < n; si++) {
      bank.slot_of[chosen[l][si]] = si;
      jobs.push_back({l, si, chosen[l][si]});
    }
    for (auto& p : layout_.parts) {
      auto buf = mx::allocator::malloc((size_t)n * p.bytes);
      if (!buf.ptr()) throw std::runtime_error("MLX allocation failed for pinned bank");
      bufs[l].push_back(buf);
    }
  }

  // Read every pinned record (aligned, uncached preadv into scratch MLX buffers)
  // and copy its parts into the bank slots.
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  std::mutex err_mutex;
  std::string error;
  int nt = std::max(1, std::min(threads, (int)jobs.size()));
  for (int t = 0; t < nt; t++) {
    pool.emplace_back([&]() {
      for (size_t i = next++; i < jobs.size(); i = next++) {
        try {
          auto v = loader_.read_record({jobs[i].layer, jobs[i].expert});
          for (size_t pi = 0; pi < layout_.parts.size(); pi++) {
            const auto& p = layout_.parts[pi];
            uint8_t* dst = (uint8_t*)bufs[jobs[i].layer][pi].raw_ptr() + (size_t)jobs[i].slot * p.bytes;
            memcpy(dst, v.parts[pi].data<uint8_t>(), p.bytes);
          }
        } catch (const std::exception& ex) {
          std::lock_guard<std::mutex> lock(err_mutex);
          error = ex.what();
        }
      }
    });
  }
  for (auto& t : pool) t.join();
  if (!error.empty()) throw std::runtime_error("pinning failed: " + error);

  for (int l = 0; l < layout_.n_layers; l++) {
    int n = (int)chosen[l].size();
    if (n == 0) continue;
    auto& bank = banks_[l];
    for (size_t pi = 0; pi < layout_.parts.size(); pi++) {
      const auto& p = layout_.parts[pi];
      mx::Shape shape{n};
      for (int d : p.shape) shape.push_back(d);
      bank.parts.push_back(array_from_buffer(bufs[l][pi], shape, dtype_from_name(p.dtype)));
    }
    for (int e : chosen[l]) pinned_mask_[idx(l, e)] = 1;
    n_pinned_ += n;
  }
  return (int)jobs.size();
}

std::vector<std::vector<int>> ExpertCache::ranking_from_counts() const {
  std::vector<std::vector<int>> r(layout_.n_layers);
  for (int l = 0; l < layout_.n_layers; l++) {
    std::vector<int> order(layout_.n_experts);
    for (int e = 0; e < layout_.n_experts; e++) order[e] = e;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return counts_[idx(l, a)] > counts_[idx(l, b)]; });
    r[l] = order;
  }
  return r;
}

CacheStats ExpertCache::snapshot_stats() const {
  CacheStats s = stats_;
  s.loader = loader_.snapshot_stats();
  return s;
}

void ExpertCache::reset_stats() {
  stats_ = CacheStats{};
  loader_.reset_stats();
}

} // namespace mmr
