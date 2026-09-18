#pragma once
// Two-tier expert store: pinned experts live permanently in MLX arrays;
// everything else goes through the ColdLoader.
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "io/cold_loader.h"
#include "model/layout.h"

namespace mmr {

struct CacheStats {
  uint64_t accesses = 0;
  uint64_t pinned_hits = 0;
  uint64_t cold_hits = 0;
  uint64_t misses = 0;
  LoaderStats loader;
};

// Indices into ExpertViews::parts (or PinnedBank::parts) for one matrix.
struct MatrixParts {
  int w = -1, scales = -1, biases = -1;
};

// All pinned experts of one layer stacked into contiguous arrays so a single
// gather_qmm per matrix serves every pinned expert a token touches.
struct PinnedBank {
  int n = 0;
  std::vector<int> slot_of;      // [n_experts] -> slot or -1
  std::vector<int> expert_of;    // [n] -> expert id
  std::vector<mx::array> parts;  // ordered like ExpertLayout::parts, shape [n, ...]
};

class ExpertCache {
 public:
  ExpertCache(const std::string& dir, const ExpertLayout& layout, uint64_t cold_budget_bytes, int io_threads,
              bool nocache = true, int split = 0);

  const ExpertLayout& layout() const { return layout_; }
  const MatrixParts& parts(const std::string& matrix) const; // "gate", "up", "down"

  void new_token();
  void begin_layer(int layer);

  // Fetch several cold experts of one layer; misses are issued together so the
  // SSD sees them at full queue depth. Pinned experts must not be passed here.
  std::vector<ExpertViews> get_cold(int layer, const std::vector<int>& experts);
  // Counts an access to a pinned expert (statistics / calibration).
  void note_pinned(int layer, int e);
  const PinnedBank* bank(int layer) const { return banks_[layer].n > 0 ? &banks_[layer] : nullptr; }
  int slot(int layer, int e) const { return banks_[layer].slot_of.empty() ? -1 : banks_[layer].slot_of[e]; }
  void prefetch_many(int layer, const std::vector<int>& experts, int prio);
  void retarget(int layer, const std::set<int>& keep);

  bool is_pinned(int layer, int e) const { return pinned_mask_[idx(layer, e)]; }
  bool is_resident(int layer, int e) const { return pinned_mask_[idx(layer, e)] || loader_.has({layer, e}); }
  // Resident mask for one layer as bytes (1 = resident).
  void resident_mask(int layer, std::vector<uint8_t>& out) const;

  // Pinning. `ranking[layer]` lists experts by decreasing priority.
  int pin(const std::vector<std::vector<int>>& ranking, const std::vector<int>& pins_per_layer, int threads);
  int pin_uniform(const std::vector<std::vector<int>>& ranking, int per_layer, int threads) {
    return pin(ranking, std::vector<int>(layout_.n_layers, per_layer), threads);
  }
  uint64_t pinned_bytes() const { return n_pinned_ * layout_.payload_bytes(); }
  int n_pinned() const { return n_pinned_; }

  // Usage counts (for calibration-driven pinning) since the last reset.
  const std::vector<uint32_t>& access_counts() const { return counts_; }
  void reset_counts() { std::fill(counts_.begin(), counts_.end(), 0); }
  std::vector<std::vector<int>> ranking_from_counts() const;

  CacheStats snapshot_stats() const;
  void reset_stats();
  uint64_t cold_budget() const { return loader_.budget(); }

 private:
  size_t idx(int layer, int e) const { return (size_t)layer * layout_.n_experts + e; }

  std::string dir_;
  ExpertLayout layout_;
  ColdLoader loader_;
  std::vector<uint8_t> pinned_mask_;
  std::vector<PinnedBank> banks_;
  std::vector<uint32_t> counts_;
  int n_pinned_ = 0;
  MatrixParts gate_, up_, down_;
  CacheStats stats_;
};

} // namespace mmr
