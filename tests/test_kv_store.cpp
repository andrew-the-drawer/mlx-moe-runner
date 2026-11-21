// Session KV store: RAM/SSD tiers, demotion, eviction, ring reuse limits, redundancy,
// persistence across restarts and fingerprint checks. Pure host memory, no model.
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <vector>

#include "engine/kv_buffer.h"
#include "engine/kv_store.h"

using namespace mmr;
namespace fs = std::filesystem;

namespace {
int failures = 0;
#define CHECK(c, ...)               \
  do {                              \
    if (!(c)) {                     \
      printf("FAIL: " __VA_ARGS__); \
      printf("\n");                 \
      failures++;                   \
    }                               \
  } while (0)

constexpr int H = 2, HD = 4;

// Live buffers: [H, slots, HD] per layer, K and V.
struct Live {
  KVGeometry g;
  std::vector<std::vector<float>> k, v;
  explicit Live(const KVGeometry& geom) : g(geom) {
    for (int l = 0; l < g.n_layers(); l++) {
      k.emplace_back((size_t)H * g.slots[l] * HD, 0.f);
      v.emplace_back((size_t)H * g.slots[l] * HD, 0.f);
    }
  }
  KVPtrs ptrs() {
    KVPtrs p;
    for (int l = 0; l < g.n_layers(); l++) {
      p.k.push_back(k[l].data());
      p.v.push_back(v[l].data());
    }
    return p;
  }
  static float val(int tok, int pos, int l, int kv, int h, int d) {
    return tok * 1000.f + pos * 0.25f + l * 100.f + kv * 50.f + h * 10.f + d;
  }
  // State after `toks`: full layers hold every position, rings the latest one per slot.
  void fill(const std::vector<int>& toks) {
    for (int l = 0; l < g.n_layers(); l++)
      for (int p = 0; p < (int)toks.size(); p++) {
        if (g.window[l] == 0 && p >= g.slots[l]) continue;
        int s = p % g.slots[l];
        for (int h = 0; h < H; h++)
          for (int d = 0; d < HD; d++) {
            k[l][((size_t)h * g.slots[l] + s) * HD + d] = val(toks[p], p, l, 0, h, d);
            v[l][((size_t)h * g.slots[l] + s) * HD + d] = val(toks[p], p, l, 1, h, d);
          }
      }
  }
  void clear() {
    for (auto& x : k) std::fill(x.begin(), x.end(), -1.f);
    for (auto& x : v) std::fill(x.begin(), x.end(), -1.f);
  }
  // Positions a continuation from m reads must match the state after toks (truncated to m).
  bool matches(const std::vector<int>& toks, int m) const {
    int n = (int)toks.size();
    for (int l = 0; l < g.n_layers(); l++) {
      int lo = g.window[l] > 0 ? std::max(0, n - g.slots[l]) : 0;
      int hi = g.window[l] > 0 ? n : m;
      for (int p = lo; p < hi; p++) {
        int s = p % g.slots[l];
        for (int h = 0; h < H; h++)
          for (int d = 0; d < HD; d++)
            if (k[l][((size_t)h * g.slots[l] + s) * HD + d] != val(toks[p], p, l, 0, h, d) ||
                v[l][((size_t)h * g.slots[l] + s) * HD + d] != val(toks[p], p, l, 1, h, d))
              return false;
      }
    }
    return true;
  }
};

std::vector<int> seq(int n, int base) {
  std::vector<int> t(n);
  for (int i = 0; i < n; i++) t[i] = base + i;
  return t;
}
// Rings in this geometry have 20 slots and were written continuously.
int ring_lo(const std::vector<int>& t) { return std::max(0, (int)t.size() - 20); }
std::vector<int> concat(std::vector<int> a, const std::vector<int>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
} // namespace

int main() {
  fs::path dir = fs::temp_directory_path() / ("mmr_kv_test_" + std::to_string(getpid()));
  fs::remove_all(dir);

  // Layers: full (64), ring (20 slots, window 8 -> rollback up to 13), full (64).
  KVGeometry g;
  g.n_kv_heads = H;
  g.head_dim = HD;
  g.slots = {64, 20, 64};
  g.window = {0, 8, 0};
  g.model_id = "test-model";
  Live live(g);

  auto A = seq(30, 100);
  auto B = seq(40, 500);

  // RAM tier: find limits for the ring, partial restore.
  {
    KVStoreOptions o;
    o.ram_bytes = 64ull << 20;
    KVStore st(g, o);
    live.fill(A);
    CHECK(st.put(A, live.ptrs(), ring_lo(A)) == KVTier::Ram, "put to RAM");
    auto m = st.find(concat(A, {7, 8}), 1000);
    CHECK(m.len == 30 && m.tier == KVTier::Ram, "full match: %d", m.len);
    auto diverge25 = concat(std::vector<int>(A.begin(), A.begin() + 25), {1, 2, 3});
    CHECK(st.find(diverge25, 1000).len == 25, "ring rollback 5 within slack");
    auto diverge10 = concat(std::vector<int>(A.begin(), A.begin() + 10), {1, 2, 3});
    CHECK(st.find(diverge10, 1000).len == 0, "ring rollback 20 beyond slack must miss");
    CHECK(st.find(concat(A, {1}), 12).len == 0, "max_len below the ring's reach must miss");
    m = st.find(diverge25, 1000);
    live.clear();
    CHECK(st.take(m.id, m.len, live.ptrs()), "take from RAM");
    CHECK(live.matches(A, 25), "RAM restore content");
    CHECK(st.find(A, 1000).len == 0, "entry leaves the store on take");
  }

  // Redundancy: a full-attention-only store drops prefixes; a ring store keeps them.
  {
    KVGeometry gf = g;
    gf.window = {0, 0, 0};
    gf.slots = {64, 64, 64};
    Live lf(gf);
    KVStoreOptions o;
    o.ram_bytes = 64ull << 20;
    KVStore st(gf, o);
    lf.fill(A);
    st.put(A, lf.ptrs(), ring_lo(A));
    lf.fill(concat(A, {1, 2}));
    st.put(concat(A, {1, 2}), lf.ptrs(), ring_lo(concat(A, {1, 2})));
    CHECK(st.stats_json()["ram"]["entries"] == 1 && st.stats_json()["dominated"] == 1, "prefix dropped");

    KVStore sr(g, o);
    live.fill(A);
    sr.put(A, live.ptrs(), ring_lo(A));
    live.fill(concat(A, seq(20, 900)));
    sr.put(concat(A, seq(20, 900)), live.ptrs(), ring_lo(concat(A, seq(20, 900))));
    CHECK(sr.stats_json()["ram"]["entries"] == 2, "ring prefix kept (different reusable range)");
    CHECK(sr.find(concat(A, {5}), 1000).len == 30, "older ring snapshot still serves its prompt");
  }

  // Demotion RAM -> SSD, restore from SSD, persistence, fingerprint.
  {
    KVStoreOptions o;
    o.dir = dir.string();
    o.disk_bytes = 64ull << 20;
    {
      KVStore probe(g, KVStoreOptions{});
      o.ram_bytes = probe.snapshot_bytes(40) + 1000; // room for one snapshot
    }
    {
      KVStore st(g, o);
      live.fill(A);
      CHECK(st.put(A, live.ptrs(), ring_lo(A)) == KVTier::Ram, "A to RAM");
      live.fill(B);
      CHECK(st.put(B, live.ptrs(), ring_lo(B)) == KVTier::Ram, "B to RAM");
      auto ma = st.find(concat(A, {3}), 1000);
      CHECK(ma.len == 30 && ma.tier == KVTier::Disk, "A demoted to disk (tier %s)", tier_name(ma.tier));
      CHECK(st.stats_json()["demoted"] == 1, "demoted count");
    }
    {
      KVStore st(g, o); // reopen: RAM is gone, the disk snapshot of A is indexed again
      auto ma = st.find(concat(A, {3}), 1000);
      CHECK(ma.len == 30 && ma.tier == KVTier::Disk, "A found after restart");
      live.clear();
      CHECK(st.take(ma.id, ma.len, live.ptrs()), "take from disk");
      CHECK(live.matches(A, 30), "disk restore content");
      CHECK(st.find(A, 1000).len == 0, "taken from disk");
      int files = 0;
      for (auto& e : fs::directory_iterator(dir)) files += e.path().extension() == ".mmrkv";
      CHECK(files == 0, "file removed after take (%d left)", files);
      // Partial restore from disk (full layers truncated).
      o.ram_bytes = 0;
    }
    {
      KVStore st(g, o);
      live.fill(B);
      CHECK(st.put(B, live.ptrs(), ring_lo(B)) == KVTier::Disk, "direct to disk when RAM is 0");
      auto mb = st.find(concat(std::vector<int>(B.begin(), B.begin() + 35), {1}), 1000);
      CHECK(mb.len == 35, "partial disk match %d", mb.len);
      live.clear();
      CHECK(st.take(mb.id, mb.len, live.ptrs()) && live.matches(B, 35), "partial disk restore");
      live.fill(B);
      st.put(B, live.ptrs(), ring_lo(B));
    }
    {
      KVGeometry other = g;
      other.model_id = "other-weights";
      KVStore st(other, o);
      CHECK(st.find(concat(B, {1}), 1000).len == 0, "other weights must not match");
      int files = 0;
      for (auto& e : fs::directory_iterator(dir)) files += e.path().extension() == ".mmrkv";
      CHECK(files == 0, "stale snapshot removed");
    }
  }

  // SSD LRU eviction, skip when too large, protected entry.
  {
    KVStoreOptions o;
    o.dir = dir.string();
    uint64_t one = 0;
    {
      KVStore probe(g, KVStoreOptions{});
      one = probe.snapshot_bytes(30) + 16384 * 2;
    }
    o.disk_bytes = 2 * one + one / 2;
    fs::remove_all(dir);
    KVStore st(g, o);
    std::vector<std::vector<int>> s = {seq(30, 1000), seq(30, 2000), seq(30, 3000)};
    for (auto& t : s) {
      live.fill(t);
      st.put(t, live.ptrs(), ring_lo(t));
    }
    CHECK(st.find(concat(s[0], {1}), 1000).len == 0, "oldest evicted");
    CHECK(st.find(concat(s[2], {1}), 1000).len == 30, "newest kept");
    CHECK(st.stats_json()["evicted"] == 1, "evicted count");
    auto keep = st.find(concat(s[1], {1}), 1000);
    live.fill(seq(30, 4000));
    st.put(seq(30, 4000), live.ptrs(), ring_lo(seq(30, 4000)), keep.id);
    CHECK(st.find(concat(s[1], {1}), 1000).len == 30, "protected entry survives eviction");
    live.fill(seq(60, 5000));
    KVStoreOptions small;
    small.ram_bytes = one / 4; // smaller than any snapshot, and no SSD tier
    KVStore st2(g, small);
    CHECK(st2.put(seq(60, 5000), live.ptrs(), ring_lo(seq(60, 5000))) == KVTier::None, "too large is skipped");
  }

  fs::remove_all(dir);
  if (failures == 0) printf("test_kv_store: all passed\n");
  return failures == 0 ? 0 : 1;
}
