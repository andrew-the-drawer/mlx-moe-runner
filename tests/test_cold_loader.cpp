// Exercises the cold loader against a tiny synthetic expert store.
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <vector>

#include "io/cold_loader.h"
#include "io/expert_cache.h"

using namespace mmr;
namespace fs = std::filesystem;

int main() {
  int failures = 0;
  fs::path dir = fs::temp_directory_path() / ("mmr_test_" + std::to_string(getpid()));
  fs::create_directories(dir);
  ExpertLayout L;
  L.n_layers = 2;
  L.n_experts = 6;
  L.align = 4096;
  L.parts.push_back({"gate.weight", {64, 4}, "uint32", 0, 1024, 4096});
  L.parts.push_back({"gate.scales", {64, 1}, "float16", 4096, 128, 4096});
  L.parts.push_back({"up.weight", {64, 4}, "uint32", 8192, 1024, 4096});
  L.parts.push_back({"up.scales", {64, 1}, "float16", 12288, 128, 4096});
  L.parts.push_back({"down.weight", {32, 8}, "uint32", 16384, 1024, 4096});
  L.parts.push_back({"down.scales", {32, 2}, "float16", 20480, 128, 4096});
  L.record_bytes = 32768;
  for (int l = 0; l < L.n_layers; l++) {
    L.files[l] = L.file_for(l);
    std::vector<uint8_t> file(L.record_bytes * L.n_experts);
    for (int e = 0; e < L.n_experts; e++) {
      uint32_t* w = (uint32_t*)&file[L.expert_offset(e)];
      for (int i = 0; i < 256; i++) w[i] = (uint32_t)(l * 1000 + e * 10 + (i % 7));
    }
    int fd = ::open((dir / L.files[l]).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    write(fd, file.data(), file.size());
    close(fd);
  }

  {
    // Budget for 3 records.
    ColdLoader loader(dir.string(), L, 3 * 24576, 2, true);
    loader.new_token();
    loader.begin_layer(0);
    auto v = loader.get({0, 2});
    if (v.parts.size() != 6) { printf("FAIL: part count\n"); failures++; }
    mx::eval(v.parts[0]);
    const uint32_t* w = v.parts[0].data<uint32_t>();
    if (w[0] != 20 || w[1] != 21 || w[6] != 26 || w[7] != 20) { printf("FAIL: content %u %u\n", w[0], w[7]); failures++; }
    if (!loader.has({0, 2})) { printf("FAIL: has after get\n"); failures++; }
    auto s = loader.snapshot_stats();
    if (s.n_reads != 1 || s.n_demand != 1) { printf("FAIL: stats after demand read\n"); failures++; }

    // Prefetch for layer 1, then fetch: should be a prefetch hit with no wait counted as demand.
    loader.request_many({{1, 0}, {1, 1}}, PRIO_LOOKAHEAD + 1);
    for (int i = 0; i < 200 && !(loader.has({1, 0}) && loader.has({1, 1})); i++) usleep(1000);
    loader.begin_layer(1);
    loader.get({1, 0});
    loader.get({1, 1});
    s = loader.snapshot_stats();
    if (s.n_prefetch_hit != 2) { printf("FAIL: prefetch hits %llu\n", (unsigned long long)s.n_prefetch_hit); failures++; }
    if (s.n_reads != 3) { printf("FAIL: reads %llu\n", (unsigned long long)s.n_reads); failures++; }

    // Over budget: fetching a 4th record must evict the oldest unprotected one ((0,2)).
    loader.get({1, 3});
    s = loader.snapshot_stats();
    if (s.n_evict != 1 || loader.has({0, 2})) { printf("FAIL: eviction (evict=%llu)\n", (unsigned long long)s.n_evict); failures++; }
    if (s.resident_bytes > 3 * 24576) { printf("FAIL: budget exceeded\n"); failures++; }

    // Stale prefetch from an old generation is dropped at the next pass.
    loader.request_many({{0, 5}}, PRIO_LOOKAHEAD + 3);
    loader.new_token();
    usleep(20000);
    // Retarget drops queued prefetches not in keep.
    loader.request_many({{0, 4}}, PRIO_LOOKAHEAD + 2);
    loader.retarget(0, {});
    usleep(20000);
    s = loader.snapshot_stats();
    printf("dropped %llu, reads %llu, resident %llu\n", (unsigned long long)s.n_dropped, (unsigned long long)s.n_reads,
           (unsigned long long)s.n_resident);
  }
  {
    ExpertCache cache(dir.string(), L, 2 * 24576, 2, true);
    std::vector<std::vector<int>> ranking = {{5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5}};
    int n = cache.pin(ranking, {2, 1}, 2);
    if (n != 3 || !cache.is_pinned(0, 5) || !cache.is_pinned(0, 4) || !cache.is_pinned(1, 0) || cache.is_pinned(1, 1)) {
      printf("FAIL: pinning\n");
      failures++;
    }
    cache.new_token();
    cache.begin_layer(0);
    auto views = cache.get_cold(0, {1});
    cache.note_pinned(0, 5);
    cache.note_pinned(0, 4);
    if (views.size() != 1) { printf("FAIL: get_cold size\n"); failures++; }
    auto s = cache.snapshot_stats();
    if (s.pinned_hits != 2 || s.misses != 1) { printf("FAIL: cache stats %llu %llu\n", (unsigned long long)s.pinned_hits, (unsigned long long)s.misses); failures++; }
    const auto* bank = cache.bank(0);
    if (!bank || bank->n != 2 || cache.slot(0, 5) != 0 || cache.slot(0, 4) != 1 || cache.slot(0, 3) != -1) { printf("FAIL: bank slots\n"); failures++; }
    if (bank && (bank->parts[0].shape(0) != 2 || bank->parts[0].shape(1) != 64)) { printf("FAIL: bank shape\n"); failures++; }
    if (bank) {
      mx::eval(bank->parts[0]);
      const uint32_t* w = bank->parts[0].data<uint32_t>();
      if (w[0] != 50 || w[256] != 40) { printf("FAIL: bank content %u %u\n", w[0], w[256]); failures++; }
    }
    std::vector<uint8_t> mask;
    cache.resident_mask(0, mask);
    if (!mask[5] || !mask[1] || mask[0]) { printf("FAIL: resident mask\n"); failures++; }
    auto r = cache.ranking_from_counts();
    if (r[0][0] != 1 && r[0][0] != 4 && r[0][0] != 5) { printf("FAIL: ranking from counts\n"); failures++; }
  }
  fs::remove_all(dir);
  printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
