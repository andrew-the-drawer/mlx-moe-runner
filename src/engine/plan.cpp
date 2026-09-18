#include "engine/plan.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "model/layout.h"
#include "model/spec.h"
#include "util/log.h"
#include "util/sysinfo.h"

namespace mmr {

namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Uncached read latency of one expert record and aggregate bandwidth with 8 in flight.
void probe_ssd(const std::string& dir, const ExpertLayout& L, double* lat_ms, double* bw_gbs) {
  std::mt19937 rng((unsigned)time(nullptr));
  std::vector<int> fds;
  for (auto& [l, f] : L.files) {
    int fd = ::open((dir + "/" + f).c_str(), O_RDONLY);
    if (fd < 0) continue;
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
    fds.push_back(fd);
  }
  if (fds.empty()) throw std::runtime_error("no expert files to probe");
  uint64_t rec = L.record_bytes;
  std::vector<void*> bufs(8);
  for (auto& b : bufs) posix_memalign(&b, 16384, rec);
  auto read_one = [&](void* buf) {
    int fd = fds[rng() % fds.size()];
    int e = (int)(rng() % L.n_experts);
    pread(fd, buf, rec, (off_t)L.expert_offset(e));
  };
  read_one(bufs[0]); // warm the path
  double t = 0;
  const int reps = 24;
  for (int i = 0; i < reps; i++) {
    double t0 = now_s();
    read_one(bufs[0]);
    t += now_s() - t0;
  }
  *lat_ms = t / reps * 1e3;
  t = 0;
  for (int r = 0; r < 4; r++) {
    double t0 = now_s();
    std::vector<std::thread> th;
    for (int i = 0; i < 8; i++) th.emplace_back([&, i]() { read_one(bufs[i]); });
    for (auto& x : th) x.join();
    t += now_s() - t0;
  }
  *bw_gbs = 8.0 * rec * 4 / t / 1e9;
  for (auto b : bufs) free(b);
  for (int fd : fds) ::close(fd);
}

double interp(const std::vector<std::pair<double, double>>& pts, double x) {
  if (x <= pts.front().first) return pts.front().second;
  for (size_t i = 1; i < pts.size(); i++)
    if (x <= pts[i].first) {
      double t = (x - pts[i - 1].first) / (pts[i].first - pts[i - 1].first);
      return pts[i - 1].second + t * (pts[i].second - pts[i - 1].second);
    }
  return pts.back().second;
}

struct Config {
  int pins;
  int cold_mb;
  float bias;
  int lookahead;
  double footprint_gb;
  double toks;
  double hit;
  double wait_ms;
  double compute_ms;
  double mb_per_tok;
};

} // namespace

int run_plan(const PlanOptions& opts) {
  std::ifstream cf(opts.model_dir + "/config.json");
  if (!cf) throw std::runtime_error("cannot open " + opts.model_dir + "/config.json");
  auto cfg = nlohmann::json::parse(cf);
  ModelSpec S = ModelSpec::from_json(cfg.at("spec"));
  ExpertLayout L = ExpertLayout::from_json(cfg.at("experts"));
  struct stat st;
  double base_gb = stat((opts.model_dir + "/base.safetensors").c_str(), &st) == 0 ? st.st_size / 1e9 : 2.5;
  double ram_gb = opts.ram_gb > 0 ? opts.ram_gb : physical_memory() / 1e9;
  double kv_gb = 2.0 * S.n_layers * S.n_kv_heads * opts.ctx * S.head_dim * 4 / 1e9;
  double rec_gb = L.record_bytes / 1e9;
  int E = S.n_experts, K = S.n_experts_used, Ln = S.n_layers;

  double lat_ms = 1.2, bw = 11.0;
  if (opts.probe) {
    probe_ssd(opts.model_dir, L, &lat_ms, &bw);
  }
  printf("Machine: %.0f GB RAM, SSD %.2f ms per %.1f MB expert read, %.1f GB/s with 8 in flight%s\n", ram_gb, lat_ms,
         L.record_bytes / 1e6, bw, opts.probe ? "" : " (assumed)");
  printf("Model: %s, %d layers x %d experts (top-%d), %.1f GB of experts, %.1f GB base, %.1f GB KV at ctx %d\n",
         S.name.c_str(), Ln, E, K, rec_gb * E * Ln, base_gb, kv_gb, opts.ctx);

  // Hit-rate curves vs pinned fraction (calibrated pinning), from the gpt-oss-120b measurements.
  const std::vector<std::pair<double, double>> hit_exact = {{0, 0}, {0.094, 0.21}, {0.25, 0.60}, {0.5, 0.85}, {0.75, 0.95}, {1, 1}};
  const std::vector<std::pair<double, double>> hit_bias = {{0, 0}, {0.094, 0.51}, {0.25, 0.73}, {0.5, 0.92}, {0.75, 0.98}, {1, 1}};
  // Per-layer sync/launch overhead measured: ~0.5 ms (gpt-oss), ~0.95 ms (qwen3: top-8, QK-norm).
  double per_layer_ms = K >= 8 ? 0.95 : 0.55;
  double gpu_bw = 400e9;

  std::vector<Config> out;
  for (int cold_mb : {512, 1024, 1536, 2048}) {
    for (float bias : opts.exact_only ? std::vector<float>{0.f} : std::vector<float>{0.f, 0.5f}) {
      for (int la : {0, 1, 2}) {
        for (int pins = 0; pins <= E; pins += std::max(1, E / 32)) {
          double pinned_gb = (double)pins * Ln * rec_gb;
          double footprint = base_gb + pinned_gb + cold_mb / 1e3 + kv_gb + 1.0;
          if (footprint > 0.6 * ram_gb) break;
          double f = (double)pins / E;
          double h = interp(bias > 0 ? hit_bias : hit_exact, f);
          double cold_experts = cold_mb * 1e6 / L.record_bytes;
          double c = 0.5 * (1 - h) * std::min(1.0, cold_experts / 150.0); // LRU reuse (measured 14-18% at 1.5 GB)
          double misses = Ln * K * std::max(0.0, 1 - h - c);
          double extra_reads = 0, wait_scale = 1.0;
          if (la == 1) { extra_reads = misses * 0.25; wait_scale = bias > 0 ? 1.0 : 0.85; }
          if (la == 2) { extra_reads = misses * 0.45; wait_scale = bias > 0 ? 1.05 : 0.95; }
          if (la > 0 && f < 0.15) wait_scale *= 0.7; // bandwidth-bound regime: prefetch pays off (study m16)
          double reads = misses + extra_reads;
          double bytes = reads * L.record_bytes;
          double wait_lat = misses * lat_ms * wait_scale;
          double wait_bw = bytes / (bw * 1e9) * 1e3;
          double wait = std::max(wait_lat, wait_bw);
          double compute = Ln * per_layer_ms + (base_gb * 1e9 + (double)K * Ln * L.record_bytes) / gpu_bw * 1e3;
          double ms = compute + wait;
          out.push_back({pins, cold_mb, bias, la, footprint, 1000.0 / ms, h, wait, compute, bytes / 1e6});
        }
      }
    }
  }
  std::sort(out.begin(), out.end(), [](const Config& a, const Config& b) { return a.toks > b.toks; });
  auto print = [&](const Config& c) {
    printf("  --pin %3d --cold-mb %4d --bias %.1f --lookahead %d   ~%5.1f tok/s   %4.1f GB footprint, pinned hit %.0f%%, "
           "wait %.0f ms, compute %.0f ms, %.0f MB/tok\n",
           c.pins, c.cold_mb, c.bias, c.lookahead, c.toks, c.footprint_gb, c.hit * 100, c.wait_ms, c.compute_ms, c.mb_per_tok);
  };
  printf("\nBest predicted configurations (footprint kept under 60%% of RAM):\n");
  for (int i = 0; i < std::min((int)out.size(), opts.top); i++) print(out[i]);
  auto best_exact = std::find_if(out.begin(), out.end(), [](const Config& c) { return c.bias == 0.f; });
  if (best_exact != out.end() && !opts.exact_only) {
    printf("\nBest with exact routing (no quality cost):\n");
    print(*best_exact);
  }
  if (!out.empty()) {
    const auto& b = out[0];
    printf("\nRecommended:\n  mlx-moe-runner serve %s --pin %d --cold-mb %d --bias %.1f --lookahead %d\n",
           opts.model_dir.c_str(), b.pins, b.cold_mb, b.bias, b.lookahead);
    printf("Predictions come from a cost model fitted to gpt-oss-120b on an M5 Max; verify with "
           "`mlx-moe-runner bench` (evict the page cache first: scripts/evict_page_cache.py).\n");
  }
  return 0;
}

} // namespace mmr
