#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>


#include "util/log.h"
#include "util/sysinfo.h"

namespace mmr {

namespace {
double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string fnv1a_hex(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
  return buf;
}

// Shorter sessions are cheap to prefill again; not worth a snapshot.
constexpr int kMinSaveTokens = 16;
} // namespace

Engine::Engine(const EngineOptions& opts) : opts_(opts), rope_freqs_(mx::array(0.f)) {
  double t0 = now_s();
  std::ifstream cf(opts.model_dir + "/config.json");
  if (!cf) throw std::runtime_error("cannot open " + opts.model_dir + "/config.json (run `mlx-moe-runner convert` first)");
  config_ = nlohmann::json::parse(cf);
  if (config_.value("format", "") != "mlx-moe-runner")
    throw std::runtime_error("not an mlx-moe-runner model directory: " + opts.model_dir);
  spec_ = ModelSpec::from_json(config_.at("spec"));
  layout_ = ExpertLayout::from_json(config_.at("experts"));
  if ((int)layout_.files.size() < spec_.n_layers)
    throw std::runtime_error("model directory holds " + std::to_string(layout_.files.size()) + " of " +
                             std::to_string(spec_.n_layers) + " expert layers (partial conversion)");
  if (config_.contains("static_ranking"))
    static_ranking_ = config_["static_ranking"].get<std::vector<std::vector<int>>>();

  if (opts.verbose)
    MMR_INFO("Model: %s (%s), %d layers, %dx%d experts, d=%d, heads %d/%d x %d, window %d, act %s",
             spec_.name.c_str(), spec_.arch.c_str(), spec_.n_layers, spec_.n_experts, spec_.n_experts_used,
             spec_.embed_dim, spec_.n_heads, spec_.n_kv_heads, spec_.head_dim, spec_.sliding_window,
             spec_.activation.c_str());

  mx::set_cache_limit((size_t)1 << 30);
  base_ = BaseWeights::load(opts.model_dir, spec_, config_);
  tokenizer_ = std::make_unique<BpeTokenizer>(opts.model_dir + "/tokenizer.json");
  cache_ = std::make_unique<ExpertCache>(opts.model_dir, layout_, opts.cold_budget_bytes, opts.io_threads, opts.nocache,
                                         opts.io_split);
  router_ = std::make_unique<Router>(spec_.n_experts, spec_.n_experts_used, spec_.norm_topk_prob);
  lookahead_ = std::make_unique<Lookahead>(base_, spec_);

  auto freqs = spec_.rope_freqs();
  rope_freqs_ = mx::array(freqs.data(), {(int)freqs.size()}, mx::float32);
  mx::eval(rope_freqs_);
  rope_mscale_ = spec_.rope_mscale();

  // Sliding-window layers get a ring of window + slack slots; the slack covers a prefill
  // chunk and short rollbacks.
  ring_slack_ = std::max(kRingSlackMin, opts.prefill_chunk);
  KVGeometry geom;
  geom.n_kv_heads = spec_.n_kv_heads;
  geom.head_dim = spec_.head_dim;
  kv_.reserve(spec_.n_layers);
  ckpt_k_.resize(spec_.n_layers);
  ckpt_v_.resize(spec_.n_layers);
  for (int i = 0; i < spec_.n_layers; i++) {
    int slots = kv_slots(spec_, i, opts.kv_max, ring_slack_);
    int window = slots < opts.kv_max ? spec_.sliding_window : 0; // a ring only if it can wrap
    kv_.emplace_back(spec_.n_kv_heads, spec_.head_dim, slots, window);
    kv_bytes_ += kv_.back().bytes();
    geom.slots.push_back(slots);
    geom.window.push_back(window);
    if (window > 0) has_ring_ = true;
  }
  if (has_ring_)
    for (int i = 0; i < spec_.n_layers; i++)
      if (kv_[i].ring()) {
        size_t n = (size_t)spec_.n_kv_heads * kv_[i].slots() * spec_.head_dim;
        ckpt_k_[i].assign(n, 0.f);
        ckpt_v_[i].assign(n, 0.f);
        kv_bytes_ += 2 * n * sizeof(float);
      }
  if (opts.kv_ram_bytes > 0 || opts.kv_disk_bytes > 0) {
    geom.model_id = fnv1a_hex(config_.dump());
    KVStoreOptions so;
    so.ram_bytes = opts.kv_ram_bytes;
    so.disk_bytes = opts.kv_disk_bytes;
    so.dir = opts.kv_dir.empty() ? opts.model_dir + "/kv-cache" : opts.kv_dir;
    so.nocache = opts.nocache;
    store_ = std::make_unique<KVStore>(geom, so);
  }
  if (opts.verbose) {
    MMR_INFO("Base weights %s, KV cache %s (%d positions%s), expert %s, cold buffer %s, %d I/O threads; ready in %.1fs",
             human_bytes((double)base_.nbytes).c_str(), human_bytes((double)kv_bytes_).c_str(), opts.kv_max,
             has_ring_ ? (", window layers " + std::to_string(spec_.sliding_window + ring_slack_) + "-slot rings").c_str() : "",
             human_bytes((double)layout_.payload_bytes()).c_str(), human_bytes((double)opts.cold_budget_bytes).c_str(),
             opts.io_threads, now_s() - t0);
    if (store_)
      MMR_INFO("Session store: %s RAM, %s SSD%s%s", human_bytes((double)opts.kv_ram_bytes).c_str(),
               human_bytes((double)opts.kv_disk_bytes).c_str(), opts.kv_disk_bytes ? " in " : "",
               opts.kv_disk_bytes ? (opts.kv_dir.empty() ? opts.model_dir + "/kv-cache" : opts.kv_dir).c_str() : "");
  }
}

Engine::~Engine() = default;

// --- pinning -------------------------------------------------------------------

void Engine::calibrate(const std::vector<int>& prompt, int n_tokens) {
  cache_->reset_counts();
  reset_kv();
  auto logits = prefill(prompt);
  int next = 0;
  for (int step = 0; step < n_tokens; step++) {
    const float* p = logits.data<float>();
    next = (int)(std::max_element(p, p + spec_.vocab_size) - p);
    logits = forward_rows({next}, (int)prompt.size() + step, true);
  }
  reset_kv();
}

int Engine::pin(int per_layer, const std::vector<std::vector<int>>* ranking) {
  return pin_per_layer(std::vector<int>(spec_.n_layers, per_layer), ranking);
}

int Engine::pin_per_layer(const std::vector<int>& per_layer, const std::vector<std::vector<int>>* ranking) {
  const auto& backfill = ranking ? *ranking : static_ranking_;
  const auto& counts = cache_->access_counts();
  std::vector<std::vector<int>> order(spec_.n_layers);
  for (int l = 0; l < spec_.n_layers; l++) {
    std::vector<int> used;
    for (int e = 0; e < spec_.n_experts; e++)
      if (counts[(size_t)l * spec_.n_experts + e] > 0) used.push_back(e);
    std::stable_sort(used.begin(), used.end(), [&](int a, int b) {
      return counts[(size_t)l * spec_.n_experts + a] > counts[(size_t)l * spec_.n_experts + b];
    });
    order[l] = used;
    if (l < (int)backfill.size())
      for (int e : backfill[l])
        if (std::find(order[l].begin(), order[l].end(), e) == order[l].end()) order[l].push_back(e);
  }
  double t0 = now_s();
  int n = cache_->pin(order, per_layer, std::max(2, opts_.io_threads));
  cache_->reset_stats();
  cache_->reset_counts();
  if (opts_.verbose)
    MMR_INFO("Pinned %d experts (%s) in %.1fs", n, human_bytes((double)cache_->pinned_bytes()).c_str(), now_s() - t0);
  apply_wired_limit();
  return n;
}

int Engine::suggest_pins(double ram_fraction) const {
  double ram = (double)physical_memory();
  double kv = (double)kv_bytes_ + (double)opts_.kv_ram_bytes;
  double budget = ram * ram_fraction - (double)base_.nbytes - (double)opts_.cold_budget_bytes - kv - (double)(1 << 30);
  if (budget <= 0) return 0;
  int per_layer = (int)(budget / ((double)layout_.payload_bytes() * spec_.n_layers));
  return std::max(0, std::min(per_layer, spec_.n_experts));
}

void Engine::apply_wired_limit() {
  if (!opts_.wire) return;
  uint64_t want = base_.nbytes + cache_->pinned_bytes() + opts_.cold_budget_bytes + kv_bytes_ + ((uint64_t)512 << 20);
  uint64_t ram = physical_memory();
  if (ram > 0) want = std::min(want, (uint64_t)(ram * 0.7));
  // MLX rejects limits above the system wired limit; back off until one is accepted.
  for (int attempt = 0; attempt < 6 && want > ((uint64_t)1 << 30); attempt++) {
    try {
      mx::set_wired_limit(want);
      wired_bytes_ = want;
      if (opts_.verbose) MMR_INFO("Wired limit set to %s", human_bytes((double)want).c_str());
      return;
    } catch (const std::exception& ex) {
      want = want * 3 / 4;
    }
  }
  MMR_WARN("could not set a wired limit; weights may be paged by the OS under memory pressure");
}

void Engine::warmup() {
  // MLX specializes kernels on first use; run the shapes decode and prefill
  // will hit so the first request does not pay for it.
  double t0 = now_s();
  std::vector<int> toks;
  int chunk = std::max(1, std::min(opts_.prefill_chunk, 16));
  for (int i = 0; i < chunk + 1; i++) toks.push_back(1000 + i * 37);
  reset_kv();
  auto saved_bias = opts_.routing.bias;
  opts_.routing.bias = 8.0f; // stay on pinned experts as far as possible
  auto logits = prefill(toks);
  forward_rows({toks.back()}, (int)toks.size(), true);
  opts_.routing.bias = saved_bias;
  reset_kv();
  reset_stats();
  if (opts_.verbose) MMR_INFO("Warmup done in %.1fs", now_s() - t0);
}

// --- forward ---------------------------------------------------------------------

void Engine::reset_kv() {
  kv_len_ = 0;
  kv_tokens_.clear();
  ckpt_pos_ = -1;
  for (auto& b : kv_) b.set_state(0, 0);
}

bool Engine::kv_holds(int m) const {
  for (auto& b : kv_)
    if (!b.holds(m)) return false;
  return true;
}

bool Engine::ckpt_holds(int m) const {
  if (ckpt_pos_ < 0 || m > ckpt_pos_) return false;
  for (auto& b : kv_)
    if (b.ring() && !kv_ring_ok(b.window(), ckpt_lo_, m)) return false;
  return true;
}

KVPtrs Engine::live_ptrs() {
  KVPtrs p;
  for (auto& b : kv_) {
    p.k.push_back(b.k_data());
    p.v.push_back(b.v_data());
  }
  return p;
}

void Engine::save_ckpt() {
  ckpt_pos_ = kv_len_;
  ckpt_lo_ = 0;
  for (size_t l = 0; l < kv_.size(); l++)
    if (kv_[l].ring()) {
      ckpt_lo_ = std::max(ckpt_lo_, kv_[l].lo());
      memcpy(ckpt_k_[l].data(), kv_[l].k_data(), ckpt_k_[l].size() * sizeof(float));
      memcpy(ckpt_v_[l].data(), kv_[l].v_data(), ckpt_v_[l].size() * sizeof(float));
    }
}

void Engine::restore_ckpt() {
  for (size_t l = 0; l < kv_.size(); l++)
    if (kv_[l].ring()) {
      memcpy(kv_[l].k_data(), ckpt_k_[l].data(), ckpt_k_[l].size() * sizeof(float));
      memcpy(kv_[l].v_data(), ckpt_v_[l].data(), ckpt_v_[l].size() * sizeof(float));
      kv_[l].set_state(ckpt_pos_, ckpt_lo_);
    }
}

void Engine::save_live(uint64_t keep) {
  // Ring models are saved at the prompt checkpoint (what a returning conversation
  // continues from); others with everything generated since.
  int n = has_ring_ ? ckpt_pos_ : kv_len_;
  if (n < kMinSaveTokens) return;
  KVPtrs src = live_ptrs();
  for (size_t l = 0; l < kv_.size(); l++)
    if (has_ring_ && kv_[l].ring()) {
      src.k[l] = ckpt_k_[l].data();
      src.v[l] = ckpt_v_[l].data();
    }
  store_->put(std::vector<int>(kv_tokens_.begin(), kv_tokens_.begin() + n), src, has_ring_ ? ckpt_lo_ : 0, keep);
}

void Engine::attention(int layer, mx::array& X, int start_pos, int k) {
  const auto& L = base_.layers[layer];
  const auto& S = spec_;
  int H = S.n_heads, Hkv = S.n_kv_heads, hd = S.head_dim;
  auto h = mx::fast::rms_norm(X, mx::astype(L.attn_norm, mx::float32), S.rms_eps);
  auto q = mx::reshape(L.q(h), {1, k, H, hd});
  auto kk = mx::reshape(L.k(h), {1, k, Hkv, hd});
  auto v = mx::reshape(L.v(h), {1, k, Hkv, hd});
  if (S.qk_norm) {
    q = mx::fast::rms_norm(q, mx::astype(*L.q_norm, mx::float32), S.rms_eps);
    kk = mx::fast::rms_norm(kk, mx::astype(*L.k_norm, mx::float32), S.rms_eps);
  }
  q = mx::transpose(q, {0, 2, 1, 3});   // [1, H, k, hd]
  kk = mx::transpose(kk, {0, 2, 1, 3}); // [1, Hkv, k, hd]
  v = mx::transpose(v, {0, 2, 1, 3});
  int rd = S.rope_dims > 0 ? S.rope_dims : hd;
  q = mx::fast::rope(q, rd, false, std::nullopt, 1.0f, start_pos, rope_freqs_);
  kk = mx::fast::rope(kk, rd, false, std::nullopt, 1.0f, start_pos, rope_freqs_);
  if (rope_mscale_ != 1.f) {
    q = mx::multiply(q, mx::array(rope_mscale_));
    kk = mx::multiply(kk, mx::array(rope_mscale_));
  }
  int end = start_pos + k;
  if (end > opts_.kv_max) throw std::runtime_error("context length exceeded (kv_max=" + std::to_string(opts_.kv_max) + ")");
  kv_[layer].write(kk, v, start_pos);
  // Sliding-window rows attend to positions > q - window: read only those.
  int window = S.is_swa_layer(layer) ? S.sliding_window : 0;
  int first = window > 0 ? std::max(0, start_pos - window + 1) : 0;
  auto [K, V] = kv_[layer].read(first, end);
  std::optional<mx::array> sinks;
  if (L.sinks) sinks = mx::astype(*L.sinks, mx::float32);
  float scale = 1.0f / sqrtf((float)hd);
  mx::array o = q;
  if (k == 1) {
    o = mx::fast::scaled_dot_product_attention(q, K, V, scale, "", std::nullopt, sinks);
  } else {
    // causal + optional sliding window mask over [k, end - first]
    auto qpos = mx::expand_dims(mx::arange(start_pos, end, mx::int32), 1);
    auto kpos = mx::expand_dims(mx::arange(first, end, mx::int32), 0);
    auto mask = mx::less_equal(kpos, qpos);
    if (window > 0) mask = mx::logical_and(mask, mx::greater(kpos, mx::subtract(qpos, mx::array(window, mx::int32))));
    o = mx::fast::scaled_dot_product_attention(q, K, V, scale, "", mask, sinks);
  }
  o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), {k, H * hd});
  X = mx::add(X, L.o(o));
}

void Engine::moe(int layer, const mx::array& N, mx::array& X, int k, const std::vector<int>& experts,
                 const std::vector<std::vector<std::pair<int, float>>>& rows_of) {
  const auto& L = base_.layers[layer];
  const auto& gq = layout_.quant.at("gate");
  const auto& uq = layout_.quant.at("up");
  const auto& dq = layout_.quant.at("down");
  const auto& gp = cache_->parts("gate");
  const auto& up = cache_->parts("up");
  const auto& dp = cache_->parts("down");
  bool gptoss = spec_.activation == "gptoss";
  int D = spec_.embed_dim;

  auto activation = [&](mx::array g, mx::array u) {
    if (gptoss) {
      g = mx::minimum(g, mx::array(7.0f));
      u = mx::clip(u, mx::array(-7.0f), mx::array(7.0f));
      return mx::multiply(mx::multiply(g, mx::sigmoid(mx::multiply(g, mx::array(1.702f)))), mx::add(u, mx::array(1.0f)));
    }
    return mx::multiply(mx::multiply(g, mx::sigmoid(g)), u);
  };
  auto accumulate = [&](mx::array& out, const mx::array& y, const std::vector<int>& ridx) {
    if (k == 1) {
      out = mx::add(out, mx::sum(y, 0, true));
      return;
    }
    mx::array idx(ridx.data(), {(int)ridx.size()}, mx::int32);
    auto bidx = mx::broadcast_to(mx::expand_dims(idx, 1), {(int)ridx.size(), D});
    out = mx::scatter_add_axis(out, bidx, y, 0);
  };

  // Split into pinned (row, slot) pairs and cold experts.
  const PinnedBank* bank = cache_->bank(layer);
  std::vector<int> p_rows, p_slots, p_experts;
  std::vector<float> p_w;
  std::vector<int> cold;
  std::vector<std::vector<std::pair<int, float>>> cold_rows;
  for (size_t i = 0; i < experts.size(); i++) {
    int slot = cache_->slot(layer, experts[i]);
    if (slot >= 0) {
      cache_->note_pinned(layer, experts[i]);
      for (auto& [r, w] : rows_of[i]) {
        p_rows.push_back(r);
        p_slots.push_back(slot);
        p_experts.push_back(experts[i]);
        p_w.push_back(w);
      }
    } else {
      cold.push_back(experts[i]);
      cold_rows.push_back(rows_of[i]);
    }
  }

  mx::array out = mx::zeros({k, D}, mx::float32);

  if (!p_rows.empty()) {
    int n = (int)p_rows.size();
    // Sort pairs by slot so gather_qmm can use its sorted-indices path.
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p_slots[a] < p_slots[b]; });
    std::vector<int> rows(n), experts_sorted(n);
    std::vector<uint32_t> slots(n);
    std::vector<float> w(n);
    for (int i = 0; i < n; i++) {
      rows[i] = p_rows[order[i]];
      slots[i] = (uint32_t)p_slots[order[i]];
      experts_sorted[i] = p_experts[order[i]];
      w[i] = p_w[order[i]];
    }
    mx::array slot_idx(slots.data(), {n, 1}, mx::uint32);
    mx::array row_idx(rows.data(), {n}, mx::int32);
    mx::array e_idx(experts_sorted.data(), {n}, mx::int32);
    mx::array wts(w.data(), {n, 1}, mx::float32);
    mx::array xg = k == 1 ? mx::broadcast_to(N, {n, D}) : mx::take(N, row_idx, 0);
    xg = mx::reshape(xg, {n, 1, 1, D});
    auto gather = [&](const mx::array& x, const MatrixParts& p, const QuantDesc& q, int out_dim) {
      mx::array y = x;
      if (!q.quantized()) {
        auto Wsel = mx::astype(mx::take(bank->parts[p.w], mx::reshape(slot_idx, {n}), 0), mx::float32); // [n, out, in]
        y = mx::matmul(x, mx::transpose(Wsel, {0, 2, 1}));
        return mx::reshape(y, {n, out_dim});
      }
      std::optional<mx::array> b;
      if (p.biases >= 0) b = bank->parts[p.biases];
      y = mx::gather_qmm(x, bank->parts[p.w], bank->parts[p.scales], b, std::nullopt, slot_idx, true, q.group, q.bits,
                         q.mode, true);
      return mx::reshape(y, {n, out_dim});
    };
    auto g = gather(xg, gp, gq, spec_.expert_inter);
    auto u = gather(xg, up, uq, spec_.expert_inter);
    if (L.gate_bias) g = mx::add(g, mx::take(*L.gate_bias, e_idx, 0));
    if (L.up_bias) u = mx::add(u, mx::take(*L.up_bias, e_idx, 0));
    auto hdn = mx::reshape(activation(g, u), {n, 1, 1, spec_.expert_inter});
    auto y = gather(hdn, dp, dq, D);
    if (L.down_bias) y = mx::add(y, mx::take(*L.down_bias, e_idx, 0));
    y = mx::multiply(y, wts);
    accumulate(out, y, rows);
  }

  if (!cold.empty()) {
    auto views = cache_->get_cold(layer, cold);
    auto linear = [&](const mx::array& x, const ExpertViews& v, const MatrixParts& p, const QuantDesc& q) {
      if (!q.quantized()) return mx::matmul(x, mx::transpose(mx::astype(v.parts[p.w], mx::float32)));
      std::optional<mx::array> b;
      if (p.biases >= 0) b = v.parts[p.biases];
      return mx::quantized_matmul(x, v.parts[p.w], v.parts[p.scales], b, true, q.group, q.bits, q.mode);
    };
    for (size_t i = 0; i < cold.size(); i++) {
      int e = cold[i];
      std::vector<int> ridx;
      std::vector<float> rw;
      for (auto& [r, w] : cold_rows[i]) {
        ridx.push_back(r);
        rw.push_back(w);
      }
      mx::array xe = N;
      if (k > 1) xe = mx::take(N, mx::array(ridx.data(), {(int)ridx.size()}, mx::int32), 0);
      auto g = linear(xe, views[i], gp, gq);
      auto u = linear(xe, views[i], up, uq);
      if (L.gate_bias) g = mx::add(g, mx::take(*L.gate_bias, e, 0));
      if (L.up_bias) u = mx::add(u, mx::take(*L.up_bias, e, 0));
      auto y = linear(activation(g, u), views[i], dp, dq);
      if (L.down_bias) y = mx::add(y, mx::take(*L.down_bias, e, 0));
      y = mx::multiply(y, mx::array(rw.data(), {(int)rw.size(), 1}, mx::float32));
      accumulate(out, y, ridx);
    }
  }
  X = mx::add(X, out);
}

mx::array Engine::forward_rows(const std::vector<int>& tokens, int start_pos, bool last_only) {
  const auto& S = spec_;
  int k = (int)tokens.size();
  int Lc = S.n_layers, K = S.n_experts_used, E = S.n_experts;
  auto& ec = *cache_;
  double t_pass = now_s();

  mx::array X = base_.embed_tokens(tokens); // [k, D] float32
  bool need_resident = opts_.routing.bias != 0.0f;
  std::vector<uint8_t> resident;
  std::vector<float> logits_host((size_t)k * E);
  std::vector<int> idx;
  std::vector<float> wts;
  std::vector<std::vector<int>> la_cand;
  std::vector<uint8_t> requested((size_t)Lc * E, 0);

  ec.new_token();
  for (int i = 0; i < Lc; i++) {
    double t0 = now_s();
    ec.begin_layer(i);
    attention(i, X, start_pos, k);

    const auto& L = base_.layers[i];
    auto N = mx::fast::rms_norm(X, mx::astype(L.ffn_norm, mx::float32), S.rms_eps);
    auto logits = mx::matmul(N, mx::transpose(mx::astype(L.router_w, mx::float32)));
    if (L.router_b) logits = mx::add(logits, mx::astype(*L.router_b, mx::float32));
    int l0 = i + 1, l1 = std::min(Lc, i + 1 + opts_.lookahead_depth);
    // Lookahead only pays off for single-token decode; a prefill chunk already
    // issues a whole layer's misses at once and is bandwidth-bound.
    bool do_la = opts_.lookahead_depth > 0 && l0 < l1 && k == 1;
    std::optional<mx::array> la;
    if (do_la) la = lookahead_->logits(l0, l1, X);
    double t1 = now_s();
    if (do_la) mx::eval(logits, *la);
    else mx::eval(logits);
    double t2 = now_s();
    timers_.attn_s += t1 - t0;
    timers_.sync_s += t2 - t1;

    // Routing (CPU).
    const float* lp = logits.data<float>();
    if (need_resident) ec.resident_mask(i, resident);
    std::vector<int> experts;
    std::vector<std::vector<std::pair<int, float>>> rows_of;
    for (int r = 0; r < k; r++) {
      router_->select(lp + (size_t)r * E, need_resident ? resident.data() : nullptr, opts_.routing, false, idx, wts);
      for (size_t j = 0; j < idx.size(); j++) {
        auto it = std::find(experts.begin(), experts.end(), idx[j]);
        if (it == experts.end()) {
          experts.push_back(idx[j]);
          rows_of.push_back({{r, wts[j]}});
        } else {
          rows_of[it - experts.begin()].push_back({r, wts[j]});
        }
      }
    }
    // Sort experts so pinned ones come first (their arrays are ready) -- order is irrelevant to the math.
    {
      std::vector<size_t> order(experts.size());
      for (size_t j = 0; j < order.size(); j++) order[j] = j;
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return experts[a] < experts[b]; });
      std::vector<int> e2;
      std::vector<std::vector<std::pair<int, float>>> r2;
      for (auto j : order) {
        e2.push_back(experts[j]);
        r2.push_back(rows_of[j]);
      }
      experts.swap(e2);
      rows_of.swap(r2);
    }
    double t3 = now_s();
    timers_.route_s += t3 - t2;

    // Lookahead prefetch for layers l0..l1-1.
    if (do_la) {
      lookahead_->candidates(la->data<float>(), k, l1 - l0, K + opts_.lookahead_extra, la_cand);
      ec.retarget(l0, std::set<int>(la_cand[0].begin(), la_cand[0].end()));
      for (int j = 0; j < l1 - l0; j++) {
        std::vector<int> fresh;
        for (int e : la_cand[j]) {
          size_t key = (size_t)(l0 + j) * E + e;
          if (requested[key] || ec.is_resident(l0 + j, e)) continue;
          requested[key] = 1;
          fresh.push_back(e);
        }
        if (!fresh.empty()) ec.prefetch_many(l0 + j, fresh, PRIO_LOOKAHEAD + j + 1);
      }
    }
    double t4 = now_s();
    timers_.prefetch_s += t4 - t3;

    moe(i, N, X, k, experts, rows_of);
    mx::async_eval(X);
    double t5 = now_s();
    timers_.moe_s += t5 - t4;
  }

  double t0 = now_s();
  mx::array Xo = last_only && k > 1 ? mx::slice(X, {k - 1, 0}, {k, S.embed_dim}) : X;
  auto h = mx::fast::rms_norm(Xo, mx::astype(base_.output_norm, mx::float32), S.rms_eps);
  auto logits = base_.output(h);
  mx::eval(logits);
  timers_.head_s += now_s() - t0;

  kv_len_ = start_pos + k;
  timers_.rows += k;
  timers_.passes++;
  (void)t_pass;
  return logits;
}

mx::array Engine::prefill(const std::vector<int>& tokens, int* n_cached, std::string* source) {
  if (tokens.empty()) throw std::runtime_error("empty prompt");
  if ((int)tokens.size() > opts_.kv_max)
    throw std::runtime_error("prompt of " + std::to_string(tokens.size()) + " tokens exceeds the context (kv_max=" +
                             std::to_string(opts_.kv_max) + ")");
  int max_reuse = (int)tokens.size() - 1; // always forward the last token for its logits
  int common = 0;
  while (common < (int)kv_tokens_.size() && common < max_reuse && kv_tokens_[common] == tokens[common]) common++;
  common = std::min(common, kv_len_);

  // 1. The active cache, as is or rolled back to the prompt checkpoint (rings only hold
  //    the last window + slack positions, so a long rollback needs the checkpoint).
  int live = kv_holds(common) ? common : 0;
  bool use_ckpt = false;
  if (has_ring_ && ckpt_pos_ > 0) {
    int c = std::min(common, ckpt_pos_);
    if (c > live && ckpt_holds(c)) {
      live = c;
      use_ckpt = true;
    }
  }
  // 2. The session store.
  KVStore::Match hit;
  if (store_) hit = store_->find(tokens, std::min(max_reuse, opts_.kv_max));
  bool from_store = hit.len > live;
  // 3. A request that does not continue the previous prompt abandons that session: save it
  //    first. (Continuing turns and regenerations keep everything up to the last prompt token.)
  int kept = from_store ? 0 : live;
  if (store_ && ckpt_pos_ > 0 && kept + 1 < ckpt_pos_) save_live(hit.id);

  std::string src = "none";
  if (from_store) {
    if (store_->take(hit.id, hit.len, live_ptrs())) {
      kv_len_ = hit.len;
      src = tier_name(hit.tier);
      for (auto& b : kv_) b.ring() ? b.set_state(hit.n, hit.lo) : b.set_state(hit.len, 0);
    } else {
      kv_len_ = 0;
      for (auto& b : kv_) b.set_state(0, 0);
    }
    ckpt_pos_ = -1;
  } else {
    if (use_ckpt) restore_ckpt();
    if (live > 0) src = use_ckpt ? "ckpt" : "live";
    kv_len_ = live;
    if (live < ckpt_pos_) ckpt_pos_ = -1; // positions past `live` are about to be rewritten
  }
  kv_tokens_.assign(tokens.begin(), tokens.begin() + kv_len_);
  if (n_cached) *n_cached = kv_len_;
  if (source) *source = src;

  int chunk = std::max(1, opts_.prefill_chunk);
  // Keep a chunk's worst-case cold footprint under the cold budget.
  double pin_frac = (double)cache_->n_pinned() / ((double)spec_.n_layers * spec_.n_experts);
  while (chunk > 4) {
    double experts_per_layer = std::min((double)spec_.n_experts, (double)chunk * spec_.n_experts_used);
    double cold_bytes = experts_per_layer * (1.0 - pin_frac) * (double)layout_.record_bytes;
    if (cold_bytes <= 0.8 * (double)opts_.cold_budget_bytes) break;
    chunk /= 2;
  }
  mx::array logits = mx::array(0.f);
  for (size_t i = kv_len_; i < tokens.size(); i += chunk) {
    size_t n = std::min((size_t)chunk, tokens.size() - i);
    std::vector<int> part(tokens.begin() + i, tokens.begin() + i + n);
    logits = forward_rows(part, (int)i, true);
    kv_tokens_.insert(kv_tokens_.end(), part.begin(), part.end());
  }
  save_ckpt();
  return logits;
}

GenerateResult Engine::generate(const std::vector<int>& prompt, int max_tokens, const SamplingParams& sampling,
                                const std::vector<int>& stop_ids, const std::function<bool(int)>& on_token) {
  GenerateResult res;
  res.n_prompt = (int)prompt.size();
  Sampler sampler(sampling);
  double t0 = now_s();
  auto logits = prefill(prompt, &res.n_cached, &res.cache_source);
  res.prefill_s = now_s() - t0;
  t0 = now_s();
  int pos = (int)prompt.size();
  for (int step = 0; step < max_tokens; step++) {
    if (pos >= opts_.kv_max) {
      res.finish_reason = "length";
      break;
    }
    int tok = sampler.sample(logits.data<float>(), spec_.vocab_size);
    res.n_generated++;
    kv_tokens_.push_back(tok);
    if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end()) {
      res.finish_reason = "stop";
      on_token(tok);
      break;
    }
    if (!on_token(tok)) {
      res.finish_reason = "abort";
      break;
    }
    if (step + 1 < max_tokens) logits = forward_rows({tok}, pos, true);
    pos++;
  }
  res.decode_s = now_s() - t0;
  return res;
}

EngineStats Engine::stats() const {
  EngineStats s = timers_;
  s.cache = cache_->snapshot_stats();
  s.router = router_->stats();
  return s;
}

void Engine::reset_stats() {
  timers_ = EngineStats{};
  cache_->reset_stats();
  router_->reset_stats();
}

nlohmann::json Engine::stats_json() const {
  auto s = stats();
  nlohmann::json j;
  j["rows"] = s.rows;
  j["passes"] = s.passes;
  j["time"] = {{"attn_s", s.attn_s}, {"sync_s", s.sync_s}, {"route_s", s.route_s}, {"prefetch_s", s.prefetch_s},
               {"moe_s", s.moe_s}, {"head_s", s.head_s}};
  j["cache"] = {{"accesses", s.cache.accesses}, {"pinned_hits", s.cache.pinned_hits}, {"cold_hits", s.cache.cold_hits},
                {"misses", s.cache.misses}, {"pinned", cache_->n_pinned()}, {"pinned_bytes", cache_->pinned_bytes()}};
  const auto& l = s.cache.loader;
  j["loader"] = {{"bytes_read", l.bytes_read}, {"n_reads", l.n_reads}, {"demand_wait_s", l.demand_wait_s},
                 {"n_demand", l.n_demand}, {"n_prefetch_hit", l.n_prefetch_hit}, {"n_prefetch_wasted", l.n_prefetch_wasted},
                 {"n_requests", l.n_requests}, {"n_evict", l.n_evict}, {"n_dropped", l.n_dropped},
                 {"n_overflow", l.n_overflow}, {"read_time_s", l.read_time_s}, {"resident_bytes", l.resident_bytes},
                 {"n_resident", l.n_resident}, {"queued", l.queued}};
  j["router"] = {{"slots", s.router.slots}, {"swapped", s.router.swapped}, {"displaced_mass", s.router.displaced_mass},
                 {"restricted_fallback", s.router.restricted_fallback}};
  j["memory"] = {{"active", mx::get_active_memory()}, {"peak", mx::get_peak_memory()}, {"cache", mx::get_cache_memory()},
                 {"rss", resident_memory()}, {"wired_limit", wired_bytes_}};
  j["kv"] = {{"len", kv_len_}, {"bytes", kv_bytes_}, {"ckpt_pos", ckpt_pos_},
             {"store", store_ ? store_->stats_json() : nlohmann::json(nullptr)}};
  return j;
}

} // namespace mmr
