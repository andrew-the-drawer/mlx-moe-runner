#include "convert/convert.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

#include "gguf/ggml_quants.h"
#include "gguf/gguf_reader.h"
#include "model/layout.h"
#include "model/spec.h"
#include "util/log.h"
#include "util/mlx_util.h"
#include "util/sysinfo.h"

namespace mx = mlx::core;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace mmr {

namespace {

using gguf::TensorInfo;
using gguf::TensorType;

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename F>
void parallel_for(size_t n, int threads, F&& fn) {
  if (threads <= 1 || n <= 1) {
    for (size_t i = 0; i < n; i++) fn(i);
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  std::exception_ptr err;
  std::mutex err_mutex;
  for (int t = 0; t < threads; t++) {
    pool.emplace_back([&]() {
      try {
        for (size_t i = next++; i < n; i = next++) fn(i);
      } catch (...) {
        std::lock_guard<std::mutex> lock(err_mutex);
        if (!err) err = std::current_exception();
      }
    });
  }
  for (auto& t : pool) t.join();
  if (err) std::rethrow_exception(err);
}

// Chooses the MLX layout for a ggml tensor type.
QuantDesc choose_quant(TensorType t, const ConvertOptions& opts) {
  QuantDesc q;
  quant::MlxQuant mq;
  if (quant::exact_mlx_quant(t, &mq)) {
    q.mode = mq.mode;
    q.bits = mq.bits;
    q.group = mq.group;
    q.has_biases = mq.has_biases;
    return q;
  }
  if (t == TensorType::F32) { q.mode = "none"; q.dtype = "float32"; return q; }
  if (t == TensorType::F16) { q.mode = "none"; q.dtype = "float16"; return q; }
  if (t == TensorType::BF16) { q.mode = "none"; q.dtype = "bfloat16"; return q; }
  if (!quant::can_dequantize(t))
    throw std::runtime_error(std::string("unsupported ggml tensor type ") + gguf::tensor_type_name(t));
  q.mode = "affine";
  q.bits = opts.lossy_bits;
  q.group = opts.lossy_group;
  q.has_biases = true;
  return q;
}

// A converted matrix in MLX layout, as raw bytes.
struct MatrixBytes {
  std::vector<uint8_t> w, scales, biases;
  std::vector<int> w_shape, s_shape;
  std::string w_dtype, s_dtype;
};

struct MatrixSizes {
  uint64_t w_bytes, s_bytes, b_bytes;
  std::vector<int> w_shape, s_shape;
  std::string w_dtype, s_dtype;
};

MatrixSizes matrix_sizes(const QuantDesc& q, int out_dim, int in_dim) {
  MatrixSizes m{};
  if (!q.quantized()) {
    m.w_bytes = (uint64_t)out_dim * in_dim * dtype_size(q.dtype);
    m.w_shape = {out_dim, in_dim};
    m.w_dtype = q.dtype;
    m.s_bytes = m.b_bytes = 0;
    return m;
  }
  int words = (int)(quant::packed_row_bytes(q.bits, in_dim) / 4);
  m.w_shape = {out_dim, words};
  m.w_dtype = "uint32";
  m.w_bytes = (uint64_t)out_dim * words * 4;
  int groups = in_dim / q.group;
  m.s_shape = {out_dim, groups};
  if (q.mode == "mxfp4") {
    m.s_dtype = "uint8";
    m.s_bytes = (uint64_t)out_dim * groups;
    m.b_bytes = 0;
  } else {
    m.s_dtype = "float16";
    m.s_bytes = (uint64_t)out_dim * groups * 2;
    m.b_bytes = q.has_biases ? m.s_bytes : 0;
  }
  return m;
}

// Dequantizes a [rows, cols] ggml matrix to f32.
void dequantize_matrix(TensorType t, const uint8_t* src, int rows, int cols, float* dst, int threads) {
  size_t bb = 0, be = 0;
  gguf::tensor_type_block(t, &bb, &be);
  size_t row_bytes = (size_t)cols / be * bb;
  parallel_for((size_t)rows, threads, [&](size_t r) {
    quant::dequantize_row(t, src + r * row_bytes, cols, dst + r * (size_t)cols);
  });
}

// Converts one [out_dim, in_dim] ggml matrix into MLX-layout bytes written to
// the given destinations (sized per matrix_sizes). Lossy paths use MLX's
// quantizer and must run on the main thread.
void convert_matrix(TensorType t, const uint8_t* src, int out_dim, int in_dim, const QuantDesc& q,
                    uint8_t* w, uint8_t* scales, uint8_t* biases, int threads) {
  size_t bb = 0, be = 0;
  gguf::tensor_type_block(t, &bb, &be);
  size_t src_row = (size_t)in_dim / be * bb;
  if (!q.quantized()) {
    if (t == TensorType::F32 || t == TensorType::F16 || t == TensorType::BF16) {
      memcpy(w, src, src_row * out_dim);
      return;
    }
    throw std::runtime_error("dense conversion of quantized tensor not supported");
  }
  quant::MlxQuant exact;
  bool exact_match = quant::exact_mlx_quant(t, &exact) && q.mode == exact.mode && q.bits == exact.bits &&
                     q.group == exact.group;
  if (exact_match) {
    size_t w_row = quant::packed_row_bytes(q.bits, in_dim);
    size_t groups = in_dim / q.group;
    size_t s_row = q.mode == "mxfp4" ? groups : groups * 2;
    parallel_for((size_t)out_dim, threads, [&](size_t r) {
      const uint8_t* s = src + r * src_row;
      uint8_t* wr = w + r * w_row;
      uint8_t* sr = scales + r * s_row;
      uint8_t* br = biases ? biases + r * s_row : nullptr;
      switch (t) {
        case TensorType::Q8_0: quant::repack_q8_0_row(s, in_dim, wr, sr, br); break;
        case TensorType::Q4_0: quant::repack_q4_0_row(s, in_dim, wr, sr, br); break;
        case TensorType::Q4_K: quant::repack_q4_k_row(s, in_dim, wr, sr, br); break;
        case TensorType::MXFP4: quant::repack_mxfp4_row(s, in_dim, wr, sr); break;
        default: throw std::runtime_error("no exact repack");
      }
    });
    return;
  }
  // Lossy: dequantize then quantize with MLX.
  std::vector<float> f((size_t)out_dim * in_dim);
  dequantize_matrix(t, src, out_dim, in_dim, f.data(), threads);
  mx::array a(f.data(), {out_dim, in_dim}, mx::float32);
  auto parts = mx::quantize(a, q.group, q.bits, "affine");
  // MLX keeps scales/biases in the input dtype; the on-disk layout uses float16.
  for (size_t i = 1; i < parts.size(); i++) parts[i] = mx::astype(parts[i], mx::float16);
  mx::eval(parts);
  auto sizes = matrix_sizes(q, out_dim, in_dim);
  if (parts[0].nbytes() != sizes.w_bytes || parts[1].nbytes() != sizes.s_bytes)
    throw std::runtime_error("unexpected quantized sizes from MLX");
  memcpy(w, parts[0].data<uint8_t>(), sizes.w_bytes);
  memcpy(scales, parts[1].data<uint8_t>(), sizes.s_bytes);
  if (parts.size() > 2 && biases) memcpy(biases, parts[2].data<uint8_t>(), sizes.b_bytes);
}

// Builds MLX arrays for a base (non-expert) matrix.
void convert_base_matrix(const TensorInfo& t, const QuantDesc& q, const std::string& base_name,
                         std::unordered_map<std::string, mx::array>& out, int threads) {
  int in_dim = (int)t.ne(0);
  int out_dim = (int)(t.n_elements() / t.ne(0));
  auto sizes = matrix_sizes(q, out_dim, in_dim);
  std::vector<uint8_t> w(sizes.w_bytes), s(sizes.s_bytes), b(sizes.b_bytes);
  convert_matrix(t.type, t.data, out_dim, in_dim, q, w.data(), s.empty() ? nullptr : s.data(),
                 b.empty() ? nullptr : b.data(), threads);
  out.emplace(base_name + ".weight", array_from_bytes(w.data(), mx::Shape(sizes.w_shape.begin(), sizes.w_shape.end()), dtype_from_name(sizes.w_dtype)));
  if (!s.empty())
    out.emplace(base_name + ".scales", array_from_bytes(s.data(), mx::Shape(sizes.s_shape.begin(), sizes.s_shape.end()), dtype_from_name(sizes.s_dtype)));
  if (!b.empty())
    out.emplace(base_name + ".biases", array_from_bytes(b.data(), mx::Shape(sizes.s_shape.begin(), sizes.s_shape.end()), mx::float16));
}

mx::array dense_f32(const TensorInfo& t, int threads) {
  std::vector<float> f(t.n_elements());
  int cols = (int)t.ne(0);
  int rows = (int)(t.n_elements() / cols);
  dequantize_matrix(t.type, t.data, rows, cols, f.data(), threads);
  mx::Shape shape;
  for (int i = (int)t.dims.size() - 1; i >= 0; i--) shape.push_back((int)t.dims[i]);
  return mx::array(f.data(), shape, mx::float32);
}

std::string strip_layer_prefix(const std::string& name) {
  // "blk.12.attn_q.weight" -> "attn_q"; "output.weight" -> "output"
  std::string n = name;
  if (n.rfind("blk.", 0) == 0) {
    size_t dot = n.find('.', 4);
    n = n.substr(dot + 1);
  }
  size_t last = n.rfind('.');
  if (last != std::string::npos) n = n.substr(0, last);
  return n;
}

const char* pre_tokenizer_regex(const std::string& pre) {
  if (pre == "gpt-4o" || pre == "o200k")
    return "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  if (pre == "qwen2" || pre == "deepseek-v3" || pre == "deepseek-llm")
    return "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  if (pre == "llama3" || pre == "llama-bpe")
    return "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  return nullptr;
}

void write_tokenizer(const gguf::Reader& r, const ModelSpec& spec, const fs::path& out) {
  const auto* tokens = r.get("tokenizer.ggml.tokens");
  const auto* merges = r.get("tokenizer.ggml.merges");
  const auto* types = r.get("tokenizer.ggml.token_type");
  if (!tokens || !merges) throw std::runtime_error("GGUF has no BPE vocabulary (tokens/merges)");
  std::string model = r.get_string("tokenizer.ggml.model", "gpt2");
  if (model != "gpt2") MMR_WARN("tokenizer model '%s' is not gpt2/BPE; results may be wrong", model.c_str());
  std::string pre = r.get_string("tokenizer.ggml.pre", "default");
  const char* regex = pre_tokenizer_regex(pre);
  if (!regex) {
    MMR_WARN("unknown pre-tokenizer '%s'; falling back to the GPT-2 pattern", pre.c_str());
    regex = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  }

  json vocab = json::object();
  json added = json::array();
  for (size_t i = 0; i < tokens->arr_str.size(); i++) {
    vocab[tokens->arr_str[i]] = (int)i;
    int type = types && i < types->arr_num.size() ? (int)types->arr_num[i] : 1;
    if (type == 3 || type == 4) {
      added.push_back(json{{"id", (int)i}, {"content", tokens->arr_str[i]}, {"single_word", false},
                           {"lstrip", false}, {"rstrip", false}, {"normalized", false},
                           {"special", type == 3}});
    }
  }
  json j;
  j["version"] = "1.0";
  j["truncation"] = nullptr;
  j["padding"] = nullptr;
  j["added_tokens"] = added;
  j["normalizer"] = nullptr;
  j["pre_tokenizer"] = json{{"type", "Sequence"}, {"pretokenizers", json::array({
      json{{"type", "Split"}, {"pattern", json{{"Regex", regex}}}, {"behavior", "Isolated"}, {"invert", false}},
      json{{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", false}, {"use_regex", false}}})}};
  j["post_processor"] = nullptr;
  j["decoder"] = json{{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}};
  j["model"] = json{{"type", "BPE"}, {"dropout", nullptr}, {"unk_token", nullptr}, {"continuing_subword_prefix", ""},
                    {"end_of_word_suffix", ""}, {"fuse_unk", false}, {"byte_fallback", false}, {"ignore_merges", false},
                    {"vocab", vocab}, {"merges", merges->arr_str}};
  j["mmr"] = json{{"pre", pre}, {"bos_id", spec.bos_id}, {"eos_id", spec.eos_id}, {"add_bos", spec.add_bos}};
  std::ofstream f(out);
  f << j.dump();
}

// Static expert ranking from router weights and random inputs; a fallback for
// pinning when no usage profile exists.
json static_profile(const gguf::Reader& r, const ModelSpec& spec, int threads) {
  json ranking = json::array();
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int n_samples = 512;
  std::vector<float> x((size_t)n_samples * spec.embed_dim);
  for (auto& v : x) v = nd(rng);
  mx::array X(x.data(), {n_samples, spec.embed_dim}, mx::float32);
  for (int l = 0; l < spec.n_layers; l++) {
    const auto* wt = r.tensor("blk." + std::to_string(l) + ".ffn_gate_inp.weight");
    if (!wt) { ranking.push_back(json::array()); continue; }
    auto W = dense_f32(*wt, threads); // [E, D]
    auto logits = mx::matmul(X, mx::transpose(W));
    if (spec.router_bias) {
      const auto* bt = r.tensor("blk." + std::to_string(l) + ".ffn_gate_inp.bias");
      if (bt) logits = mx::add(logits, dense_f32(*bt, threads));
    }
    auto idx = mx::argpartition(logits, spec.n_experts - spec.n_experts_used, 1);
    idx = mx::slice(idx, {0, spec.n_experts - spec.n_experts_used}, {n_samples, spec.n_experts});
    mx::eval(idx);
    std::vector<int> counts(spec.n_experts, 0);
    const uint32_t* p = idx.data<uint32_t>();
    for (size_t i = 0; i < idx.size(); i++) counts[p[i]]++;
    std::vector<int> order(spec.n_experts);
    for (int i = 0; i < spec.n_experts; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return counts[a] > counts[b]; });
    ranking.push_back(order);
  }
  return ranking;
}

struct ExpertTensorSet {
  const TensorInfo* gate;
  const TensorInfo* up;
  const TensorInfo* down;
};

ExpertTensorSet expert_tensors(const gguf::Reader& r, int layer) {
  std::string p = "blk." + std::to_string(layer) + ".";
  ExpertTensorSet s{r.tensor(p + "ffn_gate_exps.weight"), r.tensor(p + "ffn_up_exps.weight"),
                    r.tensor(p + "ffn_down_exps.weight")};
  if (!s.gate || !s.up || !s.down)
    throw std::runtime_error("layer " + std::to_string(layer) + ": expert tensors missing");
  return s;
}

uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

ExpertLayout build_layout(const gguf::Reader& r, const ModelSpec& spec, const ConvertOptions& opts,
                          const std::vector<int>& layers) {
  ExpertLayout L;
  L.n_layers = spec.n_layers;
  L.n_experts = spec.n_experts;
  L.align = 4096;
  auto ts = expert_tensors(r, layers[0]);
  const TensorInfo* mats[3] = {ts.gate, ts.up, ts.down};
  const char* names[3] = {"gate", "up", "down"};
  uint64_t off = 0;
  for (int i = 0; i < 3; i++) {
    const auto& t = *mats[i];
    if (t.dims.size() != 3 || (int)t.ne(2) != spec.n_experts)
      throw std::runtime_error(std::string("unexpected expert tensor shape for ") + t.name);
    QuantDesc q = choose_quant(t.type, opts);
    // The record layout is shared by all layers: if any layer stores this matrix in
    // a different ggml type, fall back to the lossy (near-lossless 8-bit) layout for all.
    bool uniform = true;
    for (int l : layers) {
      auto tl = expert_tensors(r, l);
      const TensorInfo* ml = i == 0 ? tl.gate : (i == 1 ? tl.up : tl.down);
      if (ml->type != t.type) uniform = false;
      if ((int)ml->ne(0) != (int)t.ne(0) || (int)ml->ne(1) != (int)t.ne(1))
        throw std::runtime_error("expert shapes differ between layers");
    }
    if (!uniform) {
      q.mode = "affine";
      q.bits = opts.lossy_bits;
      q.group = opts.lossy_group;
      q.has_biases = true;
      q.dtype = "float16";
      MMR_INFO("  experts.%s: mixed ggml types across layers; converting all layers to %d-bit affine", names[i], q.bits);
    }
    L.quant[names[i]] = q;
    int in_dim = (int)t.ne(0), out_dim = (int)t.ne(1);
    auto sz = matrix_sizes(q, out_dim, in_dim);
    auto add_part = [&](const std::string& n, const std::vector<int>& shape, const std::string& dt, uint64_t bytes) {
      PartDesc p;
      p.name = n;
      p.shape = shape;
      p.dtype = dt;
      p.offset = off;
      p.bytes = bytes;
      p.padded = align_up(bytes, L.align);
      off += p.padded;
      L.parts.push_back(p);
    };
    add_part(std::string(names[i]) + ".weight", sz.w_shape, sz.w_dtype, sz.w_bytes);
    if (sz.s_bytes) add_part(std::string(names[i]) + ".scales", sz.s_shape, sz.s_dtype, sz.s_bytes);
    if (sz.b_bytes) add_part(std::string(names[i]) + ".biases", sz.s_shape, "float16", sz.b_bytes);
  }
  L.record_bytes = align_up(off, 16384);
  return L;
}

// Verifies one expert record on disk against ggml dequantization.
double verify_expert(const fs::path& file, const ExpertLayout& L, const ExpertTensorSet& ts, int expert, int threads) {
  int fd = ::open(file.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("verify: cannot open " + file.string());
  std::vector<uint8_t> rec(L.record_bytes);
  ssize_t n = pread(fd, rec.data(), L.record_bytes, L.expert_offset(expert));
  ::close(fd);
  if (n != (ssize_t)L.record_bytes) throw std::runtime_error("verify: short read");
  double worst = 0;
  const TensorInfo* mats[3] = {ts.gate, ts.up, ts.down};
  const char* names[3] = {"gate", "up", "down"};
  for (int i = 0; i < 3; i++) {
    const auto& t = *mats[i];
    const auto& q = L.quant.at(names[i]);
    int in_dim = (int)t.ne(0), out_dim = (int)t.ne(1);
    size_t bb = 0, be = 0;
    gguf::tensor_type_block(t.type, &bb, &be);
    size_t expert_bytes = (size_t)out_dim * (in_dim / be) * bb;
    std::vector<float> ref((size_t)out_dim * in_dim);
    dequantize_matrix(t.type, t.data + expert_bytes * expert, out_dim, in_dim, ref.data(), threads);
    const auto* pw = L.part(std::string(names[i]) + ".weight");
    mx::array w = array_from_bytes(rec.data() + pw->offset, mx::Shape(pw->shape.begin(), pw->shape.end()), dtype_from_name(pw->dtype));
    mx::array deq = w;
    if (q.quantized()) {
      const auto* ps = L.part(std::string(names[i]) + ".scales");
      mx::array s = array_from_bytes(rec.data() + ps->offset, mx::Shape(ps->shape.begin(), ps->shape.end()), dtype_from_name(ps->dtype));
      std::optional<mx::array> b;
      if (const auto* pb = L.part(std::string(names[i]) + ".biases"))
        b = array_from_bytes(rec.data() + pb->offset, mx::Shape(pb->shape.begin(), pb->shape.end()), mx::float16);
      deq = mx::dequantize(w, s, b, q.group, q.bits, q.mode, std::nullopt, mx::float32);
    } else {
      deq = mx::astype(w, mx::float32);
    }
    mx::eval(deq);
    const float* d = deq.data<float>();
    double max_abs = 0, max_ref = 0;
    for (size_t k = 0; k < ref.size(); k++) {
      max_abs = std::max(max_abs, (double)fabsf(d[k] - ref[k]));
      max_ref = std::max(max_ref, (double)fabsf(ref[k]));
    }
    double rel = max_ref > 0 ? max_abs / max_ref : max_abs;
    worst = std::max(worst, rel);
  }
  return worst;
}

} // namespace

int run_convert(const ConvertOptions& opts) {
  double t_start = now_s();
  int threads = opts.threads > 0 ? opts.threads : performance_cores();
  gguf::Reader r(opts.gguf_path);
  ModelSpec spec = ModelSpec::from_gguf(r);
  MMR_INFO("Model: %s (%s), %d layers, %dx%d experts, d=%d, heads %d/%d x %d, vocab %d",
           spec.name.c_str(), spec.arch.c_str(), spec.n_layers, spec.n_experts, spec.n_experts_used,
           spec.embed_dim, spec.n_heads, spec.n_kv_heads, spec.head_dim, spec.vocab_size);

  fs::path out(opts.out_dir);
  fs::create_directories(out);

  std::vector<int> layers = opts.layers;
  if (layers.empty())
    for (int l = 0; l < spec.n_layers; l++) layers.push_back(l);

  ExpertLayout L = build_layout(r, spec, opts, layers);
  for (auto& [k, q] : L.quant)
    MMR_INFO("  experts.%s: %s%s", k.c_str(), q.mode.c_str(),
             q.quantized() ? (" " + std::to_string(q.bits) + "-bit g" + std::to_string(q.group)).c_str() : (" " + q.dtype).c_str());
  MMR_INFO("  expert record: %s payload, %s on disk; %s per layer",
           human_bytes((double)L.payload_bytes()).c_str(), human_bytes((double)L.record_bytes).c_str(),
           human_bytes((double)L.record_bytes * spec.n_experts).c_str());

  // --- base tensors --------------------------------------------------------
  MMR_INFO("Converting base tensors...");
  std::unordered_map<std::string, mx::array> base;
  std::map<std::string, QuantDesc> base_quant;
  for (const auto& t : r.tensors()) {
    if (t.name.find("_exps.weight") != std::string::npos) continue;
    std::string role = strip_layer_prefix(t.name);
    bool is_matrix = t.dims.size() == 2 && t.name.size() > 7 && t.name.compare(t.name.size() - 7, 7, ".weight") == 0 &&
                     (role == "attn_q" || role == "attn_k" || role == "attn_v" || role == "attn_output" ||
                      role == "output" || role == "token_embd" || role == "ffn_gate_inp");
    std::string base_name = t.name.substr(0, t.name.size() - 7);
    if (is_matrix && role != "ffn_gate_inp") {
      QuantDesc q = choose_quant(t.type, opts);
      base_quant[base_name] = q; // keyed by tensor: types may differ between layers
      convert_base_matrix(t, q, base_name, base, threads);
    } else {
      // norms, biases, routers, sinks, expert biases: dense f32
      base.emplace(t.name, dense_f32(t, threads));
    }
  }
  if (!base.count("output.weight")) {
    MMR_INFO("  no output.weight: tying to token embedding");
    base_quant["output"] = base_quant.at("token_embd");
    for (auto suffix : {".weight", ".scales", ".biases"})
      if (base.count(std::string("token_embd") + suffix)) base.emplace(std::string("output") + suffix, base.at(std::string("token_embd") + suffix));
  }
  mx::eval(std::vector<mx::array>{});
  uint64_t base_bytes = 0;
  for (auto& [k, a] : base) base_bytes += a.nbytes();
  MMR_INFO("  %zu tensors, %s", base.size(), human_bytes((double)base_bytes).c_str());
  mx::save_safetensors((out / "base.safetensors").string(), base, {{"format", "mlx-moe-runner"}});
  base.clear();

  // --- tokenizer + profile ---------------------------------------------------
  write_tokenizer(r, spec, out / "tokenizer.json");
  MMR_INFO("Computing static expert ranking...");
  json ranking = static_profile(r, spec, threads);

  // --- experts ---------------------------------------------------------------
  const TensorInfo* probe[3];
  for (int l : layers) L.files[l] = L.file_for(l);
  uint64_t total_written = 0;
  if (!opts.skip_experts) {
    for (size_t li = 0; li < layers.size(); li++) {
      int layer = layers[li];
      double t0 = now_s();
      auto ts = expert_tensors(r, layer);
      probe[0] = ts.gate; probe[1] = ts.up; probe[2] = ts.down;
      const char* names[3] = {"gate", "up", "down"};
      fs::path file = out / L.file_for(layer);
      int fd = ::open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd < 0) throw std::runtime_error("cannot create " + file.string());
      uint64_t file_bytes = L.record_bytes * spec.n_experts;
      if (ftruncate(fd, (off_t)file_bytes) != 0) throw std::runtime_error("ftruncate failed for " + file.string());

      auto exact_for = [&](const TensorInfo* t, const char* n) {
        quant::MlxQuant e;
        const auto& q = L.quant.at(n);
        return quant::exact_mlx_quant(t->type, &e) && q.mode == e.mode && q.bits == e.bits && q.group == e.group;
      };
      bool all_exact = exact_for(ts.gate, "gate") && exact_for(ts.up, "up") && exact_for(ts.down, "down");
      int outer_threads = all_exact ? threads : 1;
      int inner_threads = all_exact ? 1 : threads;
      std::vector<std::vector<uint8_t>> staging(outer_threads);
      std::mutex slot_mutex;
      std::vector<int> free_slots;
      for (int i = 0; i < outer_threads; i++) free_slots.push_back(i);

      parallel_for((size_t)spec.n_experts, outer_threads, [&](size_t e) {
        int slot;
        {
          std::lock_guard<std::mutex> lock(slot_mutex);
          slot = free_slots.back();
          free_slots.pop_back();
        }
        auto& rec = staging[slot];
        if (rec.size() != L.record_bytes) rec.assign(L.record_bytes, 0);
        for (int i = 0; i < 3; i++) {
          const auto& t = *probe[i];
          const auto& q = L.quant.at(names[i]);
          int in_dim = (int)t.ne(0), out_dim = (int)t.ne(1);
          size_t bb = 0, be = 0;
          gguf::tensor_type_block(t.type, &bb, &be);
          size_t expert_bytes = (size_t)out_dim * (in_dim / be) * bb;
          const auto* pw = L.part(std::string(names[i]) + ".weight");
          const auto* ps = L.part(std::string(names[i]) + ".scales");
          const auto* pb = L.part(std::string(names[i]) + ".biases");
          convert_matrix(t.type, t.data + expert_bytes * e, out_dim, in_dim, q, rec.data() + pw->offset,
                         ps ? rec.data() + ps->offset : nullptr, pb ? rec.data() + pb->offset : nullptr,
                         inner_threads);
        }
        ssize_t n = pwrite(fd, rec.data(), L.record_bytes, (off_t)L.expert_offset((int)e));
        if (n != (ssize_t)L.record_bytes) throw std::runtime_error("short write to " + file.string());
        {
          std::lock_guard<std::mutex> lock(slot_mutex);
          free_slots.push_back(slot);
        }
      });
      fsync(fd);
      ::close(fd);
      total_written += file_bytes;
      double dt = now_s() - t0;
      std::string vmsg;
      if (opts.verify) {
        double err = verify_expert(file, L, ts, 0, threads);
        char buf[64];
        snprintf(buf, sizeof(buf), ", verify max rel err %.2e", err);
        vmsg = buf;
        if (err > 0.05) MMR_WARN("layer %d: expert 0 mismatch after conversion (rel err %.3g)", layer, err);
      }
      MMR_INFO("  layer %3d (%zu/%zu): %s in %.1fs (%.0f MB/s)%s", layer, li + 1, layers.size(),
               human_bytes((double)file_bytes).c_str(), dt, file_bytes / 1048576.0 / std::max(dt, 1e-6), vmsg.c_str());
    }
  }

  // --- config ---------------------------------------------------------------
  json cfg;
  cfg["format"] = "mlx-moe-runner";
  cfg["format_version"] = 1;
  cfg["source"] = fs::path(opts.gguf_path).filename().string();
  cfg["spec"] = spec.to_json();
  cfg["experts"] = L.to_json();
  cfg["base_quant"] = json::object();
  for (auto& [k, q] : base_quant) cfg["base_quant"][k] = q.to_json();
  cfg["static_ranking"] = ranking;
  {
    std::ofstream f(out / "config.json");
    f << cfg.dump(2);
  }
  MMR_INFO("Done: %s of experts + base in %.0fs -> %s", human_bytes((double)total_written).c_str(),
           now_s() - t_start, out.string().c_str());
  return 0;
}

} // namespace mmr
