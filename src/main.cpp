// mlx-moe-runner: serve large Mixture-of-Experts models from SSD on Apple silicon.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "convert/convert.h"
#include "engine/engine.h"
#include "server/server.h"
#include "tokenizer/chat.h"
#include "util/log.h"
#include "util/sysinfo.h"

using json = nlohmann::json;
using namespace mmr;

namespace {

struct Args {
  std::vector<std::string> positional;
  std::map<std::string, std::string> flags;
  bool has(const std::string& k) const { return flags.count(k) > 0; }
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = flags.find(k);
    return it == flags.end() ? def : it->second;
  }
  int get_int(const std::string& k, int def) const { return has(k) ? std::stoi(get(k)) : def; }
  double get_double(const std::string& k, double def) const { return has(k) ? std::stod(get(k)) : def; }
};

Args parse_args(int argc, char** argv, int start) {
  Args a;
  for (int i = start; i < argc; i++) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      std::string key = s.substr(2), val = "true";
      size_t eq = key.find('=');
      if (eq != std::string::npos) {
        val = key.substr(eq + 1);
        key = key.substr(0, eq);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0 &&
                 key != "stats" && key != "verbose" && key != "quiet" && key != "chat" && key != "raw" &&
                 key != "no-wire" && key != "no-verify" && key != "exact" && key != "no-thinking" &&
                 key != "skip-experts" && key != "cached" && key != "help" && key != "warmup") {
        val = argv[++i];
      }
      a.flags[key] = val;
    } else if (s == "-p" && i + 1 < argc) {
      a.flags["prompt"] = argv[++i];
    } else if (s == "-n" && i + 1 < argc) {
      a.flags["tokens"] = argv[++i];
    } else if (s == "-h") {
      a.flags["help"] = "true";
    } else {
      a.positional.push_back(s);
    }
  }
  return a;
}

void usage() {
  fprintf(stderr,
          "mlx-moe-runner %s - run 100B-class MoE models from SSD on Apple silicon (MLX)\n\n"
          "Usage:\n"
          "  mlx-moe-runner convert <model.gguf> <out-dir> [--layers 0,1,2] [--lossy-bits 8] [--threads N] [--no-verify]\n"
          "  mlx-moe-runner run <model-dir> [-p PROMPT] [-n TOKENS] [--chat] [engine options] [--stats]\n"
          "  mlx-moe-runner serve <model-dir> [--host 127.0.0.1] [--port 8080] [engine options]\n"
          "  mlx-moe-runner profile <model-dir> --text FILE --out profile.json [engine options]\n"
          "  mlx-moe-runner bench <model-dir> [-n TOKENS] [--prompts FILE] [engine options]\n"
          "  mlx-moe-runner info <model-dir>\n\n"
          "Engine options:\n"
          "  --pin N           experts pinned in RAM per layer (default: auto from RAM, see --ram-fraction)\n"
          "  --ram-fraction F  share of physical RAM the model may use when --pin is auto (default 0.5)\n"
          "  --cold-mb MB      cold-expert buffer size (default 1536)\n"
          "  --io-threads N    SSD reader threads (default 8)\n"
          "  --io-split N      sub-reads per expert record: 0 = one per matrix (default), 1 = whole record\n"
          "  --bias B          cache-aware routing bias toward resident experts (default 0.5; 0 = exact routing)\n"
          "  --margin M        only swap an expert when its logit gap is below M\n"
          "  --lookahead J     prefetch experts predicted J layers ahead (default 0)\n"
          "  --extra X         extra lookahead candidates per layer (default 0)\n"
          "  --profile FILE    pin from a usage profile (see `profile`) instead of prompt calibration\n"
          "  --calibrate N     calibration tokens before pinning in `run` (default 8; 0 = static ranking)\n"
          "  --ctx N           KV cache length (default 4096)\n"
          "  --chunk N         prefill chunk size (default 32)\n"
          "  --no-wire         do not wire weights with MLX's wired limit\n"
          "  --warmup          run a warmup pass before timing (run; serve/bench always do)\n"
          "  --temp T --top-p P --top-k K --min-p M --seed S   sampling (run/bench)\n"
          "  --reasoning E     harmony reasoning effort: low|medium|high (default medium)\n"
          "  --quiet / --verbose\n",
          MMR_VERSION);
}

EngineOptions engine_options(const Args& a, const std::string& model_dir) {
  EngineOptions o;
  o.model_dir = model_dir;
  o.cold_budget_bytes = (uint64_t)(a.get_double("cold-mb", 1536.0) * 1048576.0);
  o.io_threads = a.get_int("io-threads", 8);
  o.io_split = a.get_int("io-split", 0);
  o.kv_max = a.get_int("ctx", 4096);
  o.prefill_chunk = a.get_int("chunk", 32);
  o.routing.bias = a.has("exact") ? 0.f : (float)a.get_double("bias", 0.5);
  if (a.has("margin")) o.routing.margin = (float)a.get_double("margin", 0);
  if (a.has("tau")) o.routing.tau = (float)a.get_double("tau", 0);
  o.routing.protect_top = a.get_int("protect-top", 0);
  o.lookahead_depth = a.get_int("lookahead", 0);
  o.lookahead_extra = a.get_int("extra", 0);
  o.wire = !a.has("no-wire");
  o.verbose = !a.has("quiet");
  return o;
}

SamplingParams sampling_options(const Args& a) {
  SamplingParams p;
  p.temperature = (float)a.get_double("temp", 0.7);
  p.top_p = (float)a.get_double("top-p", 0.9);
  p.top_k = a.get_int("top-k", 0);
  p.min_p = (float)a.get_double("min-p", 0.0);
  p.seed = (uint64_t)a.get_int("seed", 42);
  return p;
}

std::vector<std::vector<int>> load_profile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open profile " + path);
  json j = json::parse(f);
  return j.at("ranking").get<std::vector<std::vector<int>>>();
}

// Pins experts according to the flags; `calib_prompt` is used when no profile is given.
void setup_pins(Engine& engine, const Args& a, const std::vector<int>& calib_prompt, int calibrate_tokens) {
  int per_layer = a.has("pin") ? a.get_int("pin", 0) : engine.suggest_pins(a.get_double("ram-fraction", 0.5));
  if (!a.has("pin"))
    MMR_INFO("Auto pin budget: %d experts/layer (%s); override with --pin", per_layer,
             human_bytes((double)per_layer * engine.spec().n_layers * engine.expert_bytes()).c_str());
  if (per_layer <= 0) return;
  std::vector<std::vector<int>> ranking;
  const std::vector<std::vector<int>>* rp = nullptr;
  if (a.has("profile")) {
    ranking = load_profile(a.get("profile"));
    rp = &ranking;
  } else if (calibrate_tokens > 0 && !calib_prompt.empty()) {
    MMR_INFO("Calibrating on the prompt (%d tokens)...", calibrate_tokens);
    engine.calibrate(calib_prompt, calibrate_tokens);
  }
  engine.pin(per_layer, rp);
}

void print_stats(Engine& engine, int n_tokens, double seconds) {
  auto s = engine.stats();
  const auto& l = s.cache.loader;
  double per_tok = n_tokens > 0 ? 1000.0 / n_tokens : 0;
  fprintf(stderr,
          "\n--- stats over %d tokens (%.2f tok/s) ---\n"
          "  per token: attn+route graph %.1f ms, gpu sync %.1f ms, routing %.1f ms, prefetch %.1f ms, moe+io %.1f ms, head %.1f ms\n"
          "  experts: %llu accesses, pinned %.0f%%, cold-hit %.0f%%, miss %.0f%%\n"
          "  ssd: %.0f reads/tok, %.0f MB/tok, demand wait %.1f ms/tok, prefetch hits %llu wasted %llu, evictions %llu, overflow %llu\n"
          "  router: %llu of %llu slots swapped (%.0f%%)\n"
          "  memory: mlx active %s, mlx cache %s, rss %s\n",
          n_tokens, seconds > 0 ? n_tokens / seconds : 0.0,
          s.attn_s * per_tok, s.sync_s * per_tok, s.route_s * per_tok, s.prefetch_s * per_tok, s.moe_s * per_tok, s.head_s * per_tok,
          (unsigned long long)s.cache.accesses,
          s.cache.accesses ? 100.0 * s.cache.pinned_hits / s.cache.accesses : 0.0,
          s.cache.accesses ? 100.0 * s.cache.cold_hits / s.cache.accesses : 0.0,
          s.cache.accesses ? 100.0 * s.cache.misses / s.cache.accesses : 0.0,
          n_tokens ? (double)l.n_reads / n_tokens : 0.0, n_tokens ? l.bytes_read / 1048576.0 / n_tokens : 0.0,
          l.demand_wait_s * per_tok, (unsigned long long)l.n_prefetch_hit, (unsigned long long)l.n_prefetch_wasted,
          (unsigned long long)l.n_evict, (unsigned long long)l.n_overflow,
          (unsigned long long)s.router.swapped, (unsigned long long)s.router.slots,
          s.router.slots ? 100.0 * s.router.swapped / s.router.slots : 0.0,
          human_bytes((double)mx::get_active_memory()).c_str(), human_bytes((double)mx::get_cache_memory()).c_str(),
          human_bytes((double)resident_memory()).c_str());
}

int cmd_convert(const Args& a) {
  if (a.positional.size() < 2) {
    usage();
    return 2;
  }
  ConvertOptions o;
  o.gguf_path = a.positional[0];
  o.out_dir = a.positional[1];
  if (a.has("layers")) {
    std::stringstream ss(a.get("layers"));
    std::string item;
    while (std::getline(ss, item, ',')) o.layers.push_back(std::stoi(item));
  }
  o.lossy_bits = a.get_int("lossy-bits", 8);
  o.lossy_group = a.get_int("lossy-group", 32);
  o.threads = a.get_int("threads", 0);
  o.skip_experts = a.has("skip-experts");
  o.verify = !a.has("no-verify");
  return run_convert(o);
}

int cmd_info(const Args& a) {
  if (a.positional.empty()) { usage(); return 2; }
  std::ifstream f(a.positional[0] + "/config.json");
  if (!f) { MMR_ERROR("no config.json in %s", a.positional[0].c_str()); return 1; }
  json j = json::parse(f);
  auto spec = ModelSpec::from_json(j["spec"]);
  auto L = ExpertLayout::from_json(j["experts"]);
  printf("%s (%s)\n", spec.name.c_str(), spec.arch.c_str());
  printf("  layers %d, experts %d (top-%d), d=%d, inter=%d, heads %d/%d x %d, vocab %d, ctx %d\n", spec.n_layers,
         spec.n_experts, spec.n_experts_used, spec.embed_dim, spec.expert_inter, spec.n_heads, spec.n_kv_heads,
         spec.head_dim, spec.vocab_size, spec.context_length);
  printf("  chat format %s, activation %s, window %d, yarn %g\n", spec.chat_format.c_str(), spec.activation.c_str(),
         spec.sliding_window, spec.rope_yarn_factor);
  for (auto& [k, q] : L.quant)
    printf("  experts.%s: %s %d-bit g%d\n", k.c_str(), q.mode.c_str(), q.bits, q.group);
  printf("  expert record %s (%s on disk), all experts %s, converted layers %zu/%d\n",
         human_bytes((double)L.payload_bytes()).c_str(), human_bytes((double)L.record_bytes).c_str(),
         human_bytes((double)L.record_bytes * L.n_experts * spec.n_layers).c_str(), L.files.size(), spec.n_layers);
  if (j.contains("base_quant"))
    for (auto& [k, v] : j["base_quant"].items())
      printf("  %s: %s %d-bit g%d\n", k.c_str(), v.value("mode", "none").c_str(), v.value("bits", 0), v.value("group", 0));
  uint64_t ram = physical_memory();
  printf("  this machine: %s RAM; at 50%% budget ~%d experts/layer can be pinned\n", human_bytes((double)ram).c_str(),
         (int)((ram * 0.5 - 3.0e9 - 1.5e9) / ((double)L.payload_bytes() * spec.n_layers)));
  return 0;
}

int cmd_run(const Args& a) {
  if (a.positional.empty()) { usage(); return 2; }
  Engine engine(engine_options(a, a.positional[0]));
  const auto& spec = engine.spec();
  const auto& tok = engine.tokenizer();
  std::string prompt = a.get("prompt", "The key insight about mixture-of-experts models is that");
  bool chat = a.has("chat") || (spec.chat_format != "none" && !a.has("raw"));
  ChatOptions copts;
  copts.reasoning_effort = a.get("reasoning", "medium");
  copts.enable_thinking = !a.has("no-thinking");
  std::vector<int> stop_ids = spec.stop_ids;
  std::string text = prompt;
  if (chat) {
    std::vector<ChatMessage> msgs;
    if (a.has("system")) msgs.push_back({"system", a.get("system")});
    msgs.push_back({"user", prompt});
    text = format_chat(spec, msgs, copts);
    stop_ids = chat_stop_ids(spec, tok);
  }
  auto ids = tok.encode(text, !chat && spec.add_bos);
  MMR_INFO("Prompt: %zu tokens%s", ids.size(), chat ? " (chat template applied; use --raw for plain completion)" : "");

  setup_pins(engine, a, ids, a.get_int("calibrate", 8));
  if (a.has("warmup")) engine.warmup();
  engine.reset_stats();

  int n = a.get_int("tokens", 128);
  ChatOutputParser parser(spec, tok);
  StreamDecoder plain(tok);
  bool in_reasoning = false;
  auto on_token = [&](int t) {
    if (std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end()) return true;
    if (chat) {
      auto d = parser.push(t);
      if (!d.reasoning.empty()) {
        if (!in_reasoning) { fputs("\x1b[2m", stdout); in_reasoning = true; }
        fputs(d.reasoning.c_str(), stdout);
      }
      if (!d.content.empty()) {
        if (in_reasoning) { fputs("\x1b[0m\n", stdout); in_reasoning = false; }
        fputs(d.content.c_str(), stdout);
      }
    } else {
      fputs(plain.push(t).c_str(), stdout);
    }
    fflush(stdout);
    return true;
  };
  auto res = engine.generate(ids, n, sampling_options(a), stop_ids, on_token);
  if (chat) {
    auto d = parser.flush();
    fputs(d.reasoning.c_str(), stdout);
    fputs(d.content.c_str(), stdout);
  } else {
    fputs(plain.flush().c_str(), stdout);
  }
  if (in_reasoning) fputs("\x1b[0m", stdout);
  printf("\n");
  MMR_INFO("Prefill: %d tokens (%d cached) in %.2fs (%.1f tok/s); decode: %d tokens in %.2fs (%.2f tok/s); finish: %s",
           res.n_prompt, res.n_cached, res.prefill_s, res.prefill_s > 0 ? (res.n_prompt - res.n_cached) / res.prefill_s : 0.0,
           res.n_generated, res.decode_s, res.decode_s > 0 ? res.n_generated / res.decode_s : 0.0, res.finish_reason.c_str());
  if (a.has("stats")) print_stats(engine, res.n_generated, res.decode_s);
  return 0;
}

int cmd_serve(const Args& a) {
  if (a.positional.empty()) { usage(); return 2; }
  Engine engine(engine_options(a, a.positional[0]));
  std::vector<int> calib;
  if (!a.has("profile")) {
    // Calibrate on a generic chat prompt so the pinned set reflects real routing.
    std::vector<ChatMessage> msgs{{"user", "Explain briefly how a transformer language model generates text, then list three practical tips for writing clear code."}};
    ChatOptions copts;
    calib = engine.tokenizer().encode(format_chat(engine.spec(), msgs, copts), false);
  }
  setup_pins(engine, a, calib, a.get_int("calibrate", 16));
  engine.warmup();
  ServerOptions so;
  so.host = a.get("host", "127.0.0.1");
  so.port = a.get_int("port", 8080);
  so.model_name = a.get("model-name", "");
  so.default_max_tokens = a.get_int("max-tokens", 1024);
  so.chat.reasoning_effort = a.get("reasoning", "medium");
  so.chat.enable_thinking = !a.has("no-thinking");
  return run_server(engine, so);
}

int cmd_profile(const Args& a) {
  if (a.positional.empty() || !a.has("text")) { usage(); return 2; }
  Engine engine(engine_options(a, a.positional[0]));
  std::ifstream f(a.get("text"));
  if (!f) { MMR_ERROR("cannot open %s", a.get("text").c_str()); return 1; }
  std::stringstream ss;
  ss << f.rdbuf();
  auto ids = engine.tokenizer().encode(ss.str(), engine.spec().add_bos);
  int max_tokens = a.get_int("max-prompt", 2048);
  if ((int)ids.size() > max_tokens) ids.resize(max_tokens);
  MMR_INFO("Profiling routing over %zu tokens (nothing pinned)...", ids.size());
  engine.cache().reset_counts();
  engine.reset_kv();
  engine.prefill(ids);
  auto ranking = engine.cache().ranking_from_counts();
  const auto& counts = engine.cache().access_counts();
  json j;
  j["model"] = engine.spec().name;
  j["n_tokens"] = ids.size();
  j["ranking"] = ranking;
  json cj = json::array();
  for (int l = 0; l < engine.spec().n_layers; l++) {
    json row = json::array();
    for (int e = 0; e < engine.spec().n_experts; e++) row.push_back(counts[(size_t)l * engine.spec().n_experts + e]);
    cj.push_back(row);
  }
  j["counts"] = cj;
  std::string out = a.get("out", "profile.json");
  std::ofstream of(out);
  of << j.dump();
  MMR_INFO("Wrote %s", out.c_str());
  return 0;
}

int cmd_bench(const Args& a) {
  if (a.positional.empty()) { usage(); return 2; }
  Engine engine(engine_options(a, a.positional[0]));
  const auto& spec = engine.spec();
  const auto& tok = engine.tokenizer();
  std::vector<std::string> prompts = {
      "The key insight about mixture-of-experts models is that",
      "Write a short story about a lighthouse keeper who discovers",
      "In numerical linear algebra, the conjugate gradient method"};
  if (a.has("prompts")) {
    prompts.clear();
    std::ifstream f(a.get("prompts"));
    std::string line;
    while (std::getline(f, line))
      if (!line.empty()) prompts.push_back(line);
  }
  int n = a.get_int("tokens", 32);
  auto ids0 = tok.encode(prompts[0], spec.add_bos);
  setup_pins(engine, a, ids0, a.get_int("calibrate", 8));
  engine.warmup();
  SamplingParams sp = sampling_options(a);
  sp.temperature = (float)a.get_double("temp", 0.0);
  int total = 0;
  double total_s = 0, prefill_s = 0;
  int total_prompt = 0;
  engine.reset_stats();
  for (auto& p : prompts) {
    auto ids = tok.encode(p, spec.add_bos);
    engine.reset_kv();
    std::string out;
    StreamDecoder dec(tok);
    auto res = engine.generate(ids, n, sp, spec.stop_ids, [&](int t) { out += dec.push(t); return true; });
    total += res.n_generated;
    total_s += res.decode_s;
    prefill_s += res.prefill_s;
    total_prompt += res.n_prompt;
    MMR_INFO("[%d tok, %.2f tok/s] %s ...", res.n_generated, res.n_generated / res.decode_s, out.substr(0, 80).c_str());
  }
  fprintf(stderr, "\nbench: decode %.2f tok/s over %d tokens; prefill %.1f tok/s over %d prompt tokens\n",
          total / total_s, total, total_prompt / prefill_s, total_prompt);
  print_stats(engine, total, total_s);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  std::string cmd = argv[1];
  Args a = parse_args(argc, argv, 2);
  if (a.has("quiet")) set_log_level(LogLevel::Warn);
  if (a.has("verbose")) set_log_level(LogLevel::Debug);
  if (a.has("help") || cmd == "help" || cmd == "--help") {
    usage();
    return 0;
  }
  try {
    if (cmd == "convert") return cmd_convert(a);
    if (cmd == "run") return cmd_run(a);
    if (cmd == "serve") return cmd_serve(a);
    if (cmd == "profile") return cmd_profile(a);
    if (cmd == "bench") return cmd_bench(a);
    if (cmd == "info") return cmd_info(a);
    if (cmd == "--version" || cmd == "version") {
      printf("mlx-moe-runner %s\n", MMR_VERSION);
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& ex) {
    MMR_ERROR("%s", ex.what());
    return 1;
  }
}
