#include <cstdio>

#include <nlohmann/json.hpp>

#include "model/layout.h"
#include "model/spec.h"

using namespace mmr;

int main() {
  int failures = 0;
  ExpertLayout L;
  L.n_layers = 2;
  L.n_experts = 4;
  L.record_bytes = 32768;
  L.parts.push_back({"gate.weight", {8, 4}, "uint32", 0, 128, 4096});
  L.parts.push_back({"gate.scales", {8, 1}, "float16", 4096, 16, 4096});
  QuantDesc q;
  q.mode = "affine";
  q.bits = 4;
  q.group = 32;
  q.has_biases = true;
  L.quant["gate"] = q;
  L.files[0] = "experts_000.bin";
  auto j = L.to_json();
  auto L2 = ExpertLayout::from_json(j);
  if (L2.parts.size() != 2 || L2.part("gate.scales")->offset != 4096) { printf("FAIL: layout round trip\n"); failures++; }
  if (L2.quant["gate"].bits != 4 || !L2.quant["gate"].has_biases) { printf("FAIL: quant round trip\n"); failures++; }
  if (L2.file_for(1) != "experts_001.bin") { printf("FAIL: default file name\n"); failures++; }
  if (L2.expert_offset(3) != 3 * 32768) { printf("FAIL: expert offset\n"); failures++; }

  ModelSpec s;
  s.arch = "gpt-oss";
  s.head_dim = 64;
  s.rope_base = 150000.f;
  s.rope_yarn_factor = 32.f;
  s.rope_orig_ctx = 4096;
  s.sliding_window = 128;
  s.swa_every = 2;
  auto sj = s.to_json();
  auto s2 = ModelSpec::from_json(sj);
  if (!s2.uses_yarn() || s2.rope_freqs().size() != 32) { printf("FAIL: spec round trip\n"); failures++; }
  if (fabsf(s2.rope_mscale() - (0.1f * logf(32.f) + 1.f)) > 1e-6f) { printf("FAIL: mscale\n"); failures++; }
  if (!s2.is_swa_layer(0) || s2.is_swa_layer(1)) { printf("FAIL: swa layers\n"); failures++; }
  // Without YaRN, freqs are plain periods base^(2i/d).
  ModelSpec p;
  p.head_dim = 8;
  p.rope_base = 10000.f;
  auto f = p.rope_freqs();
  if (fabsf(f[0] - 1.f) > 1e-6f || fabsf(f[1] - 10.f) > 1e-3f) { printf("FAIL: rope freqs %g %g\n", f[0], f[1]); failures++; }
  printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
