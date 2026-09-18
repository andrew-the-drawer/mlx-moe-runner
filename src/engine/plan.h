#pragma once
// Configuration planner: probes the machine and predicts tok/s per config
// with the cost model validated by the benchmarks.
#include <string>

namespace mmr {

struct PlanOptions {
  std::string model_dir;
  double ram_gb = 0;      // 0 = physical RAM
  int ctx = 4096;
  bool exact_only = false;
  bool probe = true;      // measure SSD latency/bandwidth (else assume 1.2 ms / 11 GB/s)
  int top = 6;
};

int run_plan(const PlanOptions& opts);

} // namespace mmr
