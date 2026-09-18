#include "model/layout.h"

#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mmr {

json QuantDesc::to_json() const {
  return json{{"mode", mode}, {"bits", bits}, {"group", group}, {"has_biases", has_biases}, {"dtype", dtype}};
}

QuantDesc QuantDesc::from_json(const json& j) {
  QuantDesc q;
  q.mode = j.value("mode", "none");
  q.bits = j.value("bits", 0);
  q.group = j.value("group", 0);
  q.has_biases = j.value("has_biases", false);
  q.dtype = j.value("dtype", "float16");
  return q;
}

const PartDesc* ExpertLayout::part(const std::string& name) const {
  for (auto& p : parts)
    if (p.name == name) return &p;
  return nullptr;
}

std::string ExpertLayout::file_for(int layer) const {
  auto it = files.find(layer);
  if (it != files.end()) return it->second;
  char buf[64];
  snprintf(buf, sizeof(buf), "experts_%03d.bin", layer);
  return buf;
}

uint64_t ExpertLayout::payload_bytes() const {
  uint64_t n = 0;
  for (auto& p : parts) n += p.bytes;
  return n;
}

json ExpertLayout::to_json() const {
  json j;
  j["n_layers"] = n_layers;
  j["n_experts"] = n_experts;
  j["align"] = align;
  j["record_bytes"] = record_bytes;
  j["parts"] = json::array();
  for (auto& p : parts)
    j["parts"].push_back(json{{"name", p.name}, {"shape", p.shape}, {"dtype", p.dtype},
                              {"offset", p.offset}, {"bytes", p.bytes}, {"padded", p.padded}});
  j["quant"] = json::object();
  for (auto& [k, q] : quant) j["quant"][k] = q.to_json();
  j["files"] = json::object();
  for (auto& [l, f] : files) j["files"][std::to_string(l)] = f;
  return j;
}

ExpertLayout ExpertLayout::from_json(const json& j) {
  ExpertLayout l;
  l.n_layers = j.value("n_layers", 0);
  l.n_experts = j.value("n_experts", 0);
  l.align = j.value("align", 4096);
  l.record_bytes = j.value("record_bytes", 0);
  for (auto& pj : j.value("parts", json::array())) {
    PartDesc p;
    p.name = pj.value("name", "");
    p.shape = pj.value("shape", std::vector<int>{});
    p.dtype = pj.value("dtype", "uint32");
    p.offset = pj.value("offset", 0);
    p.bytes = pj.value("bytes", 0);
    p.padded = pj.value("padded", 0);
    l.parts.push_back(p);
  }
  if (j.contains("quant"))
    for (auto& [k, v] : j["quant"].items()) l.quant[k] = QuantDesc::from_json(v);
  if (j.contains("files"))
    for (auto& [k, v] : j["files"].items()) l.files[std::stoi(k)] = v.get<std::string>();
  return l;
}

size_t dtype_size(const std::string& dtype) {
  if (dtype == "uint32" || dtype == "float32" || dtype == "int32") return 4;
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "uint8" || dtype == "int8") return 1;
  throw std::runtime_error("unknown dtype " + dtype);
}

} // namespace mmr
