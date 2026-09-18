// Builds a small byte-level BPE tokenizer.json and checks round trips.
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "tokenizer/bpe_tokenizer.h"

using namespace mmr;
using json = nlohmann::json;
namespace fs = std::filesystem;

int main() {
  int failures = 0;
  // Vocab: all 256 byte symbols (GPT-2 byte-level alphabet) + a few merges.
  json vocab = json::object();
  int id = 0;
  // Byte-level alphabet in GPT-2 order.
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; b++) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
  std::vector<uint32_t> cs(bs.begin(), bs.end());
  int n = 0;
  for (int b = 0; b < 256; b++)
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
  auto enc = [](uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
  };
  for (size_t i = 0; i < cs.size(); i++) vocab[enc(cs[i])] = id++;
  std::vector<std::string> merges = {"h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d"};
  for (auto& m : merges) {
    std::string a = m.substr(0, m.find(' ')), b = m.substr(m.find(' ') + 1);
    vocab[a + b] = id++;
  }
  int special_id = id++;
  json j;
  j["model"] = {{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}};
  j["added_tokens"] = json::array({json{{"id", special_id}, {"content", "<|im_end|>"}, {"special", true}}});
  j["pre_tokenizer"] = {{"type", "ByteLevel"}, {"use_regex", true}};
  fs::path p = fs::temp_directory_path() / ("mmr_tok_" + std::to_string(getpid()) + ".json");
  {
    std::ofstream f(p);
    f << j.dump();
  }
  BpeTokenizer tok(p.string());
  auto ids = tok.encode("hello world<|im_end|>");
  // "hello" -> one token, " world" -> one token, special -> one token
  if (ids.size() != 3) { printf("FAIL: expected 3 tokens, got %zu\n", ids.size()); failures++; }
  if (ids.size() == 3 && ids[2] != special_id) { printf("FAIL: special token id\n"); failures++; }
  if (tok.decode(ids) != "hello world<|im_end|>") { printf("FAIL: decode round trip: '%s'\n", tok.decode(ids).c_str()); failures++; }
  if (tok.decode(ids, true) != "hello world") { printf("FAIL: skip special\n"); failures++; }
  // Unknown merges fall back to byte symbols; UTF-8 survives.
  std::string u8 = "héllo 日本";
  auto ids2 = tok.encode(u8);
  if (tok.decode(ids2) != u8) { printf("FAIL: utf8 round trip\n"); failures++; }
  // Streaming decoder never emits partial UTF-8.
  StreamDecoder sd(tok);
  std::string out;
  for (int t : ids2) out += sd.push(t);
  out += sd.flush();
  if (out != u8) { printf("FAIL: stream decode\n"); failures++; }
  if (!tok.is_special(special_id) || tok.token_to_id("<|im_end|>") != special_id) { printf("FAIL: special lookup\n"); failures++; }
  fs::remove(p);
  printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
