#include "tokenizer/bpe_tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "util/log.h"

using json = nlohmann::json;

namespace mmr {

namespace {

std::string utf8_encode(uint32_t cp) {
  std::string s;
  if (cp < 0x80) s += (char)cp;
  else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
  else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
  else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
  return s;
}

// Decodes one UTF-8 code point starting at s[i]; advances i.
uint32_t utf8_decode(const std::string& s, size_t& i) {
  unsigned char c = (unsigned char)s[i];
  uint32_t cp;
  int extra;
  if (c < 0x80) { cp = c; extra = 0; }
  else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
  else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
  else { cp = c & 0x07; extra = 3; }
  i++;
  for (int k = 0; k < extra && i < s.size(); k++, i++) cp = (cp << 6) | ((unsigned char)s[i] & 0x3F);
  return cp;
}

std::string regex_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (strchr("\\^$.|?*+()[]{}", c)) out += '\\';
    out += c;
  }
  return out;
}

} // namespace

struct BpeTokenizer::Regex {
  pcre2_code* code = nullptr;
  pcre2_match_data* md = nullptr;
  explicit Regex(const std::string& pattern) {
    int err = 0;
    PCRE2_SIZE off = 0;
    code = pcre2_compile((PCRE2_SPTR)pattern.c_str(), pattern.size(), PCRE2_UTF | PCRE2_UCP, &err, &off, nullptr);
    if (!code) {
      PCRE2_UCHAR buf[256];
      pcre2_get_error_message(err, buf, sizeof(buf));
      throw std::runtime_error("tokenizer regex error at " + std::to_string(off) + ": " + (const char*)buf);
    }
    pcre2_jit_compile(code, PCRE2_JIT_COMPLETE);
    md = pcre2_match_data_create_from_pattern(code, nullptr);
  }
  ~Regex() {
    if (md) pcre2_match_data_free(md);
    if (code) pcre2_code_free(code);
  }
  // Finds the next match at or after `start`; returns false if none.
  bool next(const std::string& s, size_t start, size_t* mstart, size_t* mend) const {
    if (start >= s.size()) return false;
    int rc = pcre2_match(code, (PCRE2_SPTR)s.data(), s.size(), start, PCRE2_NO_UTF_CHECK, md, nullptr);
    if (rc < 0) return false;
    PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
    *mstart = ov[0];
    *mend = ov[1];
    return true;
  }
};

BpeTokenizer::BpeTokenizer(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open tokenizer " + path);
  json j = json::parse(f);

  // GPT-2 byte <-> unicode table.
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; b++) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
  std::vector<uint32_t> cs(bs.begin(), bs.end());
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n++);
    }
  }
  for (size_t i = 0; i < bs.size(); i++) {
    byte_to_unicode_[bs[i]] = utf8_encode(cs[i]);
    unicode_to_byte_[cs[i]] = (uint8_t)bs[i];
  }

  const auto& model = j.at("model");
  if (model.value("type", "BPE") != "BPE") throw std::runtime_error("tokenizer: only BPE models are supported");
  const auto& vocab = model.at("vocab");
  int max_id = -1;
  for (auto& [tok, id] : vocab.items()) max_id = std::max(max_id, id.get<int>());
  id_to_token_.assign(max_id + 1, "");
  for (auto& [tok, id] : vocab.items()) {
    id_to_token_[id.get<int>()] = tok;
    token_to_id_[tok] = id.get<int>();
  }
  for (auto& t : j.value("added_tokens", json::array())) {
    int id = t.at("id").get<int>();
    std::string content = t.at("content").get<std::string>();
    if (id >= (int)id_to_token_.size()) id_to_token_.resize(id + 1);
    if (id_to_token_[id].empty()) id_to_token_[id] = content;
    token_to_id_.emplace(content, id);
    added_content_[id] = content;
    added_ids_.insert(id);
    if (t.value("special", false)) special_ids_.insert(id);
  }

  const auto& merges = model.at("merges");
  int rank = 0;
  for (auto& m : merges) {
    std::string a, b;
    if (m.is_string()) {
      std::string s = m.get<std::string>();
      size_t sp = s.find(' ');
      if (sp == std::string::npos) continue;
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    } else if (m.is_array() && m.size() == 2) {
      a = m[0].get<std::string>();
      b = m[1].get<std::string>();
    } else {
      continue;
    }
    auto ia = token_to_id_.find(a), ib = token_to_id_.find(b), ic = token_to_id_.find(a + b);
    if (ia == token_to_id_.end() || ib == token_to_id_.end() || ic == token_to_id_.end()) {
      rank++;
      continue;
    }
    uint64_t key = ((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second;
    if (!merge_rank_.count(key)) {
      merge_rank_[key] = rank;
      merge_result_[key] = ic->second;
    }
    rank++;
  }

  // Pre-tokenizer regex: find a Split/Regex entry; fall back to the GPT-2 pattern.
  std::string pattern;
  std::function<void(const json&)> find_regex = [&](const json& pt) {
    if (!pt.is_object()) return;
    std::string type = pt.value("type", "");
    if (type == "Sequence") {
      for (auto& sub : pt.value("pretokenizers", json::array())) find_regex(sub);
    } else if (type == "Split" && pattern.empty()) {
      if (pt.contains("pattern") && pt["pattern"].contains("Regex")) pattern = pt["pattern"]["Regex"].get<std::string>();
    } else if (type == "ByteLevel" && pattern.empty() && pt.value("use_regex", true)) {
      pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    }
  };
  if (j.contains("pre_tokenizer")) find_regex(j["pre_tokenizer"]);
  if (pattern.empty()) {
    MMR_WARN("tokenizer.json has no pre-tokenizer regex; using the GPT-2 pattern");
    pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  }
  pre_ = std::make_unique<Regex>(pattern);

  if (!added_content_.empty()) {
    std::vector<std::string> contents;
    for (auto& [id, c] : added_content_) contents.push_back(c);
    std::sort(contents.begin(), contents.end(), [](const std::string& a, const std::string& b) {
      return a.size() > b.size() || (a.size() == b.size() && a < b);
    });
    std::string alt;
    for (auto& c : contents) {
      if (!alt.empty()) alt += '|';
      alt += regex_escape(c);
    }
    added_ = std::make_unique<Regex>(alt);
  }

  if (j.contains("mmr")) {
    bos_id_ = j["mmr"].value("bos_id", -1);
    eos_id_ = j["mmr"].value("eos_id", -1);
  }
}

BpeTokenizer::~BpeTokenizer() = default;

int BpeTokenizer::token_to_id(const std::string& token) const {
  auto it = token_to_id_.find(token);
  return it == token_to_id_.end() ? -1 : it->second;
}

void BpeTokenizer::bpe_piece(const std::string& piece, std::vector<int>& out) const {
  // Byte-level encode, then look up each symbol.
  std::vector<int> syms;
  syms.reserve(piece.size());
  for (unsigned char c : piece) {
    auto it = token_to_id_.find(byte_to_unicode_[c]);
    if (it == token_to_id_.end()) throw std::runtime_error("tokenizer: byte symbol missing from vocab");
    syms.push_back(it->second);
  }
  while (syms.size() > 1) {
    int best_rank = INT32_MAX, best_pos = -1;
    for (size_t i = 0; i + 1 < syms.size(); i++) {
      uint64_t key = ((uint64_t)(uint32_t)syms[i] << 32) | (uint32_t)syms[i + 1];
      auto it = merge_rank_.find(key);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pos = (int)i;
      }
    }
    if (best_pos < 0) break;
    uint64_t key = ((uint64_t)(uint32_t)syms[best_pos] << 32) | (uint32_t)syms[best_pos + 1];
    syms[best_pos] = merge_result_.at(key);
    syms.erase(syms.begin() + best_pos + 1);
  }
  out.insert(out.end(), syms.begin(), syms.end());
}

std::vector<int> BpeTokenizer::encode(const std::string& text, bool add_bos) const {
  std::vector<int> out;
  if (add_bos && bos_id_ >= 0) out.push_back(bos_id_);
  auto encode_plain = [&](const std::string& s) {
    size_t pos = 0, ms, me;
    while (pre_->next(s, pos, &ms, &me)) {
      if (me == ms) { pos = ms + 1; continue; }
      bpe_piece(s.substr(ms, me - ms), out);
      pos = me;
    }
  };
  if (!added_) {
    encode_plain(text);
    return out;
  }
  size_t pos = 0, ms, me;
  while (added_->next(text, pos, &ms, &me)) {
    if (ms > pos) encode_plain(text.substr(pos, ms - pos));
    out.push_back(token_to_id_.at(text.substr(ms, me - ms)));
    pos = me;
  }
  if (pos < text.size()) encode_plain(text.substr(pos));
  return out;
}

std::string BpeTokenizer::token_bytes(int id) const {
  if (id < 0 || id >= (int)id_to_token_.size()) return "";
  auto a = added_content_.find(id);
  if (a != added_content_.end()) return a->second;
  const std::string& t = id_to_token_[id];
  std::string out;
  size_t i = 0;
  while (i < t.size()) {
    uint32_t cp = utf8_decode(t, i);
    auto it = unicode_to_byte_.find(cp);
    if (it != unicode_to_byte_.end()) out += (char)it->second;
    else out += utf8_encode(cp);
  }
  return out;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string out;
  for (int id : ids) {
    if (skip_special && special_ids_.count(id)) continue;
    out += token_bytes(id);
  }
  return out;
}

std::string StreamDecoder::push(int id) {
  pending_ += tok_.token_bytes(id);
  // Emit up to the last complete UTF-8 sequence.
  size_t n = pending_.size();
  size_t cut = n;
  // Walk back over at most 3 continuation bytes to find an incomplete lead byte.
  size_t i = n;
  int cont = 0;
  while (i > 0 && cont < 4) {
    unsigned char c = (unsigned char)pending_[i - 1];
    if ((c & 0xC0) == 0x80) { cont++; i--; continue; }
    int need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
    if (cont + 1 < need) cut = i - 1;
    break;
  }
  std::string out = pending_.substr(0, cut);
  pending_.erase(0, cut);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out = pending_;
  pending_.clear();
  return out;
}

} // namespace mmr
