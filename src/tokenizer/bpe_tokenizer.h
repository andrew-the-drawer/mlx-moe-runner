#pragma once
// Byte-level BPE tokenizer loaded from an HF tokenizer.json (GPT-2 / Qwen /
// o200k style): added tokens are matched literally, then each pre-tokenizer
// regex piece is byte-encoded and merged by rank.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mmr {

class BpeTokenizer {
 public:
  explicit BpeTokenizer(const std::string& tokenizer_json_path);
  ~BpeTokenizer();
  BpeTokenizer(const BpeTokenizer&) = delete;
  BpeTokenizer& operator=(const BpeTokenizer&) = delete;

  std::vector<int> encode(const std::string& text, bool add_bos = false) const;
  std::string decode(const std::vector<int>& ids, bool skip_special = false) const;
  // Bytes for one token (special tokens return their literal content).
  std::string token_bytes(int id) const;
  bool is_special(int id) const { return special_ids_.count(id) > 0; }
  bool is_added(int id) const { return added_ids_.count(id) > 0; }
  int token_to_id(const std::string& token) const; // -1 if absent
  int vocab_size() const { return (int)id_to_token_.size(); }
  int bos_id() const { return bos_id_; }
  int eos_id() const { return eos_id_; }

 private:
  struct Regex;
  void bpe_piece(const std::string& piece, std::vector<int>& out) const;

  std::vector<std::string> id_to_token_;         // byte-level encoded strings
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<uint64_t, int> merge_rank_; // (a << 32 | b) -> rank
  std::unordered_map<uint64_t, int> merge_result_;
  std::unordered_map<int, std::string> added_content_;
  std::unordered_set<int> added_ids_;
  std::unordered_set<int> special_ids_;
  std::string byte_to_unicode_[256];             // UTF-8 encoded char per byte
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;
  std::unique_ptr<Regex> pre_;
  std::unique_ptr<Regex> added_;
  int bos_id_ = -1;
  int eos_id_ = -1;
};

// Incremental UTF-8 safe detokenizer for streaming output.
class StreamDecoder {
 public:
  explicit StreamDecoder(const BpeTokenizer& tok) : tok_(tok) {}
  // Returns the printable text newly completed by this token.
  std::string push(int id);
  std::string flush();

 private:
  const BpeTokenizer& tok_;
  std::string pending_;
};

} // namespace mmr
