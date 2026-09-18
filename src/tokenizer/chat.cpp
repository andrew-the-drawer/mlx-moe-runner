#include "tokenizer/chat.h"

#include <ctime>

namespace mmr {

namespace {

std::string today() {
  time_t t = time(nullptr);
  struct tm tmv;
  localtime_r(&t, &tmv);
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%d", &tmv);
  return buf;
}

std::string format_harmony(const std::vector<ChatMessage>& messages, const ChatOptions& opts) {
  std::string identity = opts.model_identity.empty()
      ? "You are ChatGPT, a large language model trained by OpenAI."
      : opts.model_identity;
  std::string out;
  out += "<|start|>system<|message|>" + identity + "\nKnowledge cutoff: 2024-06\nCurrent date: " + today() +
         "\n\nReasoning: " + opts.reasoning_effort +
         "\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>";
  std::string developer;
  for (auto& m : messages)
    if (m.role == "system" || m.role == "developer") developer += (developer.empty() ? "" : "\n\n") + m.content;
  if (!developer.empty()) out += "<|start|>developer<|message|># Instructions\n\n" + developer + "<|end|>";
  for (auto& m : messages) {
    if (m.role == "user") out += "<|start|>user<|message|>" + m.content + "<|end|>";
    else if (m.role == "assistant") out += "<|start|>assistant<|channel|>final<|message|>" + m.content + "<|end|>";
  }
  out += "<|start|>assistant";
  return out;
}

std::string format_chatml(const std::vector<ChatMessage>& messages, const ChatOptions& opts) {
  std::string out;
  for (auto& m : messages) {
    std::string role = m.role == "developer" ? "system" : m.role;
    out += "<|im_start|>" + role + "\n" + m.content + "<|im_end|>\n";
  }
  out += "<|im_start|>assistant\n";
  if (!opts.enable_thinking) out += "<think>\n\n</think>\n\n";
  return out;
}

} // namespace

std::string format_chat(const ModelSpec& spec, const std::vector<ChatMessage>& messages, const ChatOptions& opts) {
  if (spec.chat_format == "harmony") return format_harmony(messages, opts);
  if (spec.chat_format == "chatml") return format_chatml(messages, opts);
  // Plain concatenation for unknown formats.
  std::string out;
  for (auto& m : messages) out += m.role + ": " + m.content + "\n";
  out += "assistant: ";
  return out;
}

std::vector<int> chat_stop_ids(const ModelSpec& spec, const BpeTokenizer& tok) {
  std::vector<int> ids = spec.stop_ids;
  auto add = [&](const char* s) {
    int id = tok.token_to_id(s);
    if (id >= 0 && std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
  };
  if (spec.chat_format == "harmony") {
    add("<|return|>");
    add("<|call|>");
  } else if (spec.chat_format == "chatml") {
    add("<|im_end|>");
    add("<|endoftext|>");
  }
  if (spec.eos_id >= 0 && std::find(ids.begin(), ids.end(), spec.eos_id) == ids.end()) ids.push_back(spec.eos_id);
  return ids;
}

ChatOutputParser::ChatOutputParser(const ModelSpec& spec, const BpeTokenizer& tok)
    : format_(spec.chat_format), tok_(tok), dec_(tok) {
  if (format_ == "harmony") {
    id_start_ = tok.token_to_id("<|start|>");
    id_channel_ = tok.token_to_id("<|channel|>");
    id_message_ = tok.token_to_id("<|message|>");
    id_end_ = tok.token_to_id("<|end|>");
    in_header_ = true; // the prompt ends with "<|start|>assistant": we are in the header
  } else if (format_ == "chatml") {
    id_think_ = tok.token_to_id("<think>");
    id_think_end_ = tok.token_to_id("</think>");
  }
}

ChatOutputParser::Delta ChatOutputParser::push(int id) {
  Delta d;
  if (format_ == "harmony") {
    if (id == id_start_) { in_header_ = true; header_.clear(); return d; }
    if (id == id_channel_) { in_header_ = true; header_.clear(); return d; }
    if (id == id_message_) {
      in_header_ = false;
      reasoning_ = header_.find("analysis") != std::string::npos || header_.find("commentary") != std::string::npos;
      header_.clear();
      return d;
    }
    if (id == id_end_) { std::string rest = dec_.flush(); (reasoning_ ? d.reasoning : d.content) += rest; return d; }
    if (tok_.is_special(id)) return d;
    if (in_header_) { header_ += tok_.token_bytes(id); return d; }
    std::string text = dec_.push(id);
    (reasoning_ ? d.reasoning : d.content) += text;
    return d;
  }
  if (format_ == "chatml") {
    if (id_think_ >= 0 && id == id_think_) { d.content = dec_.flush(); reasoning_ = true; return d; }
    if (id_think_end_ >= 0 && id == id_think_end_) { d.reasoning = dec_.flush(); reasoning_ = false; return d; }
    if (tok_.is_special(id)) return d;
    std::string text = dec_.push(id);
    if (id_think_ < 0) {
      // <think> is not a single token: detect it textually.
      think_buf_ += text;
      std::string emit;
      while (true) {
        if (!think_open_) {
          size_t p = think_buf_.find("<think>");
          if (p != std::string::npos) {
            d.content += think_buf_.substr(0, p);
            think_buf_.erase(0, p + 7);
            think_open_ = true;
            reasoning_ = true;
            continue;
          }
          // keep a possible partial tag
          size_t keep = 0;
          for (size_t k = std::min<size_t>(6, think_buf_.size()); k > 0; k--)
            if (think_buf_.compare(think_buf_.size() - k, k, std::string("<think>").substr(0, k)) == 0) { keep = k; break; }
          d.content += think_buf_.substr(0, think_buf_.size() - keep);
          think_buf_.erase(0, think_buf_.size() - keep);
          break;
        } else {
          size_t p = think_buf_.find("</think>");
          if (p != std::string::npos) {
            d.reasoning += think_buf_.substr(0, p);
            think_buf_.erase(0, p + 8);
            think_open_ = false;
            reasoning_ = false;
            continue;
          }
          size_t keep = 0;
          for (size_t k = std::min<size_t>(7, think_buf_.size()); k > 0; k--)
            if (think_buf_.compare(think_buf_.size() - k, k, std::string("</think>").substr(0, k)) == 0) { keep = k; break; }
          d.reasoning += think_buf_.substr(0, think_buf_.size() - keep);
          think_buf_.erase(0, think_buf_.size() - keep);
          break;
        }
      }
      return d;
    }
    (reasoning_ ? d.reasoning : d.content) += text;
    return d;
  }
  if (tok_.is_special(id)) return d;
  d.content = dec_.push(id);
  return d;
}

ChatOutputParser::Delta ChatOutputParser::flush() {
  Delta d;
  std::string rest = dec_.flush() + think_buf_;
  think_buf_.clear();
  (reasoning_ ? d.reasoning : d.content) += rest;
  return d;
}

} // namespace mmr
