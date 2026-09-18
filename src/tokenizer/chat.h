#pragma once
// Chat prompt formatting and output parsing for the supported chat formats:
//   harmony (gpt-oss)  <|start|>role<|message|>...<|end|>, channels analysis/final
//   chatml  (qwen)     <|im_start|>role\n...<|im_end|>
#include <string>
#include <vector>

#include "model/spec.h"
#include "tokenizer/bpe_tokenizer.h"

namespace mmr {

struct ChatMessage {
  std::string role;     // system, user, assistant
  std::string content;
};

struct ChatOptions {
  std::string reasoning_effort = "medium"; // harmony: low | medium | high
  bool enable_thinking = true;             // chatml (qwen3): allow <think> blocks
  std::string model_identity;              // harmony system identity (default text if empty)
};

// Renders messages into the model's prompt text (ends with the assistant header).
std::string format_chat(const ModelSpec& spec, const std::vector<ChatMessage>& messages, const ChatOptions& opts);

// Stop token ids for generation in this chat format.
std::vector<int> chat_stop_ids(const ModelSpec& spec, const BpeTokenizer& tok);

// Splits generated tokens into reasoning and answer text as they stream in.
class ChatOutputParser {
 public:
  ChatOutputParser(const ModelSpec& spec, const BpeTokenizer& tok);
  struct Delta {
    std::string content;
    std::string reasoning;
  };
  Delta push(int id);
  Delta flush();
  bool in_reasoning() const { return reasoning_; }

 private:
  std::string format_;
  const BpeTokenizer& tok_;
  StreamDecoder dec_;
  bool reasoning_ = false;
  bool in_header_ = false;
  std::string header_;
  int id_start_ = -1, id_channel_ = -1, id_message_ = -1, id_end_ = -1;
  int id_think_ = -1, id_think_end_ = -1;
  std::string think_buf_;
  bool think_open_ = false;
};

} // namespace mmr
