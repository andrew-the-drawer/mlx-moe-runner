#pragma once
// OpenAI-compatible HTTP server (chat/completions, completions, models) with
// SSE streaming. One request generates at a time; others queue.
#include <string>

#include "engine/engine.h"
#include "tokenizer/chat.h"

namespace mmr {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string model_name;       // reported model id
  int default_max_tokens = 1024;
  ChatOptions chat;
};

int run_server(Engine& engine, const ServerOptions& opts);

} // namespace mmr
