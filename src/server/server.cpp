#include "server/server.h"

#include <atomic>
#include <chrono>
#include <mutex>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "tokenizer/chat.h"
#include "util/log.h"

using json = nlohmann::json;

namespace mmr {

namespace {

std::string new_id(const char* prefix) {
  static std::atomic<uint64_t> counter{0};
  auto t = std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
  return std::string(prefix) + "-" + std::to_string(t) + "-" + std::to_string(counter++);
}

int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

SamplingParams sampling_from(const json& body) {
  SamplingParams p;
  p.temperature = body.value("temperature", 0.7f);
  p.top_p = body.value("top_p", 0.9f);
  p.top_k = body.value("top_k", 0);
  p.min_p = body.value("min_p", 0.0f);
  if (body.contains("seed") && body["seed"].is_number()) p.seed = body["seed"].get<uint64_t>();
  else p.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
  return p;
}

struct Request {
  std::vector<int> prompt_ids;
  int max_tokens;
  SamplingParams sampling;
  std::vector<int> stop_ids;
  std::vector<std::string> stop_strings;
  bool stream;
  bool chat;
};

} // namespace

int run_server(Engine& engine, const ServerOptions& opts) {
  httplib::Server svr;
  std::mutex gen_mutex;
  const auto& spec = engine.spec();
  const auto& tok = engine.tokenizer();
  std::string model_name = opts.model_name.empty() ? spec.name : opts.model_name;
  std::vector<int> chat_stops = chat_stop_ids(spec, tok);

  svr.set_read_timeout(600, 0);
  svr.set_write_timeout(600, 0);

  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "ok"}, {"model", model_name}}.dump(), "application/json");
  });
  svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    json j = {{"object", "list"},
              {"data", json::array({json{{"id", model_name}, {"object", "model"}, {"owned_by", "mlx-moe-runner"},
                                         {"created", unix_now()}}})}};
    res.set_content(j.dump(), "application/json");
  });
  svr.Get("/v1/stats", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(engine.stats_json().dump(2), "application/json");
  });

  auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (const std::exception& ex) {
      res.status = 400;
      res.set_content(json{{"error", {{"message", std::string("invalid JSON: ") + ex.what()}}}}.dump(), "application/json");
      return;
    }
    Request r;
    r.chat = chat;
    r.stream = body.value("stream", false);
    r.max_tokens = body.value("max_tokens", body.value("max_completion_tokens", opts.default_max_tokens));
    r.sampling = sampling_from(body);
    std::string prompt_text;
    ChatOptions copts = opts.chat;
    if (body.contains("reasoning_effort")) copts.reasoning_effort = body["reasoning_effort"].get<std::string>();
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].contains("enable_thinking"))
      copts.enable_thinking = body["chat_template_kwargs"]["enable_thinking"].get<bool>();
    if (chat) {
      std::vector<ChatMessage> msgs;
      for (auto& m : body.value("messages", json::array())) {
        ChatMessage cm;
        cm.role = m.value("role", "user");
        if (m.contains("content") && m["content"].is_string()) cm.content = m["content"].get<std::string>();
        else if (m.contains("content") && m["content"].is_array())
          for (auto& part : m["content"])
            if (part.value("type", "") == "text") cm.content += part.value("text", "");
        msgs.push_back(cm);
      }
      prompt_text = format_chat(spec, msgs, copts);
      r.stop_ids = chat_stops;
    } else {
      if (body.contains("prompt") && body["prompt"].is_string()) prompt_text = body["prompt"].get<std::string>();
      else if (body.contains("prompt") && body["prompt"].is_array() && !body["prompt"].empty())
        prompt_text = body["prompt"][0].get<std::string>();
      r.stop_ids = spec.stop_ids;
    }
    if (body.contains("stop")) {
      if (body["stop"].is_string()) r.stop_strings.push_back(body["stop"].get<std::string>());
      else if (body["stop"].is_array())
        for (auto& s : body["stop"]) r.stop_strings.push_back(s.get<std::string>());
    }
    r.prompt_ids = tok.encode(prompt_text, !chat && spec.add_bos);
    if (r.prompt_ids.empty()) {
      res.status = 400;
      res.set_content(json{{"error", {{"message", "empty prompt"}}}}.dump(), "application/json");
      return;
    }
    std::string id = new_id(chat ? "chatcmpl" : "cmpl");
    int64_t created = unix_now();
    const char* object = chat ? "chat.completion" : "text_completion";

    auto run = [&, r, id, created, object](const std::function<void(const std::string&)>& emit) {
      std::lock_guard<std::mutex> lock(gen_mutex);
      ChatOutputParser parser(spec, tok);
      StreamDecoder plain(tok);
      std::string content, reasoning;
      GenerateResult gr;
      bool stopped_by_string = false;
      auto on_token = [&](int t) {
        std::string dc, dr;
        if (std::find(r.stop_ids.begin(), r.stop_ids.end(), t) != r.stop_ids.end()) {
          auto d = r.chat ? parser.flush() : ChatOutputParser::Delta{plain.flush(), ""};
          dc = d.content;
          dr = d.reasoning;
        } else if (r.chat) {
          auto d = parser.push(t);
          dc = d.content;
          dr = d.reasoning;
        } else {
          dc = plain.push(t);
        }
        if (!r.stop_strings.empty() && !dc.empty()) {
          // Truncate at the first stop string that appears in the answer so far.
          std::string all = content + dc;
          size_t cut = std::string::npos;
          for (auto& s : r.stop_strings) {
            size_t p = all.find(s, content.size() >= s.size() ? content.size() - s.size() + 1 : 0);
            if (p != std::string::npos) cut = std::min(cut, p);
          }
          if (cut != std::string::npos) {
            dc = cut > content.size() ? all.substr(content.size(), cut - content.size()) : "";
            stopped_by_string = true;
          }
        }
        if (!dc.empty() || !dr.empty()) {
          content += dc;
          reasoning += dr;
          if (r.stream) {
            json delta = r.chat ? json{{"role", "assistant"}} : json::object();
            if (r.chat) {
              if (!dc.empty()) delta["content"] = dc;
              if (!dr.empty()) delta["reasoning_content"] = dr;
            }
            json choice = r.chat ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                                 : json{{"index", 0}, {"text", dc}, {"finish_reason", nullptr}};
            json ev = {{"id", id}, {"object", r.chat ? "chat.completion.chunk" : "text_completion"},
                       {"created", created}, {"model", model_name}, {"choices", json::array({choice})}};
            emit("data: " + ev.dump() + "\n\n");
          }
        }
        return !stopped_by_string;
      };
      gr = engine.generate(r.prompt_ids, r.max_tokens, r.sampling, r.stop_ids, on_token);
      if (stopped_by_string) gr.finish_reason = "stop";
      if (gr.finish_reason == "abort") gr.finish_reason = "stop";
      json usage = {{"prompt_tokens", gr.n_prompt}, {"completion_tokens", gr.n_generated},
                    {"total_tokens", gr.n_prompt + gr.n_generated},
                    {"prompt_tokens_cached", gr.n_cached}, {"prompt_cache_source", gr.cache_source},
                    {"prefill_s", gr.prefill_s}, {"decode_s", gr.decode_s},
                    {"tokens_per_second", gr.decode_s > 0 ? gr.n_generated / gr.decode_s : 0.0}};
      if (r.stream) {
        json choice = r.chat ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", gr.finish_reason}}
                             : json{{"index", 0}, {"text", ""}, {"finish_reason", gr.finish_reason}};
        json ev = {{"id", id}, {"object", r.chat ? "chat.completion.chunk" : "text_completion"},
                   {"created", created}, {"model", model_name}, {"choices", json::array({choice})}, {"usage", usage}};
        emit("data: " + ev.dump() + "\n\n");
        emit("data: [DONE]\n\n");
      } else {
        json message = {{"role", "assistant"}, {"content", content}};
        if (!reasoning.empty()) message["reasoning_content"] = reasoning;
        json choice = r.chat ? json{{"index", 0}, {"message", message}, {"finish_reason", gr.finish_reason}}
                             : json{{"index", 0}, {"text", content}, {"finish_reason", gr.finish_reason}};
        json out = {{"id", id}, {"object", object}, {"created", created}, {"model", model_name},
                    {"choices", json::array({choice})}, {"usage", usage}};
        emit(out.dump());
      }
      MMR_INFO("%s: %d prompt (%d cached, %s), %d generated, prefill %.2fs, decode %.2fs (%.1f tok/s), %s",
               id.c_str(), gr.n_prompt, gr.n_cached, gr.cache_source.c_str(), gr.n_generated, gr.prefill_s, gr.decode_s,
               gr.decode_s > 0 ? gr.n_generated / gr.decode_s : 0.0, gr.finish_reason.c_str());
    };

    if (r.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("X-Accel-Buffering", "no");
      auto runner = std::make_shared<std::function<void(const std::function<void(const std::string&)>&)>>(run);
      res.set_chunked_content_provider("text/event-stream", [runner](size_t, httplib::DataSink& sink) {
        try {
          (*runner)([&](const std::string& s) { sink.write(s.data(), s.size()); });
        } catch (const std::exception& ex) {
          std::string err = "data: " + json{{"error", {{"message", ex.what()}}}}.dump() + "\n\n";
          sink.write(err.data(), err.size());
        }
        sink.done();
        return true;
      });
    } else {
      try {
        std::string out;
        run([&](const std::string& s) { out += s; });
        res.set_content(out, "application/json");
      } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
      }
    }
  };

  svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, true); });
  svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, false); });

  MMR_INFO("Serving %s on http://%s:%d (OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models)",
           model_name.c_str(), opts.host.c_str(), opts.port);
  if (!svr.listen(opts.host, opts.port)) {
    MMR_ERROR("failed to listen on %s:%d", opts.host.c_str(), opts.port);
    return 1;
  }
  return 0;
}

} // namespace mmr
