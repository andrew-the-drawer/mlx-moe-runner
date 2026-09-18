#include "util/log.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace mmr {

static LogLevel g_level = LogLevel::Info;
static std::mutex g_mutex;

void set_log_level(LogLevel level) { g_level = level; }
LogLevel log_level() { return g_level; }

void log_msg(LogLevel level, const char* fmt, ...) {
  if (level < g_level) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  const char* prefix = "";
  switch (level) {
    case LogLevel::Debug: prefix = "[debug] "; break;
    case LogLevel::Warn: prefix = "[warn] "; break;
    case LogLevel::Error: prefix = "[error] "; break;
    default: break;
  }
  fputs(prefix, stderr);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  fflush(stderr);
}

std::string human_bytes(double bytes) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  int u = 0;
  while (bytes >= 1024.0 && u < 4) {
    bytes /= 1024.0;
    u++;
  }
  char buf[64];
  snprintf(buf, sizeof(buf), u >= 3 ? "%.2f %s" : "%.0f %s", bytes, units[u]);
  return buf;
}

} // namespace mmr
