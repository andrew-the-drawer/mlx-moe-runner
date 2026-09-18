#pragma once
#include <cstdio>
#include <string>

namespace mmr {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3, Quiet = 4 };

void set_log_level(LogLevel level);
LogLevel log_level();

void log_msg(LogLevel level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

#define MMR_DEBUG(...) ::mmr::log_msg(::mmr::LogLevel::Debug, __VA_ARGS__)
#define MMR_INFO(...) ::mmr::log_msg(::mmr::LogLevel::Info, __VA_ARGS__)
#define MMR_WARN(...) ::mmr::log_msg(::mmr::LogLevel::Warn, __VA_ARGS__)
#define MMR_ERROR(...) ::mmr::log_msg(::mmr::LogLevel::Error, __VA_ARGS__)

std::string human_bytes(double bytes);

} // namespace mmr
