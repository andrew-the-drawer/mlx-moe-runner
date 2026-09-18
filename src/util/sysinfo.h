#pragma once
#include <cstddef>
#include <cstdint>

namespace mmr {

// Physical RAM in bytes.
uint64_t physical_memory();
// Number of performance (non-efficiency) cores, or half the logical cores as a fallback.
int performance_cores();
// Current resident set size of this process in bytes.
uint64_t resident_memory();
// Free + inactive memory as reported by the VM subsystem (approximate).
uint64_t available_memory();

} // namespace mmr
