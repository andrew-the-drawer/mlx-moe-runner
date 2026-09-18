#include "util/sysinfo.h"

#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <string>
#include <thread>

namespace mmr {

uint64_t physical_memory() {
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) return 0;
  return mem;
}

int performance_cores() {
  int best = 0;
  for (int level = 0; level < 4; level++) {
    char name[64] = {0};
    size_t nlen = sizeof(name);
    std::string key = "hw.perflevel" + std::to_string(level) + ".name";
    if (sysctlbyname(key.c_str(), name, &nlen, nullptr, 0) != 0) break;
    int ncpu = 0;
    size_t clen = sizeof(ncpu);
    std::string ckey = "hw.perflevel" + std::to_string(level) + ".physicalcpu";
    if (sysctlbyname(ckey.c_str(), &ncpu, &clen, nullptr, 0) != 0) break;
    std::string n(name);
    for (auto& c : n) c = (char)tolower(c);
    if (n.find("efficiency") == std::string::npos && ncpu > best) best = ncpu;
  }
  if (best > 0) return best;
  int hw = (int)std::thread::hardware_concurrency();
  return hw > 1 ? hw / 2 : 1;
}

uint64_t resident_memory() {
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
    return 0;
  return info.resident_size;
}

uint64_t available_memory() {
  vm_size_t page = 0;
  host_page_size(mach_host_self(), &page);
  vm_statistics64_data_t vm;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS)
    return 0;
  return (uint64_t)(vm.free_count + vm.inactive_count) * page;
}

} // namespace mmr
