#pragma once
// Cold-expert loader: a priority queue of expert reads served by a pool of I/O
// threads. Each expert record is read with F_NOCACHE preadv straight into MLX
// buffers (no copies, no page cache), split into one sub-read per matrix so
// several threads work on one expert and its latency drops, into a bounded
// LRU buffer.
//
// Semantics (from the tinygiant study):
//   * demand reads (prio 0) always run; prefetches are ordered by predicted
//     distance and dropped when they belong to an earlier forward pass.
//   * entries used by the layer currently computing cannot be evicted.
//   * a request for a resident entry refreshes its LRU position.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include "model/layout.h"

namespace mmr {

namespace mx = mlx::core;

constexpr int PRIO_DEMAND = 0;
constexpr int PRIO_DRAFT = 5;
constexpr int PRIO_LOOKAHEAD = 10; // + distance in layers
constexpr int PRIO_TOKEN_PRIOR = 30;

using ExpertKey = std::pair<int, int>; // (layer, expert)

// One expert's parts as MLX arrays, ordered like ExpertLayout::parts.
struct ExpertViews {
  std::vector<mx::array> parts;
};

struct LoaderStats {
  uint64_t bytes_read = 0;
  uint64_t n_reads = 0;
  double demand_wait_s = 0;
  uint64_t n_demand = 0;
  uint64_t n_prefetch_hit = 0;
  uint64_t n_prefetch_wasted = 0;
  uint64_t n_requests = 0;
  uint64_t n_evict = 0;
  uint64_t n_dropped = 0;
  uint64_t n_overflow = 0;
  double read_time_s = 0; // sum of per-record latencies (first sub-read start to completion)
  uint64_t resident_bytes = 0;
  uint64_t n_resident = 0;
  uint64_t queued = 0;
};

class ColdLoader {
 public:
  // split: sub-reads per record (1 = whole record in one preadv; 0 = one per matrix).
  ColdLoader(const std::string& dir, const ExpertLayout& layout, uint64_t budget_bytes, int n_threads,
             bool nocache = true, int split = 0);
  ~ColdLoader();
  ColdLoader(const ColdLoader&) = delete;
  ColdLoader& operator=(const ColdLoader&) = delete;

  // Synchronous read of one record into fresh MLX buffers (used for pinning).
  ExpertViews read_record(ExpertKey key) const;

  // Start of a forward pass: queued prefetches from older passes are dropped.
  void new_token();
  // Experts fetched for earlier layers of this pass become evictable.
  void begin_layer(int layer);

  bool has(ExpertKey key) const { return ready_[index(key)].load(std::memory_order_acquire); }
  const std::vector<std::atomic<bool>>& ready_mask() const { return ready_; }

  // Drop queued (not yet started) prefetches for `layer` not in `keep`.
  void retarget(int layer, const std::set<int>& keep);
  void request_many(const std::vector<ExpertKey>& keys, int prio);
  void request(ExpertKey key, int prio);
  // Blocking fetch.
  ExpertViews get(ExpertKey key);

  LoaderStats snapshot_stats() const;
  void reset_stats();
  uint64_t budget() const { return budget_; }
  uint64_t record_bytes() const { return layout_.record_bytes; }
  int n_groups() const { return (int)groups_.size(); }

 private:
  struct Group {
    int first_part, n_parts;
    uint64_t offset; // within the record
    uint64_t bytes;  // padded bytes covered
  };
  struct Entry {
    uint64_t id = 0;
    ExpertKey key;
    uint64_t size;
    int prio;
    uint64_t generation;
    bool ready = false;
    bool inflight = false;
    bool dropped = false;
    bool failed = false;
    bool prefetched;
    bool used = false;
    int remaining = 0;
    double t_start = 0;
    std::pair<int, int> last_use{-1, -1};
    std::vector<mx::allocator::Buffer> bufs;
    std::shared_ptr<ExpertViews> views;
    std::list<ExpertKey>::iterator lru_it;
    bool in_lru = false;
  };
  struct Job {
    int prio;
    uint64_t seq;
    ExpertKey key;
    uint64_t id; // entry the job belongs to; stale jobs of dropped entries are skipped
    int group;
    bool operator>(const Job& o) const { return std::tie(prio, seq) > std::tie(o.prio, o.seq); }
  };

  size_t index(ExpertKey k) const { return (size_t)k.first * layout_.n_experts + k.second; }
  void push_locked(int prio, const std::shared_ptr<Entry>& e);
  void drop_locked(std::shared_ptr<Entry>& e);
  bool evict_locked(uint64_t need);
  bool next_job_locked(std::shared_ptr<Entry>& out, int* group);
  void release_bufs(Entry& e);
  void worker();
  void read_group(int fd, uint64_t record_off, const Group& g, const std::vector<mx::allocator::Buffer>& bufs) const;

  std::string dir_;
  ExpertLayout layout_;
  uint64_t budget_;
  uint64_t read_len_;              // bytes of a record that hold parts (without trailing pad)
  std::vector<int> fds_;
  std::vector<mx::Shape> shapes_;
  std::vector<mx::Dtype> dtypes_;
  std::vector<Group> groups_;
  std::vector<std::atomic<bool>> ready_;

  mutable std::mutex mutex_;
  std::condition_variable cv_work_;
  std::condition_variable cv_ready_;
  std::map<ExpertKey, std::shared_ptr<Entry>> entries_;
  std::list<ExpertKey> lru_;       // front = oldest
  std::priority_queue<Job, std::vector<Job>, std::greater<Job>> queue_;
  uint64_t seq_ = 0;
  uint64_t next_id_ = 0;
  uint64_t generation_ = 0;
  int token_ = 0;
  int layer_ = -1;
  uint64_t resident_bytes_ = 0;
  bool stop_ = false;
  LoaderStats stats_;
  std::vector<std::thread> threads_;
};

} // namespace mmr
