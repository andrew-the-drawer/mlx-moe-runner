#include "io/cold_loader.h"

#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

#include "util/log.h"
#include "util/mlx_util.h"

namespace mmr {

namespace {
double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string matrix_of(const std::string& part_name) {
  size_t dot = part_name.find('.');
  return dot == std::string::npos ? part_name : part_name.substr(0, dot);
}
} // namespace

ColdLoader::ColdLoader(const std::string& dir, const ExpertLayout& layout, uint64_t budget_bytes, int n_threads,
                       bool nocache, int split)
    : dir_(dir), layout_(layout), budget_(budget_bytes), ready_((size_t)layout.n_layers * layout.n_experts) {
  for (auto& r : ready_) r.store(false);
  fds_.assign(layout.n_layers, -1);
  for (int l = 0; l < layout.n_layers; l++) {
    std::string path = dir + "/" + layout.file_for(l);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open expert file " + path);
    if (nocache) {
      fcntl(fd, F_NOCACHE, 1);
      fcntl(fd, F_RDAHEAD, 0);
    }
    fds_[l] = fd;
  }
  read_len_ = 0;
  for (auto& p : layout.parts) {
    shapes_.push_back(mx::Shape(p.shape.begin(), p.shape.end()));
    dtypes_.push_back(dtype_from_name(p.dtype));
    read_len_ = std::max(read_len_, p.offset + p.padded);
  }
  // Sub-read groups: one per matrix (parts of a matrix are contiguous), or one for all.
  if (split == 1) {
    groups_.push_back({0, (int)layout.parts.size(), 0, read_len_});
  } else {
    for (size_t i = 0; i < layout.parts.size(); i++) {
      const auto& p = layout.parts[i];
      if (!groups_.empty() && matrix_of(layout.parts[groups_.back().first_part].name) == matrix_of(p.name)) {
        groups_.back().n_parts++;
        groups_.back().bytes = p.offset + p.padded - groups_.back().offset;
      } else {
        groups_.push_back({(int)i, 1, p.offset, p.padded});
      }
    }
  }
  for (int i = 0; i < std::max(1, n_threads); i++) threads_.emplace_back([this]() { worker(); });
}

ColdLoader::~ColdLoader() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_work_.notify_all();
  for (auto& t : threads_) t.join();
  for (int fd : fds_)
    if (fd >= 0) ::close(fd);
}

void ColdLoader::read_group(int fd, uint64_t record_off, const Group& g,
                            const std::vector<mx::allocator::Buffer>& bufs) const {
  std::vector<struct iovec> iov;
  for (int i = g.first_part; i < g.first_part + g.n_parts; i++)
    iov.push_back({const_cast<void*>(bufs[i].ptr() ? (const void*)const_cast<mx::allocator::Buffer&>(bufs[i]).raw_ptr() : nullptr),
                   (size_t)layout_.parts[i].padded});
  size_t expected = g.bytes, done = 0;
  off_t off = (off_t)(record_off + g.offset);
  while (done < expected) {
    std::vector<struct iovec> rem;
    size_t skip = done;
    for (auto& v : iov) {
      if (skip >= v.iov_len) {
        skip -= v.iov_len;
        continue;
      }
      rem.push_back({(char*)v.iov_base + skip, v.iov_len - skip});
      skip = 0;
    }
    ssize_t n = preadv(fd, rem.data(), (int)rem.size(), off + (off_t)done);
    if (n <= 0) throw std::runtime_error(std::string("preadv failed: ") + strerror(errno));
    done += (size_t)n;
  }
}

ExpertViews ColdLoader::read_record(ExpertKey key) const {
  std::vector<mx::allocator::Buffer> bufs;
  for (auto& p : layout_.parts) {
    auto buf = mx::allocator::malloc(p.padded);
    if (!buf.ptr()) throw std::runtime_error("MLX allocation failed for expert part");
    bufs.push_back(buf);
  }
  try {
    Group all{0, (int)layout_.parts.size(), 0, read_len_};
    read_group(fds_[key.first], layout_.expert_offset(key.second), all, bufs);
  } catch (...) {
    for (auto& b : bufs) mx::allocator::free(b);
    throw;
  }
  ExpertViews views;
  for (size_t i = 0; i < bufs.size(); i++) views.parts.push_back(array_from_buffer(bufs[i], shapes_[i], dtypes_[i]));
  return views;
}

void ColdLoader::new_token() {
  std::lock_guard<std::mutex> lock(mutex_);
  generation_++;
  token_++;
  layer_ = -1;
}

void ColdLoader::begin_layer(int layer) {
  std::lock_guard<std::mutex> lock(mutex_);
  layer_ = layer;
}

void ColdLoader::push_locked(int prio, const std::shared_ptr<Entry>& e) {
  for (int g = 0; g < (int)groups_.size(); g++) queue_.push(Job{prio, ++seq_, e->key, e->id, g});
  cv_work_.notify_all();
}

void ColdLoader::drop_locked(std::shared_ptr<Entry>& e) {
  entries_.erase(e->key);
  stats_.n_dropped++;
  e->dropped = true;
  e->ready = true;
  cv_ready_.notify_all();
}

void ColdLoader::release_bufs(Entry& e) {
  for (auto& b : e.bufs) mx::allocator::free(b);
  e.bufs.clear();
}

void ColdLoader::retarget(int layer, const std::set<int>& keep) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::shared_ptr<Entry>> victims;
  for (auto& [key, e] : entries_) {
    if (key.first == layer && !e->inflight && !e->ready && e->prio != PRIO_DEMAND && !keep.count(key.second))
      victims.push_back(e);
  }
  for (auto& e : victims) drop_locked(e);
}

void ColdLoader::request_many(const std::vector<ExpertKey>& keys, int prio) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto key : keys) {
    stats_.n_requests++;
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      auto& e = it->second;
      if (e->ready && e->in_lru) {
        lru_.splice(lru_.end(), lru_, e->lru_it);
      } else if (!e->ready && !e->inflight && prio < e->prio) {
        e->prio = prio;
        e->generation = generation_;
        push_locked(prio, e);
      }
      continue;
    }
    auto e = std::make_shared<Entry>();
    e->id = ++next_id_;
    e->key = key;
    e->size = read_len_;
    e->prio = prio;
    e->generation = generation_;
    e->prefetched = prio != PRIO_DEMAND;
    entries_[key] = e;
    push_locked(prio, e);
  }
}

void ColdLoader::request(ExpertKey key, int prio) {
  request_many({key}, prio);
}

ExpertViews ColdLoader::get(ExpertKey key) {
  double waited = 0;
  while (true) {
    std::shared_ptr<Entry> e;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stats_.n_requests++;
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        e = std::make_shared<Entry>();
        e->id = ++next_id_;
        e->key = key;
        e->size = read_len_;
        e->prio = PRIO_DEMAND;
        e->generation = generation_;
        e->prefetched = false;
        entries_[key] = e;
        push_locked(PRIO_DEMAND, e);
      } else {
        e = it->second;
        if (!e->ready && !e->inflight && e->prio != PRIO_DEMAND) {
          e->prio = PRIO_DEMAND;
          e->generation = generation_;
          push_locked(PRIO_DEMAND, e);
        }
      }
      if (!e->ready) {
        double t0 = now_s();
        cv_ready_.wait(lock, [&]() { return e->ready; });
        waited += now_s() - t0;
      }
      if (e->dropped || !e->views) continue; // dropped or failed before it ran; retry
      if (e->prefetched && !e->used) stats_.n_prefetch_hit++;
      if (waited > 0) {
        stats_.demand_wait_s += waited;
        stats_.n_demand++;
      }
      e->used = true;
      e->last_use = {token_, layer_};
      if (e->in_lru) lru_.splice(lru_.end(), lru_, e->lru_it);
      return *e->views;
    }
  }
}

bool ColdLoader::evict_locked(uint64_t need) {
  std::pair<int, int> protected_use{token_, layer_};
  while (resident_bytes_ + need > budget_ && !lru_.empty()) {
    auto victim = lru_.end();
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
      auto& e = entries_.at(*it);
      if (e->last_use != protected_use) {
        victim = it;
        break;
      }
    }
    if (victim == lru_.end()) return false;
    auto e = entries_.at(*victim);
    entries_.erase(*victim);
    lru_.erase(victim);
    ready_[index(e->key)].store(false, std::memory_order_release);
    resident_bytes_ -= e->size;
    stats_.n_evict++;
    if (e->prefetched && !e->used) stats_.n_prefetch_wasted++;
    e->views.reset(); // arrays are freed here; buffers return to the MLX cache
  }
  return resident_bytes_ + need <= budget_;
}

bool ColdLoader::next_job_locked(std::shared_ptr<Entry>& out, int* group) {
  while (!queue_.empty()) {
    Job job = queue_.top();
    queue_.pop();
    auto it = entries_.find(job.key);
    if (it == entries_.end() || it->second->id != job.id) continue; // entry dropped/replaced
    auto e = it->second;
    if (e->ready) continue;
    if (!e->inflight) {
      // First sub-read of this record: admission control and allocation.
      if (e->prio != PRIO_DEMAND && e->generation != generation_) {
        drop_locked(e);
        continue;
      }
      if (!evict_locked(e->size)) {
        if (e->prio != PRIO_DEMAND) {
          drop_locked(e);
          continue;
        }
        stats_.n_overflow++;
      }
      e->bufs.reserve(layout_.parts.size());
      for (auto& p : layout_.parts) {
        auto buf = mx::allocator::malloc(p.padded);
        if (!buf.ptr()) {
          release_bufs(*e);
          throw std::runtime_error("MLX allocation failed for expert part");
        }
        e->bufs.push_back(buf);
      }
      e->inflight = true;
      e->remaining = (int)groups_.size();
      e->t_start = now_s();
      resident_bytes_ += e->size;
    }
    out = e;
    *group = job.group;
    return true;
  }
  return false;
}

void ColdLoader::worker() {
  while (true) {
    std::shared_ptr<Entry> e;
    int group = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (!stop_ && !next_job_locked(e, &group)) cv_work_.wait(lock);
      if (stop_) return;
    }
    std::string error;
    try {
      read_group(fds_[e->key.first], layout_.expert_offset(e->key.second), groups_[group], e->bufs);
    } catch (const std::exception& ex) {
      error = ex.what();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error.empty()) {
      MMR_ERROR("expert read failed: %s", error.c_str());
      e->failed = true;
    }
    if (--e->remaining > 0) continue;
    if (e->ready) continue; // defensive: never publish twice
    if (e->failed) {
      release_bufs(*e);
      resident_bytes_ -= e->size;
      e->inflight = false;
      entries_.erase(e->key);
      e->dropped = true;
      e->ready = true;
      cv_ready_.notify_all();
      continue;
    }
    auto views = std::make_shared<ExpertViews>();
    for (size_t i = 0; i < e->bufs.size(); i++)
      views->parts.push_back(array_from_buffer(e->bufs[i], shapes_[i], dtypes_[i]));
    e->bufs.clear(); // ownership moved to the arrays
    stats_.bytes_read += read_len_;
    stats_.n_reads++;
    stats_.read_time_s += now_s() - e->t_start;
    e->views = views;
    e->inflight = false;
    e->ready = true;
    lru_.push_back(e->key);
    e->lru_it = std::prev(lru_.end());
    e->in_lru = true;
    ready_[index(e->key)].store(true, std::memory_order_release);
    cv_ready_.notify_all();
  }
}

LoaderStats ColdLoader::snapshot_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  LoaderStats s = stats_;
  s.resident_bytes = resident_bytes_;
  s.n_resident = lru_.size();
  s.queued = queue_.size();
  return s;
}

void ColdLoader::reset_stats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = LoaderStats{};
}

} // namespace mmr
