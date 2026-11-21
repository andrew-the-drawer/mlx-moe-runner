#include "engine/kv_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#include "engine/kv_buffer.h"
#include "util/log.h"

namespace fs = std::filesystem;

namespace mmr {

namespace {

constexpr uint64_t kAlign = 16384;
constexpr char kMagic[8] = {'M', 'M', 'R', 'K', 'V', '0', '0', '1'};

uint64_t align_up(uint64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// preadv/pwritev until every byte moved (they may return short counts).
bool pio_all(int fd, std::vector<struct iovec> iov, off_t off, bool write) {
  size_t i = 0;
  while (i < iov.size()) {
    int cnt = (int)std::min<size_t>(iov.size() - i, IOV_MAX);
    ssize_t r = write ? ::pwritev(fd, &iov[i], cnt, off) : ::preadv(fd, &iov[i], cnt, off);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (r == 0) return false;
    off += r;
    while (r > 0 && i < iov.size()) {
      if ((size_t)r >= iov[i].iov_len) {
        r -= iov[i].iov_len;
        i++;
      } else {
        iov[i].iov_base = (uint8_t*)iov[i].iov_base + r;
        iov[i].iov_len -= r;
        r = 0;
      }
    }
  }
  return true;
}

size_t common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
  size_t n = std::min(a.size(), b.size()), i = 0;
  while (i < n && a[i] == b[i]) i++;
  return i;
}

} // namespace

const char* tier_name(KVTier t) {
  switch (t) {
    case KVTier::Ram: return "ram";
    case KVTier::Disk: return "disk";
    default: return "none";
  }
}

int KVGeometry::min_reuse(int lo) const {
  int m = 0;
  for (int l = 0; l < n_layers(); l++) m = std::max(m, kv_ring_min(window[l], lo));
  return m;
}

std::string KVGeometry::fingerprint() const {
  std::string s = model_id + "/f32/h" + std::to_string(n_kv_heads) + "x" + std::to_string(head_dim) + "/";
  for (int l = 0; l < n_layers(); l++)
    s += window[l] > 0 ? "r" + std::to_string(slots[l]) + "w" + std::to_string(window[l]) + "," : "f,";
  return s;
}

KVStore::KVStore(const KVGeometry& geom, const KVStoreOptions& opts) : geom_(geom), opts_(opts) {
  if (opts_.disk_bytes > 0) {
    if (opts_.dir.empty()) throw std::runtime_error("KV store: SSD tier needs a directory");
    fs::create_directories(opts_.dir);
    scan_dir();
  }
}

KVStore::~KVStore() {
  for (auto& e : entries_) free(e.blob);
}

std::vector<KVStore::Section> KVStore::layout(int n, uint64_t* total) const {
  std::vector<Section> out;
  uint64_t off = 0;
  for (int l = 0; l < geom_.n_layers(); l++) {
    int rows = kv_rows(geom_.slots[l], geom_.window[l], n, n);
    uint64_t bytes = (uint64_t)geom_.n_kv_heads * rows * geom_.head_dim * sizeof(float);
    for (int kv = 0; kv < 2; kv++) {
      out.push_back({l, kv, rows, off});
      off += align_up(bytes);
    }
  }
  if (total) *total = off;
  return out;
}

uint64_t KVStore::snapshot_bytes(int n) const {
  uint64_t total = 0;
  layout(n, &total);
  return total;
}

KVStore::Entry* KVStore::find_entry(uint64_t id) {
  for (auto& e : entries_)
    if (e.id == id) return &e;
  return nullptr;
}

uint64_t KVStore::used(KVTier t) const {
  uint64_t u = 0;
  for (auto& e : entries_)
    if (e.tier == t) u += t == KVTier::Ram ? e.data_bytes : e.file_bytes;
  return u;
}

void KVStore::drop(size_t idx) {
  auto& e = entries_[idx];
  free(e.blob);
  if (e.tier == KVTier::Disk && !e.path.empty()) ::unlink(e.path.c_str());
  entries_.erase(entries_.begin() + idx);
}

KVStore::Match KVStore::find(const std::vector<int>& tokens, int max_len) const {
  std::lock_guard<std::mutex> lock(mutex_);
  Match best;
  uint64_t best_use = 0;
  for (auto& e : entries_) {
    int n = (int)e.tokens.size();
    int m = (int)std::min<size_t>(common_prefix(e.tokens, tokens), (size_t)std::max(0, max_len));
    if (m <= 0 || m < geom_.min_reuse(e.lo)) continue;
    bool better = m > best.len || (m == best.len && e.tier == KVTier::Ram && best.tier == KVTier::Disk) ||
                  (m == best.len && e.tier == best.tier && e.last_use > best_use);
    if (better) {
      best = {e.id, m, n, e.lo, e.tier};
      best_use = e.last_use;
    }
  }
  return best;
}

std::string KVStore::header_for(const Entry& e, uint64_t* header_bytes) const {
  nlohmann::json j = {{"fingerprint", geom_.fingerprint()}, {"lo", e.lo}, {"tokens", e.tokens}};
  std::string body = j.dump();
  std::string h(16, '\0');
  memcpy(&h[0], kMagic, 8);
  uint64_t len = body.size();
  memcpy(&h[8], &len, 8);
  h += body;
  *header_bytes = align_up(h.size());
  h.resize(*header_bytes, '\0');
  return h;
}

bool KVStore::write_file(Entry& e, const uint8_t* blob, const KVPtrs* src) {
  uint64_t hb = 0;
  std::string header = header_for(e, &hb);
  std::string path = opts_.dir + "/kv-" + std::to_string(e.id) + ".mmrkv";
  std::string tmp = path + ".tmp";
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    MMR_WARN("KV store: cannot create %s: %s", tmp.c_str(), strerror(errno));
    return false;
  }
  if (opts_.nocache) fcntl(fd, F_NOCACHE, 1);
  bool ok = pio_all(fd, {{header.data(), header.size()}}, 0, true);
  if (ok && blob) {
    ok = pio_all(fd, {{(void*)blob, e.data_bytes}}, (off_t)hb, true);
  } else if (ok && src) {
    int H = geom_.n_kv_heads, hd = geom_.head_dim;
    for (auto& s : layout((int)e.tokens.size(), nullptr)) {
      float* base = s.kv == 0 ? src->k[s.layer] : src->v[s.layer];
      std::vector<struct iovec> iov;
      for (int h = 0; h < H; h++)
        iov.push_back({base + (size_t)h * geom_.slots[s.layer] * hd, (size_t)s.rows * hd * sizeof(float)});
      if (!(ok = pio_all(fd, iov, (off_t)(hb + s.offset), true))) break;
    }
  }
  if (ok) ok = ::ftruncate(fd, (off_t)(hb + e.data_bytes)) == 0;
  ::close(fd);
  if (ok) ok = ::rename(tmp.c_str(), path.c_str()) == 0;
  if (!ok) {
    MMR_WARN("KV store: writing %s failed: %s", path.c_str(), strerror(errno));
    ::unlink(tmp.c_str());
    stats_.io_errors++;
    return false;
  }
  e.path = path;
  e.header_bytes = hb;
  e.file_bytes = hb + e.data_bytes;
  stats_.bytes_written += e.file_bytes;
  return true;
}

void KVStore::make_room(KVTier tier, uint64_t bytes, uint64_t keep) {
  uint64_t budget = tier == KVTier::Ram ? opts_.ram_bytes : opts_.disk_bytes;
  while (used(tier) + bytes > budget) {
    size_t victim = entries_.size();
    for (size_t i = 0; i < entries_.size(); i++)
      if (entries_[i].tier == tier && entries_[i].id != keep && entries_[i].id != protect_ &&
          (victim == entries_.size() || entries_[i].last_use < entries_[victim].last_use))
        victim = i;
    if (victim == entries_.size()) return;
    if (tier == KVTier::Ram && opts_.disk_bytes > 0) {
      uint64_t id = entries_[victim].id;
      if (demote(entries_[victim])) continue;
      // demote() may have reordered entries_ while making room on disk.
      for (size_t i = 0; i < entries_.size(); i++)
        if (entries_[i].id == id) victim = i;
    }
    drop(victim);
    stats_.evicted++;
  }
}

bool KVStore::demote(Entry& e) {
  uint64_t id = e.id;
  uint64_t hb = 0;
  header_for(e, &hb);
  if (hb + e.data_bytes > opts_.disk_bytes) return false;
  make_room(KVTier::Disk, hb + e.data_bytes, id);
  Entry* p = find_entry(id); // make_room may have moved entries
  if (!p || !write_file(*p, p->blob, nullptr)) return false;
  free(p->blob);
  p->blob = nullptr;
  p->tier = KVTier::Disk;
  stats_.demoted++;
  MMR_INFO("KV store: moved %zu tokens (%s) from ram to disk", p->tokens.size(), human_bytes((double)p->data_bytes).c_str());
  return true;
}

KVTier KVStore::put(const std::vector<int>& tokens, const KVPtrs& src, int ring_lo, uint64_t keep) {
  std::lock_guard<std::mutex> lock(mutex_);
  int n = (int)tokens.size();
  if (!enabled() || n == 0) return KVTier::None;
  double t0 = now_s();

  Entry e;
  e.id = next_id_++;
  e.tokens = tokens;
  e.lo = ring_lo;
  e.last_use = ++clock_;
  layout(n, &e.data_bytes);
  uint64_t hb = 0;
  header_for(e, &hb);
  if (e.data_bytes > opts_.ram_bytes && (opts_.disk_bytes == 0 || hb + e.data_bytes > opts_.disk_bytes)) {
    stats_.skipped++;
    MMR_INFO("KV store: session of %d tokens (%s) exceeds both budgets; not saved", n,
             human_bytes((double)e.data_bytes).c_str());
    return KVTier::None;
  }
  protect_ = keep;

  // Drop entries this one makes redundant: a prefix of it whose reusable range it covers.
  int lo = geom_.min_reuse(ring_lo);
  for (size_t i = entries_.size(); i-- > 0;) {
    auto& o = entries_[i];
    int en = (int)o.tokens.size();
    if (o.id == keep || en > n || common_prefix(o.tokens, tokens) < (size_t)en) continue;
    if (geom_.min_reuse(o.lo) >= lo) {
      drop(i);
      stats_.dominated++;
    }
  }

  KVTier tier = KVTier::None;
  if (e.data_bytes <= opts_.ram_bytes) {
    make_room(KVTier::Ram, e.data_bytes, keep);
    if (used(KVTier::Ram) + e.data_bytes <= opts_.ram_bytes) {
      void* p = nullptr;
      if (posix_memalign(&p, kAlign, e.data_bytes) == 0) {
        e.blob = (uint8_t*)p;
        int H = geom_.n_kv_heads, hd = geom_.head_dim;
        for (auto& s : layout(n, nullptr)) {
          const float* base = s.kv == 0 ? src.k[s.layer] : src.v[s.layer];
          size_t chunk = (size_t)s.rows * hd * sizeof(float);
          for (int h = 0; h < H; h++)
            memcpy(e.blob + s.offset + h * chunk, base + (size_t)h * geom_.slots[s.layer] * hd, chunk);
        }
        e.tier = tier = KVTier::Ram;
        stats_.saved_ram++;
      }
    }
  }
  if (tier == KVTier::None && opts_.disk_bytes > 0 && hb + e.data_bytes <= opts_.disk_bytes) {
    make_room(KVTier::Disk, hb + e.data_bytes, keep);
    if (used(KVTier::Disk) + hb + e.data_bytes <= opts_.disk_bytes && write_file(e, nullptr, &src)) {
      e.tier = tier = KVTier::Disk;
      stats_.saved_disk++;
    }
  }
  protect_ = 0;
  double dt = now_s() - t0;
  if (tier == KVTier::None) {
    stats_.skipped++;
    MMR_WARN("KV store: could not save session of %d tokens (%s)", n, human_bytes((double)e.data_bytes).c_str());
    return tier;
  }
  stats_.save_s += dt;
  MMR_INFO("KV store: saved %d tokens (%s) to %s in %.3fs", n, human_bytes((double)e.data_bytes).c_str(),
           tier_name(tier), dt);
  entries_.push_back(std::move(e));
  return tier;
}

bool KVStore::take(uint64_t id, int m, const KVPtrs& dst) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t idx = entries_.size();
  for (size_t i = 0; i < entries_.size(); i++)
    if (entries_[i].id == id) idx = i;
  if (idx == entries_.size()) return false;
  double t0 = now_s();
  auto& e = entries_[idx];
  int n = (int)e.tokens.size();
  int H = geom_.n_kv_heads, hd = geom_.head_dim;
  auto sections = layout(n, nullptr);
  bool ok = true;
  int fd = -1;
  if (e.tier == KVTier::Disk) {
    fd = ::open(e.path.c_str(), O_RDONLY);
    if (fd < 0) ok = false;
    else if (opts_.nocache) {
      fcntl(fd, F_NOCACHE, 1);
      fcntl(fd, F_RDAHEAD, 0);
    }
  }
  uint64_t bytes = 0;
  for (auto& s : sections) {
    if (!ok) break;
    int l = s.layer;
    float* base = s.kv == 0 ? dst.k[l] : dst.v[l];
    size_t stored = (size_t)s.rows * hd * sizeof(float);
    size_t copy = (size_t)kv_rows(geom_.slots[l], geom_.window[l], n, m) * hd * sizeof(float);
    std::vector<struct iovec> iov;
    for (int h = 0; h < H; h++) {
      float* d = base + (size_t)h * geom_.slots[l] * hd;
      if (e.tier == KVTier::Ram) memcpy(d, e.blob + s.offset + h * stored, copy);
      else if (copy == stored) iov.push_back({d, copy});
      else ok = ok && pio_all(fd, {{d, copy}}, (off_t)(e.header_bytes + s.offset + h * stored), false);
      bytes += copy;
    }
    if (ok && !iov.empty()) ok = pio_all(fd, iov, (off_t)(e.header_bytes + s.offset), false);
  }
  if (fd >= 0) ::close(fd);
  double dt = now_s() - t0;
  KVTier tier = e.tier;
  if (!ok) {
    stats_.io_errors++;
    MMR_WARN("KV store: restoring %s failed: %s", e.path.c_str(), strerror(errno));
  } else {
    stats_.restore_s += dt;
    stats_.tokens_restored += m;
    if (tier == KVTier::Ram) stats_.hits_ram++;
    else {
      stats_.hits_disk++;
      stats_.bytes_read += bytes;
    }
    MMR_INFO("KV store: restored %d of %d tokens (%s) from %s in %.3fs", m, n, human_bytes((double)bytes).c_str(),
             tier_name(tier), dt);
  }
  drop(idx);
  return ok;
}

void KVStore::scan_dir() {
  std::string fp = geom_.fingerprint();
  struct Found {
    Entry e;
    fs::file_time_type mtime;
  };
  std::vector<Found> found;
  int stale = 0;
  for (auto& de : fs::directory_iterator(opts_.dir)) {
    if (!de.is_regular_file()) continue;
    std::string name = de.path().filename().string();
    if (name.rfind("kv-", 0) != 0) continue;
    if (de.path().extension() == ".tmp") {
      fs::remove(de.path());
      continue;
    }
    if (de.path().extension() != ".mmrkv") continue;
    Entry e;
    e.tier = KVTier::Disk;
    e.path = de.path().string();
    bool valid = false;
    int fd = ::open(e.path.c_str(), O_RDONLY);
    if (fd >= 0) {
      char head[16];
      if (::pread(fd, head, 16, 0) == 16 && memcmp(head, kMagic, 8) == 0) {
        uint64_t len = 0;
        memcpy(&len, head + 8, 8);
        std::string body(len, '\0');
        if (len < (64u << 20) && ::pread(fd, body.data(), len, 16) == (ssize_t)len) {
          try {
            auto j = nlohmann::json::parse(body);
            if (j.value("fingerprint", "") == fp) {
              e.tokens = j.at("tokens").get<std::vector<int>>();
              e.lo = j.value("lo", 0);
              layout((int)e.tokens.size(), &e.data_bytes);
              e.header_bytes = align_up(16 + len);
              e.file_bytes = e.header_bytes + e.data_bytes;
              valid = !e.tokens.empty() && (uint64_t)de.file_size() >= e.file_bytes;
            }
          } catch (const std::exception&) {
          }
        }
      }
      ::close(fd);
    }
    if (!valid) {
      // Snapshots of other weights or another ring geometry (--chunk) can never be used.
      fs::remove(de.path());
      stale++;
      continue;
    }
    try {
      e.id = std::stoull(name.substr(3, name.size() - 3 - 6));
    } catch (const std::exception&) {
      continue;
    }
    next_id_ = std::max(next_id_, e.id + 1);
    found.push_back({std::move(e), de.last_write_time()});
  }
  std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.mtime < b.mtime; });
  for (auto& f : found) {
    f.e.last_use = ++clock_;
    entries_.push_back(std::move(f.e));
  }
  make_room(KVTier::Disk, 0, 0);
  if (!entries_.empty() || stale)
    MMR_INFO("KV store: %zu snapshots on disk (%s) in %s%s", entries_.size(), human_bytes((double)used(KVTier::Disk)).c_str(),
             opts_.dir.c_str(), stale ? (", removed " + std::to_string(stale) + " stale").c_str() : "");
}

nlohmann::json KVStore::stats_json() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t n_ram = 0, n_disk = 0;
  for (auto& e : entries_) (e.tier == KVTier::Ram ? n_ram : n_disk)++;
  const auto& s = stats_;
  return {{"ram", {{"entries", n_ram}, {"bytes", used(KVTier::Ram)}, {"budget", opts_.ram_bytes}}},
          {"disk", {{"entries", n_disk}, {"bytes", used(KVTier::Disk)}, {"budget", opts_.disk_bytes}, {"dir", opts_.dir}}},
          {"saved_ram", s.saved_ram}, {"saved_disk", s.saved_disk}, {"demoted", s.demoted}, {"evicted", s.evicted},
          {"dominated", s.dominated}, {"skipped", s.skipped}, {"hits_ram", s.hits_ram}, {"hits_disk", s.hits_disk},
          {"tokens_restored", s.tokens_restored}, {"bytes_written", s.bytes_written}, {"bytes_read", s.bytes_read},
          {"save_s", s.save_s}, {"restore_s", s.restore_s}, {"io_errors", s.io_errors}};
}

} // namespace mmr
