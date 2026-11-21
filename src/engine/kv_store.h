#pragma once
// Session KV store: snapshots of sequences that are no longer the active one, kept in
// a RAM tier (bounded blobs) and an SSD tier (one file per snapshot), both LRU. A
// snapshot lives in exactly one tier; RAM entries are demoted to SSD when RAM is full,
// and an entry leaves the store when it is restored into the active cache.
//
// Data layout (the RAM blob and the file body are identical): for each layer, K then
// V, each a section of H_kv chunks of `rows * head_dim` float32, sections aligned to
// 16 KiB. A file adds a 16 KiB-aligned header (magic, JSON with fingerprint + tokens).
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mmr {

struct KVGeometry {
  int n_kv_heads = 0;
  int head_dim = 0;
  std::vector<int> slots;   // per layer: buffer slots (ctx for full attention)
  std::vector<int> window;  // per layer: sliding window, 0 for full attention
  std::string model_id;     // identifies the weights (hash of config.json)

  int n_layers() const { return (int)slots.size(); }
  // Smallest position a snapshot whose rings are valid from `lo` can continue from.
  int min_reuse(int lo) const;
  // Excludes full-attention slot counts, so snapshots survive a different --ctx.
  std::string fingerprint() const;
};

// Per-layer base pointers of K and V buffers laid out [H_kv, slots, head_dim].
struct KVPtrs {
  std::vector<float*> k, v;
};

enum class KVTier { None, Ram, Disk };
const char* tier_name(KVTier t);

struct KVStoreOptions {
  uint64_t ram_bytes = 0;
  uint64_t disk_bytes = 0;
  std::string dir;           // SSD tier directory (created if missing)
  bool nocache = true;       // F_NOCACHE on snapshot files
};

class KVStore {
 public:
  KVStore(const KVGeometry& geom, const KVStoreOptions& opts);
  ~KVStore();
  KVStore(const KVStore&) = delete;
  KVStore& operator=(const KVStore&) = delete;

  bool enabled() const { return opts_.ram_bytes > 0 || opts_.disk_bytes > 0; }

  struct Match {
    uint64_t id = 0;
    int len = 0;               // positions reusable for this request
    int n = 0;                 // positions in the snapshot
    int lo = 0;                // lowest valid ring position of the snapshot
    KVTier tier = KVTier::None;
  };
  // The entry that restores the longest prefix of `tokens`, capped at max_len.
  Match find(const std::vector<int>& tokens, int max_len) const;

  // Stores the state after `tokens` (n = tokens.size()). Full-attention layers are read
  // from `src` rows [0, n); sliding-window layers from their ring, valid from `ring_lo`.
  // Entries this one makes redundant are dropped. Entry `keep` is never evicted to make
  // room. Returns the tier used.
  KVTier put(const std::vector<int>& tokens, const KVPtrs& src, int ring_lo, uint64_t keep = 0);

  // Copies entry `id`, truncated to m positions, into `dst` and removes it from the
  // store. On an I/O error returns false; `dst` may then be partially overwritten.
  bool take(uint64_t id, int m, const KVPtrs& dst);

  nlohmann::json stats_json() const;
  // Bytes of a snapshot of n positions (data only, excluding the file header).
  uint64_t snapshot_bytes(int n) const;

 private:
  struct Entry {
    uint64_t id = 0;
    std::vector<int> tokens;
    int lo = 0;
    KVTier tier = KVTier::None;
    uint64_t data_bytes = 0;
    uint64_t file_bytes = 0;     // header + data, disk tier
    uint64_t header_bytes = 0;   // disk tier
    uint8_t* blob = nullptr;     // RAM tier
    std::string path;            // disk tier
    uint64_t last_use = 0;
  };
  struct Section {
    int layer, kv;               // kv: 0 = K, 1 = V
    int rows;                    // rows stored per head
    uint64_t offset;             // within the data area
  };

  std::vector<Section> layout(int n, uint64_t* total) const;
  Entry* find_entry(uint64_t id);
  void drop(size_t idx);
  void make_room(KVTier tier, uint64_t bytes, uint64_t keep);
  bool demote(Entry& e);
  bool write_file(Entry& e, const uint8_t* blob, const KVPtrs* src);
  std::string header_for(const Entry& e, uint64_t* header_bytes) const;
  void scan_dir();
  uint64_t used(KVTier t) const;

  KVGeometry geom_;
  KVStoreOptions opts_;
  std::vector<Entry> entries_;
  uint64_t next_id_ = 1;
  uint64_t clock_ = 0;
  uint64_t protect_ = 0;        // entry put() must not evict (the one about to be restored)
  mutable std::mutex mutex_;    // public methods lock; the server reads stats while generating
  struct Stats {
    uint64_t saved_ram = 0, saved_disk = 0, demoted = 0, evicted = 0, skipped = 0, dominated = 0;
    uint64_t hits_ram = 0, hits_disk = 0, tokens_restored = 0, io_errors = 0;
    uint64_t bytes_written = 0, bytes_read = 0;
    double save_s = 0, restore_s = 0;
  } stats_;
};

} // namespace mmr
