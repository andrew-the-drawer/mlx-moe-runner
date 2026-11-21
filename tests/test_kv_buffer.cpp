// KV ring buffers: reads must return exactly the positions a linear cache would, through
// wrap-around, chunked writes and rollbacks; host pointers must see GPU writes and vice versa.
#include <cstdio>
#include <random>
#include <vector>

#include "engine/kv_buffer.h"

using namespace mmr;

namespace {
int failures = 0;
#define CHECK(c, ...)            \
  do {                           \
    if (!(c)) {                  \
      printf("FAIL: " __VA_ARGS__); \
      printf("\n");              \
      failures++;                \
    }                            \
  } while (0)

constexpr int H = 2, HD = 4;

float value(int pos, int version, int h, int d, int kv) { return pos * 1000.f + version * 100.f + h * 10.f + d + kv * 0.5f; }

mx::array rows(int start, int n, int version, int kv) {
  std::vector<float> v((size_t)H * n * HD);
  for (int h = 0; h < H; h++)
    for (int p = 0; p < n; p++)
      for (int d = 0; d < HD; d++) v[((size_t)h * n + p) * HD + d] = value(start + p, version, h, d, kv);
  return mx::array(v.data(), {1, H, n, HD}, mx::float32);
}

// Checks buf.read(first, end) against the versions recorded for each position.
bool check_read(KVBuffer& buf, int first, int end, const std::vector<int>& ver) {
  auto [K, V] = buf.read(first, end);
  mx::eval(K, V);
  auto Kc = mx::contiguous(K), Vc = mx::contiguous(V);
  mx::eval(Kc, Vc);
  int n = end - first;
  const float* k = Kc.data<float>();
  const float* v = Vc.data<float>();
  for (int h = 0; h < H; h++)
    for (int p = 0; p < n; p++)
      for (int d = 0; d < HD; d++) {
        size_t i = ((size_t)h * n + p) * HD + d;
        if (k[i] != value(first + p, ver[first + p], h, d, 0) || v[i] != value(first + p, ver[first + p], h, d, 1))
          return false;
      }
  return true;
}
} // namespace

int main() {
  const int W = 8, SLACK = 6, SLOTS = W + SLACK, CTX = 400;

  // A ring written continuously to 100 can roll back slack + 1 positions.
  CHECK(kv_ring_ok(W, 100 - SLOTS, 100 - (SLACK + 1)), "holds at the boundary");
  CHECK(!kv_ring_ok(W, 100 - SLOTS, 100 - (SLACK + 2)), "holds past the boundary");
  CHECK(kv_ring_ok(W, 0, 0) && kv_ring_ok(0, 500, 3), "never wrapped / linear");
  CHECK(kv_ring_min(W, 100 - SLOTS) == 100 - (SLACK + 1), "ring min");
  CHECK(kv_rows(SLOTS, W, 5, 3) == 5 && kv_rows(SLOTS, W, 50, 3) == SLOTS && kv_rows(CTX, 0, 50, 3) == 3, "rows");

  // Two rollbacks in a row: the second can't go as far back as a fresh state could,
  // because the writes before the first rollback already overwrote older slots.
  {
    KVBuffer ring(H, HD, SLOTS, W);
    for (int p = 0; p < 98; p += 7) ring.write(rows(p, 7, 0, 0), rows(p, 7, 0, 1), p);
    ring.write(rows(98, 2, 0, 0), rows(98, 2, 0, 1), 98); // end 100, lo 86
    CHECK(ring.holds(93) && !ring.holds(92), "first rollback limit");
    ring.write(rows(93, 2, 1, 0), rows(93, 2, 1, 1), 93); // end 95, lo still 86
    CHECK(ring.holds(93) && !ring.holds(92), "second rollback limited by history (fresh formula would allow 88)");
    bool threw = false;
    try {
      ring.write(rows(90, 1, 2, 0), rows(90, 1, 2, 1), 90);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "write past the valid range must throw");
    CHECK(ring.holds(0), "restarting from 0 needs no history");
    ring.write(rows(0, 5, 3, 0), rows(0, 5, 3, 1), 0);
    CHECK(ring.lo() == 0 && ring.holds(1), "a write at 0 resets the history");
    std::vector<int> ver(CTX, 3);
    CHECK(check_read(ring, 0, 5, ver), "read after restart");
  }

  // Random chunked writes with rollbacks, read back as the windowed attention would.
  std::mt19937 rng(7);
  for (int trial = 0; trial < 20; trial++) {
    KVBuffer ring(H, HD, SLOTS, W);
    KVBuffer lin(H, HD, CTX, 0);
    std::vector<int> ver(CTX, -1);
    int pos = 0, version = 0;
    bool ok = true;
    while (pos < CTX - SLACK - 1 && ok) {
      int n = 1 + (int)(rng() % (SLACK + 1));
      ring.write(rows(pos, n, version, 0), rows(pos, n, version, 1), pos);
      lin.write(rows(pos, n, version, 0), rows(pos, n, version, 1), pos);
      for (int p = pos; p < pos + n; p++) ver[p] = version;
      int first = std::max(0, pos - W + 1);
      bool ok_ring = check_read(ring, first, pos + n, ver), ok_lin = check_read(lin, 0, pos + n, ver);
      ok = ok_ring && ok_lin;
      CHECK(ok, "trial %d: read [%d, %d) after writing %d rows at %d (ring %d, linear %d)", trial, first, pos + n, n,
            pos, ok_ring, ok_lin);
      pos += n;
      if (rng() % 4 == 0) {
        // Roll back as far as the ring allows (often exactly to the limit).
        int lo_m = kv_ring_min(W, ring.lo());
        int m = rng() % 2 ? lo_m : lo_m + (int)(rng() % (pos - lo_m + 1));
        if (rng() % 8 == 0) m = 0; // full re-prefill
        m = std::min(std::max(m, 0), pos);
        CHECK(ring.holds(m) && (m == 0 || !ring.holds(m - 1) || m > lo_m), "rollback %d -> %d", pos, m);
        pos = m;
        version++;
      }
    }
  }

  // Too many rows at once for the ring is rejected.
  {
    KVBuffer ring(H, HD, SLOTS, W);
    bool threw = false;
    try {
      ring.write(rows(0, SLACK + 2, 0, 0), rows(0, SLACK + 2, 0, 1), 0);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw, "oversized ring write accepted");
  }

  // Host pointers: see GPU writes; CPU writes are seen by later reads.
  {
    KVBuffer ring(H, HD, SLOTS, W);
    std::vector<int> ver(CTX, 0);
    for (int p = 0; p < 20; p += 4) ring.write(rows(p, 4, 0, 0), rows(p, 4, 0, 1), p);
    float* k = ring.k_data();
    // Position 19 lives in slot 19 % 14 = 5.
    CHECK(k[(size_t)(0 * SLOTS + 5) * HD + 2] == value(19, 0, 0, 2, 0), "host view of GPU write");
    std::vector<float> saved(k, k + (size_t)H * SLOTS * HD);
    ring.write(rows(20, 4, 3, 0), rows(20, 4, 3, 1), 20);
    float* k2 = ring.k_data();
    std::copy(saved.begin(), saved.end(), k2); // restore K as it was at position 20
    for (int p = 20; p < 24; p++) ver[p] = 3;
    auto [K, V] = ring.read(13, 20);
    mx::eval(K);
    auto Kc = mx::contiguous(K);
    mx::eval(Kc);
    bool same = true;
    for (int p = 0; p < 7; p++)
      if (Kc.data<float>()[(size_t)p * HD] != value(13 + p, 0, 0, 0, 0)) same = false;
    CHECK(same, "CPU restore visible to reads");
  }

  if (failures == 0) printf("test_kv_buffer: all passed\n");
  return failures == 0 ? 0 : 1;
}
