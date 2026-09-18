// Repackers must dequantize (via MLX) to the same values as the ggml decoders.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <mlx/mlx.h>

#include "gguf/ggml_quants.h"
#include "util/mlx_util.h"

using namespace mmr;
namespace mx = mlx::core;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } } while (0)

static std::mt19937 rng(7);

static uint16_t rand_f16_scale() {
  std::uniform_real_distribution<float> u(0.001f, 0.5f);
  return quant::f32_to_f16(u(rng));
}

static void fill_random(std::vector<uint8_t>& v) {
  std::uniform_int_distribution<int> d(0, 255);
  for (auto& b : v) b = (uint8_t)d(rng);
}

// Dequantizes through the quantized_matmul kernel (float32 activations), which is
// the path inference uses; mx::dequantize rounds in float16 and is only approximate.
static mx::array dequant_exact(const mx::array& W, const mx::array& S, const std::optional<mx::array>& B, int group, int bits,
                               const char* mode, int n) {
  auto I = mx::eye(n, n, 0, mx::float32);
  auto Wt = mx::quantized_matmul(I, W, S, B, true, group, bits, mode); // [n, rows]
  auto out = mx::transpose(Wt);
  mx::eval(out);
  return out;
}

static double compare(const std::vector<float>& ref, const mx::array& deq_in) {
  auto deq = mx::contiguous(deq_in);
  mx::eval(deq);
  double max_err = 0;
  const float* d = deq.data<float>();
  for (size_t i = 0; i < ref.size(); i++) max_err = std::max(max_err, (double)fabsf(d[i] - ref[i]));
  return max_err;
}

static void test_q8_0() {
  const int n = 256, rows = 3;
  std::vector<uint8_t> src(rows * (n / 32) * 34);
  fill_random(src);
  for (int r = 0; r < rows; r++)
    for (int b = 0; b < n / 32; b++) *(uint16_t*)&src[(r * (n / 32) + b) * 34] = rand_f16_scale();
  std::vector<float> ref(rows * n);
  for (int r = 0; r < rows; r++) quant::dequantize_row(gguf::TensorType::Q8_0, &src[r * (n / 32) * 34], n, &ref[r * n]);
  std::vector<uint8_t> w(rows * quant::packed_row_bytes(8, n)), s(rows * (n / 32) * 2), b(rows * (n / 32) * 2);
  for (int r = 0; r < rows; r++)
    quant::repack_q8_0_row(&src[r * (n / 32) * 34], n, &w[r * quant::packed_row_bytes(8, n)], &s[r * (n / 32) * 2], &b[r * (n / 32) * 2]);
  auto W = array_from_bytes(w.data(), {rows, (int)quant::packed_row_bytes(8, n) / 4}, mx::uint32);
  auto S = array_from_bytes(s.data(), {rows, n / 32}, mx::float16);
  auto B = array_from_bytes(b.data(), {rows, n / 32}, mx::float16);
  auto deq = dequant_exact(W, S, B, 32, 8, "affine", n);
  double err = compare(ref, deq);
  printf("q8_0 max err %.3g\n", err);
  CHECK(err < 1e-4, "q8_0 repack not exact");
}

static void test_q4_0() {
  const int n = 128, rows = 2;
  std::vector<uint8_t> src(rows * (n / 32) * 18);
  fill_random(src);
  for (int r = 0; r < rows; r++)
    for (int b = 0; b < n / 32; b++) *(uint16_t*)&src[(r * (n / 32) + b) * 18] = rand_f16_scale();
  std::vector<float> ref(rows * n);
  for (int r = 0; r < rows; r++) quant::dequantize_row(gguf::TensorType::Q4_0, &src[r * (n / 32) * 18], n, &ref[r * n]);
  size_t wr = quant::packed_row_bytes(4, n);
  std::vector<uint8_t> w(rows * wr, 0), s(rows * (n / 32) * 2), b(rows * (n / 32) * 2);
  for (int r = 0; r < rows; r++)
    quant::repack_q4_0_row(&src[r * (n / 32) * 18], n, &w[r * wr], &s[r * (n / 32) * 2], &b[r * (n / 32) * 2]);
  auto W = array_from_bytes(w.data(), {rows, (int)wr / 4}, mx::uint32);
  auto S = array_from_bytes(s.data(), {rows, n / 32}, mx::float16);
  auto B = array_from_bytes(b.data(), {rows, n / 32}, mx::float16);
  auto deq = dequant_exact(W, S, B, 32, 4, "affine", n);
  double err = compare(ref, deq);
  printf("q4_0 max err %.3g\n", err);
  CHECK(err < 1e-5, "q4_0 repack not exact");
}

static void test_q4_k() {
  const int n = 512, rows = 2;
  std::vector<uint8_t> src(rows * (n / 256) * 144);
  fill_random(src);
  for (int r = 0; r < rows; r++)
    for (int b = 0; b < n / 256; b++) {
      uint8_t* blk = &src[(r * (n / 256) + b) * 144];
      *(uint16_t*)blk = rand_f16_scale();
      *(uint16_t*)(blk + 2) = rand_f16_scale();
    }
  std::vector<float> ref(rows * n);
  for (int r = 0; r < rows; r++) quant::dequantize_row(gguf::TensorType::Q4_K, &src[r * (n / 256) * 144], n, &ref[r * n]);
  size_t wr = quant::packed_row_bytes(4, n);
  std::vector<uint8_t> w(rows * wr, 0), s(rows * (n / 32) * 2), b(rows * (n / 32) * 2);
  for (int r = 0; r < rows; r++)
    quant::repack_q4_k_row(&src[r * (n / 256) * 144], n, &w[r * wr], &s[r * (n / 32) * 2], &b[r * (n / 32) * 2]);
  auto W = array_from_bytes(w.data(), {rows, (int)wr / 4}, mx::uint32);
  auto S = array_from_bytes(s.data(), {rows, n / 32}, mx::float16);
  auto B = array_from_bytes(b.data(), {rows, n / 32}, mx::float16);
  auto deq = dequant_exact(W, S, B, 32, 4, "affine", n);
  double err = compare(ref, deq);
  double max_ref = 0;
  for (float v : ref) max_ref = std::max(max_ref, (double)fabsf(v));
  printf("q4_k max err %.3g (max |ref| %.3g)\n", err, max_ref);
  CHECK(err <= 2e-3 * max_ref, "q4_k repack error beyond f16 rounding");
}

static void test_mxfp4() {
  const int n = 256, rows = 2;
  std::vector<uint8_t> src(rows * (n / 32) * 17);
  fill_random(src);
  std::uniform_int_distribution<int> e(110, 135);
  for (int r = 0; r < rows; r++)
    for (int b = 0; b < n / 32; b++) src[(r * (n / 32) + b) * 17] = (uint8_t)e(rng);
  std::vector<float> ref(rows * n);
  for (int r = 0; r < rows; r++) quant::dequantize_row(gguf::TensorType::MXFP4, &src[r * (n / 32) * 17], n, &ref[r * n]);
  size_t wr = quant::packed_row_bytes(4, n);
  std::vector<uint8_t> w(rows * wr, 0), s(rows * (n / 32));
  for (int r = 0; r < rows; r++) quant::repack_mxfp4_row(&src[r * (n / 32) * 17], n, &w[r * wr], &s[r * (n / 32)]);
  auto W = array_from_bytes(w.data(), {rows, (int)wr / 4}, mx::uint32);
  auto S = array_from_bytes(s.data(), {rows, n / 32}, mx::uint8);
  auto deq = dequant_exact(W, S, std::nullopt, 32, 4, "mxfp4", n);
  double err = compare(ref, deq);
  printf("mxfp4 max err %.3g\n", err);
  CHECK(err < 1e-6, "mxfp4 repack not exact");
}

static void test_q6_k_lossy() {
  // Q6_K has no exact mapping; check that dequantize + MLX 8-bit requantization is close.
  const int n = 256;
  std::vector<uint8_t> src(210);
  fill_random(src);
  *(uint16_t*)&src[208] = rand_f16_scale();
  std::vector<float> ref(n);
  quant::dequantize_row(gguf::TensorType::Q6_K, src.data(), n, ref.data());
  mx::array a(ref.data(), {1, n}, mx::float32);
  auto parts = mx::quantize(a, 32, 8, "affine");
  auto deq = dequant_exact(parts[0], parts[1], parts[2], 32, 8, "affine", n);
  double err = compare(ref, deq), max_ref = 0;
  for (float v : ref) max_ref = std::max(max_ref, (double)fabsf(v));
  printf("q6_k -> affine8 max err %.3g (max |ref| %.3g)\n", err, max_ref);
  CHECK(err <= 0.01 * max_ref, "q6_k requantization too lossy");
}

int main() {
  test_q8_0();
  test_q4_0();
  test_q4_k();
  test_mxfp4();
  test_q6_k_lossy();
  printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
  return failures ? 1 : 0;
}
