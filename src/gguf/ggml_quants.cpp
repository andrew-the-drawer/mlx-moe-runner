#include "gguf/ggml_quants.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mmr::quant {

float f16_to_f32(uint16_t h) {
  _Float16 v;
  memcpy(&v, &h, 2);
  return (float)v;
}

uint16_t f32_to_f16(float f) {
  _Float16 v = (_Float16)f;
  uint16_t h;
  memcpy(&h, &v, 2);
  return h;
}

float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

bool can_dequantize(TensorType t) {
  switch (t) {
    case TensorType::F32: case TensorType::F16: case TensorType::BF16:
    case TensorType::Q4_0: case TensorType::Q4_1: case TensorType::Q5_0: case TensorType::Q5_1:
    case TensorType::Q8_0: case TensorType::Q4_K: case TensorType::Q5_K: case TensorType::Q6_K:
    case TensorType::MXFP4:
      return true;
    default:
      return false;
  }
}

namespace {

inline void get_scale_min_k4(int j, const uint8_t* scales, uint8_t* sc, uint8_t* m) {
  if (j < 4) {
    *sc = scales[j] & 63;
    *m = scales[j + 4] & 63;
  } else {
    *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

const float kMxfp4Values[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0, -0.5f, -1, -1.5f, -2, -3, -4, -6};

inline float e8m0_to_f32(uint8_t e) {
  if (e == 0) return ldexpf(1.0f, -127);
  if (e == 255) return NAN;
  return ldexpf(1.0f, (int)e - 127);
}

void dq_q4_0(const uint8_t* b, float* y) {
  float d = f16_to_f32(*(const uint16_t*)b);
  const uint8_t* qs = b + 2;
  for (int j = 0; j < 16; j++) {
    y[j] = ((int)(qs[j] & 0xF) - 8) * d;
    y[j + 16] = ((int)(qs[j] >> 4) - 8) * d;
  }
}

void dq_q4_1(const uint8_t* b, float* y) {
  float d = f16_to_f32(*(const uint16_t*)b);
  float m = f16_to_f32(*(const uint16_t*)(b + 2));
  const uint8_t* qs = b + 4;
  for (int j = 0; j < 16; j++) {
    y[j] = (qs[j] & 0xF) * d + m;
    y[j + 16] = (qs[j] >> 4) * d + m;
  }
}

void dq_q5_0(const uint8_t* b, float* y) {
  float d = f16_to_f32(*(const uint16_t*)b);
  uint32_t qh;
  memcpy(&qh, b + 2, 4);
  const uint8_t* qs = b + 6;
  for (int j = 0; j < 16; j++) {
    int xh0 = ((qh >> j) << 4) & 0x10;
    int xh1 = (qh >> (j + 12)) & 0x10;
    y[j] = (((qs[j] & 0xF) | xh0) - 16) * d;
    y[j + 16] = (((qs[j] >> 4) | xh1) - 16) * d;
  }
}

void dq_q5_1(const uint8_t* b, float* y) {
  float d = f16_to_f32(*(const uint16_t*)b);
  float m = f16_to_f32(*(const uint16_t*)(b + 2));
  uint32_t qh;
  memcpy(&qh, b + 4, 4);
  const uint8_t* qs = b + 8;
  for (int j = 0; j < 16; j++) {
    int xh0 = ((qh >> j) << 4) & 0x10;
    int xh1 = (qh >> (j + 12)) & 0x10;
    y[j] = ((qs[j] & 0xF) | xh0) * d + m;
    y[j + 16] = ((qs[j] >> 4) | xh1) * d + m;
  }
}

void dq_q8_0(const uint8_t* b, float* y) {
  float d = f16_to_f32(*(const uint16_t*)b);
  const int8_t* qs = (const int8_t*)(b + 2);
  for (int j = 0; j < 32; j++) y[j] = qs[j] * d;
}

void dq_q4_k(const uint8_t* b, float* y) {
  float d = f16_to_f32(*(const uint16_t*)b);
  float dmin = f16_to_f32(*(const uint16_t*)(b + 2));
  const uint8_t* scales = b + 4;
  const uint8_t* q = b + 16;
  int is = 0;
  for (int j = 0; j < 256; j += 64) {
    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales, &sc, &m);
    float d1 = d * sc, m1 = dmin * m;
    get_scale_min_k4(is + 1, scales, &sc, &m);
    float d2 = d * sc, m2 = dmin * m;
    for (int l = 0; l < 32; l++) y[j + l] = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * (q[l] >> 4) - m2;
    q += 32;
    is += 2;
  }
}

void dq_q5_k(const uint8_t* b, float* y) {
  float d = f16_to_f32(*(const uint16_t*)b);
  float dmin = f16_to_f32(*(const uint16_t*)(b + 2));
  const uint8_t* scales = b + 4;
  const uint8_t* qh = b + 16;
  const uint8_t* ql = b + 48;
  int is = 0;
  uint8_t u1 = 1, u2 = 2;
  for (int j = 0; j < 256; j += 64) {
    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales, &sc, &m);
    float d1 = d * sc, m1 = dmin * m;
    get_scale_min_k4(is + 1, scales, &sc, &m);
    float d2 = d * sc, m2 = dmin * m;
    for (int l = 0; l < 32; l++) y[j + l] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
    for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
    ql += 32;
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
  }
}

void dq_q6_k(const uint8_t* b, float* y) {
  const uint8_t* ql = b;
  const uint8_t* qh = b + 128;
  const int8_t* sc = (const int8_t*)(b + 192);
  float d = f16_to_f32(*(const uint16_t*)(b + 208));
  for (int n = 0; n < 256; n += 128) {
    for (int l = 0; l < 32; l++) {
      int is = l / 16;
      int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l + 0] = d * sc[is + 0] * q1;
      y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3;
      y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

void dq_mxfp4(const uint8_t* b, float* y) {
  float s = e8m0_to_f32(b[0]);
  const uint8_t* qs = b + 1;
  for (int j = 0; j < 16; j++) {
    y[j] = kMxfp4Values[qs[j] & 0xF] * s;
    y[j + 16] = kMxfp4Values[qs[j] >> 4] * s;
  }
}

} // namespace

void dequantize_row(TensorType t, const uint8_t* src, size_t n, float* dst) {
  size_t bb = 0, be = 0;
  gguf::tensor_type_block(t, &bb, &be);
  if (bb == 0 || n % be != 0) throw std::runtime_error("dequantize_row: bad type/size");
  size_t nb = n / be;
  switch (t) {
    case TensorType::F32: memcpy(dst, src, n * 4); return;
    case TensorType::F16:
      for (size_t i = 0; i < n; i++) dst[i] = f16_to_f32(((const uint16_t*)src)[i]);
      return;
    case TensorType::BF16:
      for (size_t i = 0; i < n; i++) dst[i] = bf16_to_f32(((const uint16_t*)src)[i]);
      return;
    case TensorType::Q4_0: for (size_t i = 0; i < nb; i++) dq_q4_0(src + i * bb, dst + i * be); return;
    case TensorType::Q4_1: for (size_t i = 0; i < nb; i++) dq_q4_1(src + i * bb, dst + i * be); return;
    case TensorType::Q5_0: for (size_t i = 0; i < nb; i++) dq_q5_0(src + i * bb, dst + i * be); return;
    case TensorType::Q5_1: for (size_t i = 0; i < nb; i++) dq_q5_1(src + i * bb, dst + i * be); return;
    case TensorType::Q8_0: for (size_t i = 0; i < nb; i++) dq_q8_0(src + i * bb, dst + i * be); return;
    case TensorType::Q4_K: for (size_t i = 0; i < nb; i++) dq_q4_k(src + i * bb, dst + i * be); return;
    case TensorType::Q5_K: for (size_t i = 0; i < nb; i++) dq_q5_k(src + i * bb, dst + i * be); return;
    case TensorType::Q6_K: for (size_t i = 0; i < nb; i++) dq_q6_k(src + i * bb, dst + i * be); return;
    case TensorType::MXFP4: for (size_t i = 0; i < nb; i++) dq_mxfp4(src + i * bb, dst + i * be); return;
    default: throw std::runtime_error(std::string("dequantize_row: unsupported type ") + gguf::tensor_type_name(t));
  }
}

size_t packed_row_bytes(int bits, size_t n) {
  return (n * bits + 31) / 32 * 4;
}

size_t scales_row_bytes(const MlxQuant& q, size_t n) {
  size_t groups = n / q.group;
  return strcmp(q.mode, "mxfp4") == 0 ? groups : groups * 2;
}

void repack_q8_0_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales, uint8_t* biases) {
  size_t nb = n / 32;
  for (size_t i = 0; i < nb; i++) {
    const uint8_t* b = src + i * 34;
    uint16_t d = *(const uint16_t*)b;
    float df = f16_to_f32(d);
    const int8_t* qs = (const int8_t*)(b + 2);
    uint8_t* out = w + i * 32;
    for (int j = 0; j < 32; j++) out[j] = (uint8_t)(qs[j] + 128);
    ((uint16_t*)scales)[i] = d;
    ((uint16_t*)biases)[i] = f32_to_f16(-128.0f * df);
  }
}

void repack_q4_0_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales, uint8_t* biases) {
  size_t nb = n / 32;
  for (size_t i = 0; i < nb; i++) {
    const uint8_t* b = src + i * 18;
    uint16_t d = *(const uint16_t*)b;
    float df = f16_to_f32(d);
    const uint8_t* qs = b + 2;
    uint8_t* out = w + i * 16; // 32 nibbles = 16 bytes; MLX nibble i = element i
    for (int j = 0; j < 16; j++) {
      uint8_t lo = qs[j] & 0xF;      // element j
      uint8_t hi = qs[j] >> 4;       // element j + 16
      // element j lives at byte j/2, nibble j%2
      if (j % 2 == 0) out[j / 2] = (out[j / 2] & 0xF0) | lo; else out[j / 2] = (out[j / 2] & 0x0F) | (lo << 4);
      int e = j + 16;
      if (e % 2 == 0) out[e / 2] = (out[e / 2] & 0xF0) | hi; else out[e / 2] = (out[e / 2] & 0x0F) | (hi << 4);
    }
    ((uint16_t*)scales)[i] = d;
    ((uint16_t*)biases)[i] = f32_to_f16(-8.0f * df);
  }
}

void repack_q4_k_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales, uint8_t* biases) {
  size_t nb = n / 256;
  for (size_t i = 0; i < nb; i++) {
    const uint8_t* b = src + i * 144;
    float d = f16_to_f32(*(const uint16_t*)b);
    float dmin = f16_to_f32(*(const uint16_t*)(b + 2));
    const uint8_t* sc8 = b + 4;
    const uint8_t* q = b + 16;
    uint8_t* out = w + i * 128;
    uint16_t* sc_out = (uint16_t*)scales + i * 8;
    uint16_t* bi_out = (uint16_t*)biases + i * 8;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc, m;
      get_scale_min_k4(is + 0, sc8, &sc, &m);
      sc_out[is] = f32_to_f16(d * sc);
      bi_out[is] = f32_to_f16(-dmin * m);
      get_scale_min_k4(is + 1, sc8, &sc, &m);
      sc_out[is + 1] = f32_to_f16(d * sc);
      bi_out[is + 1] = f32_to_f16(-dmin * m);
      // values j..j+31 are low nibbles of q[0..31], j+32..j+63 are high nibbles.
      uint8_t* o0 = out + j / 2;        // 32 elements = 16 bytes
      uint8_t* o1 = out + j / 2 + 16;
      for (int l = 0; l < 32; l += 2) {
        o0[l / 2] = (uint8_t)((q[l] & 0xF) | ((q[l + 1] & 0xF) << 4));
        o1[l / 2] = (uint8_t)((q[l] >> 4) | ((q[l + 1] >> 4) << 4));
      }
      q += 32;
      is += 2;
    }
  }
}

void repack_mxfp4_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales) {
  size_t nb = n / 32;
  for (size_t i = 0; i < nb; i++) {
    const uint8_t* b = src + i * 17;
    scales[i] = b[0];
    const uint8_t* qs = b + 1;
    uint8_t* out = w + i * 16;
    // element j (0..15) = low nibble of qs[j]; element j+16 = high nibble of qs[j].
    for (int j = 0; j < 16; j += 2) {
      out[j / 2] = (uint8_t)((qs[j] & 0xF) | ((qs[j + 1] & 0xF) << 4));
      out[8 + j / 2] = (uint8_t)((qs[j] >> 4) | ((qs[j + 1] >> 4) << 4));
    }
  }
}

bool exact_mlx_quant(TensorType t, MlxQuant* out) {
  switch (t) {
    case TensorType::Q8_0: *out = MlxQuant{"affine", 8, 32, true}; return true;
    case TensorType::Q4_0: *out = MlxQuant{"affine", 4, 32, true}; return true;
    case TensorType::Q4_K: *out = MlxQuant{"affine", 4, 32, true}; return true;
    case TensorType::MXFP4: *out = MlxQuant{"mxfp4", 4, 32, false}; return true;
    default: return false;
  }
}

} // namespace mmr::quant
