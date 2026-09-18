#pragma once
// ggml block formats: dequantization to float and exact repacking into the
// MLX quantized layouts (affine: w = scale * q + bias; mxfp4: e8m0 * e2m1).
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gguf/gguf_reader.h"

namespace mmr::quant {

using mmr::gguf::TensorType;

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
float bf16_to_f32(uint16_t b);

// Whether we can dequantize this type at all.
bool can_dequantize(TensorType t);
// Dequantize one row of n elements (n must be a multiple of the block size).
void dequantize_row(TensorType t, const uint8_t* src, size_t n, float* dst);

// Description of an MLX quantized tensor layout.
struct MlxQuant {
  const char* mode = "affine"; // "affine" or "mxfp4"
  int bits = 4;
  int group = 32;
  bool has_biases = true;      // affine only
};

// Row byte sizes for packed weights (uint32 words), scales and biases.
size_t packed_row_bytes(int bits, size_t n);        // n * bits / 8, rounded to uint32
size_t scales_row_bytes(const MlxQuant& q, size_t n); // f16 for affine, u8 for mxfp4

// Exact repackers (one row). Outputs are byte buffers in the MLX layouts.
// Q8_0  -> affine 8-bit, group 32 (q_u8 = q_s8 + 128, scale = d, bias = -128 d)
void repack_q8_0_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales, uint8_t* biases);
// Q4_0  -> affine 4-bit, group 32 (scale = d, bias = -8 d)
void repack_q4_0_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales, uint8_t* biases);
// Q4_K  -> affine 4-bit, group 32 (scale = d * sc, bias = -dmin * m), lossless up to f16 rounding of d*sc
void repack_q4_k_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales, uint8_t* biases);
// MXFP4 -> mxfp4, group 32 (nibble reorder only)
void repack_mxfp4_row(const uint8_t* src, size_t n, uint8_t* w, uint8_t* scales);

// Which MLX layout an exact repack of this type produces; returns false if
// the type has no exact mapping and needs dequantize + requantize.
bool exact_mlx_quant(TensorType t, MlxQuant* out);

} // namespace mmr::quant
