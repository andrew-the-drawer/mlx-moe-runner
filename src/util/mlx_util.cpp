#include "util/mlx_util.h"

#include <stdexcept>

namespace mmr {

mx::Dtype dtype_from_name(const std::string& s) {
  if (s == "uint32") return mx::uint32;
  if (s == "uint8") return mx::uint8;
  if (s == "int32") return mx::int32;
  if (s == "float16") return mx::float16;
  if (s == "bfloat16") return mx::bfloat16;
  if (s == "float32") return mx::float32;
  throw std::runtime_error("unknown dtype " + s);
}

std::string dtype_to_name(mx::Dtype d) {
  if (d == mx::uint32) return "uint32";
  if (d == mx::uint8) return "uint8";
  if (d == mx::int32) return "int32";
  if (d == mx::float16) return "float16";
  if (d == mx::bfloat16) return "bfloat16";
  if (d == mx::float32) return "float32";
  throw std::runtime_error("unsupported dtype");
}

mx::array array_from_bytes(const void* p, const mx::Shape& shape, mx::Dtype dtype) {
  if (dtype == mx::uint32) return mx::array((const uint32_t*)p, shape, dtype);
  if (dtype == mx::uint8) return mx::array((const uint8_t*)p, shape, dtype);
  if (dtype == mx::int32) return mx::array((const int32_t*)p, shape, dtype);
  if (dtype == mx::float16) return mx::array((const mx::float16_t*)p, shape, dtype);
  if (dtype == mx::bfloat16) return mx::array((const mx::bfloat16_t*)p, shape, dtype);
  if (dtype == mx::float32) return mx::array((const float*)p, shape, dtype);
  throw std::runtime_error("array_from_bytes: unsupported dtype");
}

mx::array array_from_buffer(mx::allocator::Buffer buf, const mx::Shape& shape, mx::Dtype dtype) {
  return mx::array(buf, shape, dtype);
}

} // namespace mmr
