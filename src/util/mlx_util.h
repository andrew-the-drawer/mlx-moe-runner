#pragma once
#include <string>

#include <mlx/mlx.h>

namespace mmr {

namespace mx = mlx::core;

mx::Dtype dtype_from_name(const std::string& s);
std::string dtype_to_name(mx::Dtype d);

// Copies raw bytes (already in the target dtype's binary layout) into a new array.
mx::array array_from_bytes(const void* p, const mx::Shape& shape, mx::Dtype dtype);

// Wraps an MLX allocator buffer (no copy). The array owns the buffer.
mx::array array_from_buffer(mx::allocator::Buffer buf, const mx::Shape& shape, mx::Dtype dtype);

} // namespace mmr
