#pragma once
// Minimal read-only GGUF v2/v3 reader backed by mmap.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mmr::gguf {

enum class ValueType : uint32_t {
  UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5, FLOAT32 = 6,
  BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

// ggml tensor types we know how to interpret.
enum class TensorType : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
  Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
  BF16 = 30, MXFP4 = 39,
};

const char* tensor_type_name(TensorType t);
// Bytes per block and elements per block for a tensor type (0 if unknown).
void tensor_type_block(TensorType t, size_t* block_bytes, size_t* block_elems);

struct Value {
  ValueType type = ValueType::UINT8;
  // Scalars stored here; strings in str; arrays in arr (elements share arr_type).
  double num = 0;
  std::string str;
  ValueType arr_type = ValueType::UINT8;
  std::vector<double> arr_num;
  std::vector<std::string> arr_str;

  int64_t as_int(int64_t def = 0) const;
  double as_float(double def = 0) const;
  const std::string& as_string() const { return str; }
};

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> dims;    // GGUF order: dims[0] is the fastest-varying (row length)
  TensorType type;
  uint64_t offset;               // relative to data section
  const uint8_t* data = nullptr; // mapped pointer
  uint64_t n_bytes = 0;
  uint64_t n_elements() const;
  // Row-major view: [rows..., ne0]. ne0 = dims[0].
  uint64_t ne(int i) const { return i < (int)dims.size() ? dims[i] : 1; }
};

class Reader {
 public:
  explicit Reader(const std::string& path);
  ~Reader();
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  const std::map<std::string, Value>& metadata() const { return kv_; }
  const Value* get(const std::string& key) const;
  int64_t get_int(const std::string& key, int64_t def) const;
  double get_float(const std::string& key, double def) const;
  std::string get_string(const std::string& key, const std::string& def = "") const;
  bool has(const std::string& key) const { return kv_.count(key) > 0; }

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* tensor(const std::string& name) const;
  bool has_tensor(const std::string& name) const { return tensor_index_.count(name) > 0; }
  std::string architecture() const { return get_string("general.architecture"); }
  const std::string& path() const { return path_; }
  uint64_t file_size() const { return size_; }

 private:
  std::string path_;
  int fd_ = -1;
  const uint8_t* map_ = nullptr;
  uint64_t size_ = 0;
  uint32_t version_ = 0;
  std::map<std::string, Value> kv_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, size_t> tensor_index_;
};

} // namespace mmr::gguf
