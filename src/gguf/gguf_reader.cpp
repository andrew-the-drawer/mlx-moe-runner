#include "gguf/gguf_reader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace mmr::gguf {

const char* tensor_type_name(TensorType t) {
  switch (t) {
    case TensorType::F32: return "F32";
    case TensorType::F16: return "F16";
    case TensorType::Q4_0: return "Q4_0";
    case TensorType::Q4_1: return "Q4_1";
    case TensorType::Q5_0: return "Q5_0";
    case TensorType::Q5_1: return "Q5_1";
    case TensorType::Q8_0: return "Q8_0";
    case TensorType::Q8_1: return "Q8_1";
    case TensorType::Q2_K: return "Q2_K";
    case TensorType::Q3_K: return "Q3_K";
    case TensorType::Q4_K: return "Q4_K";
    case TensorType::Q5_K: return "Q5_K";
    case TensorType::Q6_K: return "Q6_K";
    case TensorType::Q8_K: return "Q8_K";
    case TensorType::BF16: return "BF16";
    case TensorType::MXFP4: return "MXFP4";
  }
  return "UNKNOWN";
}

void tensor_type_block(TensorType t, size_t* block_bytes, size_t* block_elems) {
  switch (t) {
    case TensorType::F32: *block_bytes = 4; *block_elems = 1; return;
    case TensorType::F16: *block_bytes = 2; *block_elems = 1; return;
    case TensorType::BF16: *block_bytes = 2; *block_elems = 1; return;
    case TensorType::Q4_0: *block_bytes = 18; *block_elems = 32; return;
    case TensorType::Q4_1: *block_bytes = 20; *block_elems = 32; return;
    case TensorType::Q5_0: *block_bytes = 22; *block_elems = 32; return;
    case TensorType::Q5_1: *block_bytes = 24; *block_elems = 32; return;
    case TensorType::Q8_0: *block_bytes = 34; *block_elems = 32; return;
    case TensorType::Q8_1: *block_bytes = 36; *block_elems = 32; return;
    case TensorType::Q2_K: *block_bytes = 84; *block_elems = 256; return;
    case TensorType::Q3_K: *block_bytes = 110; *block_elems = 256; return;
    case TensorType::Q4_K: *block_bytes = 144; *block_elems = 256; return;
    case TensorType::Q5_K: *block_bytes = 176; *block_elems = 256; return;
    case TensorType::Q6_K: *block_bytes = 210; *block_elems = 256; return;
    case TensorType::Q8_K: *block_bytes = 292; *block_elems = 256; return;
    case TensorType::MXFP4: *block_bytes = 17; *block_elems = 32; return;
  }
  *block_bytes = 0;
  *block_elems = 0;
}

int64_t Value::as_int(int64_t def) const {
  if (type == ValueType::STRING || type == ValueType::ARRAY) return def;
  return (int64_t)num;
}

double Value::as_float(double def) const {
  if (type == ValueType::STRING || type == ValueType::ARRAY) return def;
  return num;
}

uint64_t TensorInfo::n_elements() const {
  uint64_t n = 1;
  for (auto d : dims) n *= d;
  return n;
}

namespace {

struct Cursor {
  const uint8_t* p;
  const uint8_t* end;
  template <typename T>
  T read() {
    if (p + sizeof(T) > end) throw std::runtime_error("GGUF: truncated file");
    T v;
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string read_string() {
    uint64_t len = read<uint64_t>();
    if (p + len > end) throw std::runtime_error("GGUF: truncated string");
    std::string s((const char*)p, len);
    p += len;
    return s;
  }
  double read_scalar(ValueType t) {
    switch (t) {
      case ValueType::UINT8: return read<uint8_t>();
      case ValueType::INT8: return read<int8_t>();
      case ValueType::UINT16: return read<uint16_t>();
      case ValueType::INT16: return read<int16_t>();
      case ValueType::UINT32: return read<uint32_t>();
      case ValueType::INT32: return read<int32_t>();
      case ValueType::FLOAT32: return read<float>();
      case ValueType::BOOL: return read<uint8_t>() != 0;
      case ValueType::UINT64: return (double)read<uint64_t>();
      case ValueType::INT64: return (double)read<int64_t>();
      case ValueType::FLOAT64: return read<double>();
      default: throw std::runtime_error("GGUF: not a scalar type");
    }
  }
};

} // namespace

Reader::Reader(const std::string& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("GGUF: cannot open " + path);
  struct stat st;
  if (fstat(fd_, &st) != 0) throw std::runtime_error("GGUF: cannot stat " + path);
  size_ = (uint64_t)st.st_size;
  void* m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (m == MAP_FAILED) throw std::runtime_error("GGUF: mmap failed for " + path);
  map_ = (const uint8_t*)m;

  Cursor c{map_, map_ + size_};
  uint32_t magic = c.read<uint32_t>();
  if (magic != 0x46554747u) throw std::runtime_error("GGUF: bad magic in " + path);
  version_ = c.read<uint32_t>();
  if (version_ < 2 || version_ > 3) throw std::runtime_error("GGUF: unsupported version " + std::to_string(version_));
  uint64_t n_tensors = c.read<uint64_t>();
  uint64_t n_kv = c.read<uint64_t>();

  for (uint64_t i = 0; i < n_kv; i++) {
    std::string key = c.read_string();
    Value v;
    v.type = (ValueType)c.read<uint32_t>();
    if (v.type == ValueType::STRING) {
      v.str = c.read_string();
    } else if (v.type == ValueType::ARRAY) {
      v.arr_type = (ValueType)c.read<uint32_t>();
      uint64_t n = c.read<uint64_t>();
      if (v.arr_type == ValueType::STRING) {
        v.arr_str.reserve(n);
        for (uint64_t j = 0; j < n; j++) v.arr_str.push_back(c.read_string());
      } else if (v.arr_type == ValueType::ARRAY) {
        throw std::runtime_error("GGUF: nested arrays unsupported (" + key + ")");
      } else {
        v.arr_num.reserve(n);
        for (uint64_t j = 0; j < n; j++) v.arr_num.push_back(c.read_scalar(v.arr_type));
      }
    } else {
      v.num = c.read_scalar(v.type);
    }
    kv_[key] = std::move(v);
  }

  tensors_.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors; i++) {
    TensorInfo t;
    t.name = c.read_string();
    uint32_t nd = c.read<uint32_t>();
    if (nd > 8) throw std::runtime_error("GGUF: bad n_dims for " + t.name);
    for (uint32_t d = 0; d < nd; d++) t.dims.push_back(c.read<uint64_t>());
    t.type = (TensorType)c.read<uint32_t>();
    t.offset = c.read<uint64_t>();
    tensors_.push_back(std::move(t));
  }

  uint64_t alignment = (uint64_t)get_int("general.alignment", 32);
  uint64_t data_start = (uint64_t)(c.p - map_);
  data_start = (data_start + alignment - 1) / alignment * alignment;

  for (size_t i = 0; i < tensors_.size(); i++) {
    auto& t = tensors_[i];
    size_t bb = 0, be = 0;
    tensor_type_block(t.type, &bb, &be);
    if (bb == 0) {
      t.n_bytes = 0;
    } else {
      uint64_t ne0 = t.ne(0);
      if (ne0 % be != 0) throw std::runtime_error("GGUF: row length not a block multiple: " + t.name);
      uint64_t rows = t.n_elements() / ne0;
      t.n_bytes = rows * (ne0 / be) * bb;
    }
    if (data_start + t.offset + t.n_bytes > size_)
      throw std::runtime_error("GGUF: tensor beyond end of file: " + t.name);
    t.data = map_ + data_start + t.offset;
    tensor_index_[t.name] = i;
  }
}

Reader::~Reader() {
  if (map_) munmap((void*)map_, size_);
  if (fd_ >= 0) ::close(fd_);
}

const Value* Reader::get(const std::string& key) const {
  auto it = kv_.find(key);
  return it == kv_.end() ? nullptr : &it->second;
}

int64_t Reader::get_int(const std::string& key, int64_t def) const {
  auto v = get(key);
  return v ? v->as_int(def) : def;
}

double Reader::get_float(const std::string& key, double def) const {
  auto v = get(key);
  return v ? v->as_float(def) : def;
}

std::string Reader::get_string(const std::string& key, const std::string& def) const {
  auto v = get(key);
  return (v && v->type == ValueType::STRING) ? v->str : def;
}

const TensorInfo* Reader::tensor(const std::string& name) const {
  auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

} // namespace mmr::gguf
