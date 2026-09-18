cmake_minimum_required(VERSION 3.24)
project(mlx-moe-runner VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(MMR_BUILD_TESTS "Build unit tests" ON)

# --- MLX (C++ library, e.g. `brew install mlx`) ---------------------------
# MLXConfig.cmake exports an imported target whose link interface names
# absolute Xcode SDK framework paths that may not exist on a Command Line
# Tools only machine, so we link the dylib directly. libmlx.dylib already
# carries its Metal/Accelerate framework dependencies.
find_package(MLX CONFIG QUIET)
if(NOT MLX_FOUND)
  find_path(MLX_INCLUDE_DIRS mlx/mlx.h HINTS /opt/homebrew/opt/mlx/include /usr/local/opt/mlx/include)
  find_library(MLX_LIBRARY mlx HINTS /opt/homebrew/opt/mlx/lib /usr/local/opt/mlx/lib)
  if(NOT MLX_INCLUDE_DIRS OR NOT MLX_LIBRARY)
    message(FATAL_ERROR "MLX not found. Install it with `brew install mlx` or set CMAKE_PREFIX_PATH.")
  endif()
endif()
message(STATUS "MLX library: ${MLX_LIBRARY}")
message(STATUS "MLX include: ${MLX_INCLUDE_DIRS}")

# --- PCRE2 (tokenizer pre-tokenization regex) -----------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

# --- header-only deps -----------------------------------------------------
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
  HINTS /opt/homebrew/opt/nlohmann-json/include /usr/local/opt/nlohmann-json/include)
find_path(HTTPLIB_INCLUDE_DIR httplib.h
  HINTS /opt/homebrew/opt/cpp-httplib/include /usr/local/opt/cpp-httplib/include)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
  message(FATAL_ERROR "nlohmann/json.hpp not found (brew install nlohmann-json)")
endif()
if(NOT HTTPLIB_INCLUDE_DIR)
  message(FATAL_ERROR "httplib.h not found (brew install cpp-httplib)")
endif()

# --- core library ---------------------------------------------------------
add_library(mmr_core STATIC
  src/util/log.cpp
  src/util/sysinfo.cpp
  src/util/mlx_util.cpp
  src/gguf/gguf_reader.cpp
  src/gguf/ggml_quants.cpp
  src/model/spec.cpp
  src/model/layout.cpp
  src/model/weights.cpp
  src/convert/convert.cpp
  src/io/cold_loader.cpp
  src/io/expert_cache.cpp
  src/engine/router.cpp
  src/engine/lookahead.cpp
  src/engine/sampler.cpp
  src/engine/engine.cpp
  src/tokenizer/bpe_tokenizer.cpp
  src/tokenizer/chat.cpp
  src/server/server.cpp
)
target_include_directories(mmr_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${MLX_INCLUDE_DIRS}
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(mmr_core PUBLIC ${MLX_LIBRARY} PkgConfig::PCRE2 pthread)
target_compile_definitions(mmr_core PUBLIC MMR_VERSION="${PROJECT_VERSION}" CPPHTTPLIB_NO_EXCEPTIONS=0)
target_compile_options(mmr_core PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)

add_executable(mlx-moe-runner src/main.cpp)
target_link_libraries(mlx-moe-runner PRIVATE mmr_core)

install(TARGETS mlx-moe-runner RUNTIME DESTINATION bin)

if(MMR_BUILD_TESTS)
  enable_testing()
  foreach(t test_quants test_tokenizer test_cold_loader test_router test_layout)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE mmr_core)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()
