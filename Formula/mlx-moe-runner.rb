class MlxMoeRunner < Formula
  desc "Run 100B-class Mixture-of-Experts LLMs from SSD on Apple silicon with MLX"
  homepage "https://github.com/lantrungseo/mlx-moe-runner"
  url "https://github.com/lantrungseo/mlx-moe-runner/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # updated by scripts/release.sh
  license "MIT"
  head "https://github.com/lantrungseo/mlx-moe-runner.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "cpp-httplib" => :build
  depends_on "nlohmann-json" => :build
  depends_on "pkgconf" => :build
  depends_on arch: :arm64
  depends_on macos: :sonoma
  depends_on "mlx"
  depends_on "pcre2"

  def install
    args = %W[
      -DMMR_BUILD_TESTS=OFF
      -DCMAKE_PREFIX_PATH=#{Formula["mlx"].opt_prefix}
      -DNLOHMANN_JSON_INCLUDE_DIR=#{Formula["nlohmann-json"].opt_include}
      -DHTTPLIB_INCLUDE_DIR=#{Formula["cpp-httplib"].opt_include}
    ]
    system "cmake", "-S", ".", "-B", "build", *args, *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  def caveats
    <<~EOS
      Convert a GGUF MoE model once, then run or serve it:
        mlx-moe-runner convert gpt-oss-120b-MXFP4.gguf ~/models/gptoss120b.mlxmoe
        mlx-moe-runner serve ~/models/gptoss120b.mlxmoe --port 8080
      The converted directory streams experts from SSD; keep it on the internal drive.
    EOS
  end

  test do
    assert_match "mlx-moe-runner", shell_output("#{bin}/mlx-moe-runner --version")
    assert_match "convert", shell_output("#{bin}/mlx-moe-runner help 2>&1", 0)
  end
end
