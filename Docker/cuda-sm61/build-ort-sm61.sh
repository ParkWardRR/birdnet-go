#!/bin/sh
# Build ONNX Runtime with the CUDA execution provider for Pascal (sm_61).
# Runs inside nvidia/cuda:12.9.1-devel-ubuntu22.04 with cuDNN 9.1 at /opt/cudnn
# (see Dockerfile in this directory). Output: /out/libonnxruntime*.so*
#   - CUDA toolkit 12.9: nvcc 12.4 can't parse the abseil ONNX Runtime 1.25
#     bundles; 12.9 still targets sm_61 (deprecated, not removed)
#   - flash/memory-efficient/lean attention need sm_80+ and are disabled
set -eu
ORT_VERSION="${ORT_VERSION:-1.25.1}"
PARALLEL="${PARALLEL:-4}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends git ca-certificates python3 python3-pip >/dev/null
pip3 install -q 'cmake>=3.28' ninja packaging
mkdir -p /build && cd /build
git clone -q --depth 1 --branch "v${ORT_VERSION}" --recursive --shallow-submodules \
  https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel "${PARALLEL}" --nvcc_threads 1 \
  --use_cuda --cuda_home /usr/local/cuda --cudnn_home /opt/cudnn \
  --cmake_generator Ninja --skip_tests --allow_running_as_root --compile_no_warning_as_error \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=61 onnxruntime_BUILD_UNIT_TESTS=OFF \
    onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
    onnxruntime_USE_LEAN_ATTENTION=OFF "CMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets"
mkdir -p /out
cp -a build/Linux/Release/libonnxruntime.so* build/Linux/Release/libonnxruntime_providers_*.so /out/
rm -rf /build
ls -la /out
