#!/bin/sh
# Build ONNX Runtime with the CUDA EP for Pascal (sm_61, Quadro P1000).
# Official ORT GPU releases only ship sm_70+ kernels, and cuDNN >= 9.5 has no
# Pascal kernels, so this compiles ORT for sm_61 against cuDNN 9.1.0.70 (verified
# on the P1000). The toolkit is CUDA 12.9: nvcc 12.4 cannot parse the abseil that
# ORT 1.25 bundles; 12.9 still targets sm_61 (deprecated, not removed).
# Runs inside docker.io/nvidia/cuda:12.9.1-devel-ubuntu22.04 with the cuDNN 9.1
# redist archive mounted at /cudnn.tar.xz:
#   podman run --rm --cpus 5 -v "$PWD/ort-build":/work:Z \
#     -v <cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz>:/cudnn.tar.xz:ro,z \
#     docker.io/nvidia/cuda:12.9.1-devel-ubuntu22.04 sh /work/build-ort-sm61.sh
# Output: /work/out/libonnxruntime*.so*
set -eu
ORT_VERSION="${ORT_VERSION:-1.25.1}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends git ca-certificates python3 python3-pip xz-utils >/dev/null
pip3 install -q 'cmake>=3.28' ninja packaging
mkdir -p /opt/cudnn && tar -xJf /cudnn.tar.xz -C /opt/cudnn --strip-components=1
cd /work
[ -d onnxruntime ] || git clone -q --depth 1 --branch "v${ORT_VERSION}" --recursive --shallow-submodules \
  https://github.com/microsoft/onnxruntime.git
cd onnxruntime
rm -rf build
./build.sh --config Release --build_shared_lib --parallel 5 --nvcc_threads 1 \
  --use_cuda --cuda_home /usr/local/cuda --cudnn_home /opt/cudnn \
  --cmake_generator Ninja --skip_tests --allow_running_as_root --compile_no_warning_as_error \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=61 onnxruntime_BUILD_UNIT_TESTS=OFF \
    onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
    onnxruntime_USE_LEAN_ATTENTION=OFF "CMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets"
mkdir -p /work/out
cp -a build/Linux/Release/libonnxruntime.so* build/Linux/Release/libonnxruntime_providers_*.so /work/out/
ls -la /work/out
