# Pascal (compute capability 6.x, e.g. Quadro P1000 / GTX 10xx) variant of the
# BirdNET-Go CUDA image.
#
# The standard CUDA image (./Dockerfile with BASE_IMAGE=runtime-base-cuda) also
# runs on a Quadro P1000 today: the official ONNX Runtime GPU build carries PTX
# the driver JIT-compiles for sm_61 when each model loads, and cuDNN 9.10.2
# reaches Pascal through its runtime-compiled engines. That depends on what the
# next ONNX Runtime / cuDNN bump still ships for Pascal, and the JIT costs 1-4 s
# per model load. This file pins a known-good Pascal stack instead, by layering
# two replacements on top of that image (nothing else changes):
#   - ONNX Runtime ${ONNXRUNTIME_VERSION} built from source for sm_61
#     (build-ort-sm61.sh; ~80 min on 5 cores), same version as the base so the
#     Go bindings' C API matches
#   - cuDNN 9.1.0.70 (cuda12 build), the last line verified on Pascal
# The CUDA 12.9 runtime libraries from the base image are kept: ONNX Runtime is
# compiled with the 12.9 toolkit, which still targets sm_61.
#
# Build:
#   podman build --build-arg BASE_IMAGE=runtime-base-cuda -t localhost/birdnet-go:cuda .
#   podman build -f Docker/cuda-sm61/Dockerfile --build-arg BASE=localhost/birdnet-go:cuda \
#     -t localhost/birdnet-go:cuda-sm61 .
# To skip the ONNX Runtime compile, pass a prebuilt artifact image that holds the
# libraries in /out (for example one produced by a previous build of the
# ort-sm61-build stage):  --build-arg ORT_SM61_FROM=localhost/ort-sm61:1.25.1
# Run: --device nvidia.com/gpu=all (or --gpus all) and birdnet.onnxprovider: cuda.
# Use FP32 model variants: Pascal has no fast FP16.

ARG BASE=localhost/birdnet-go:cuda
ARG ONNXRUNTIME_VERSION=1.25.1
ARG CUDA_DEVEL_IMAGE=docker.io/nvidia/cuda:12.9.1-devel-ubuntu22.04
ARG CUDNN_VERSION=9.1.0.70
ARG CUDNN_SHA256=fdc6d6f28217b82c400adfdae8f0364fa36c5fb6cb683933e8900a748e70b263
ARG ORT_SM61_FROM=ort-sm61-build
ARG ORT_BUILD_PARALLEL=4

# cuDNN 9.1 from NVIDIA's redist archive, checksum-verified.
FROM ${CUDA_DEVEL_IMAGE} AS cudnn
ARG CUDNN_VERSION
ARG CUDNN_SHA256
RUN set -eu; \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates xz-utils >/dev/null; \
    f="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz"; \
    curl -fsSL -o "/tmp/$f" "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/$f"; \
    echo "${CUDNN_SHA256}  /tmp/$f" | sha256sum -c -; \
    mkdir -p /opt/cudnn && tar -xJf "/tmp/$f" -C /opt/cudnn --strip-components=1; \
    find /opt/cudnn/lib -name "*.a" -delete; \
    rm -f "/tmp/$f"

# ONNX Runtime built for sm_61 against that cuDNN.
FROM ${CUDA_DEVEL_IMAGE} AS ort-sm61-build
ARG ONNXRUNTIME_VERSION
ARG ORT_BUILD_PARALLEL
COPY --from=cudnn /opt/cudnn /opt/cudnn
COPY Docker/cuda-sm61/build-ort-sm61.sh /usr/local/bin/build-ort-sm61.sh
RUN ORT_VERSION="${ONNXRUNTIME_VERSION}" PARALLEL="${ORT_BUILD_PARALLEL}" sh /usr/local/bin/build-ort-sm61.sh

FROM ${ORT_SM61_FROM} AS ort-sm61

FROM ${BASE}
ARG ONNXRUNTIME_VERSION
ARG CUDNN_VERSION
USER root
RUN rm -f /usr/lib/libonnxruntime*.so* /usr/local/cuda/lib64/libcudnn*
COPY --from=ort-sm61 /out/ /usr/lib/
COPY --from=cudnn /opt/cudnn/lib/ /usr/local/cuda/lib64/
COPY --from=cudnn /opt/cudnn/LICENSE /usr/share/doc/nvidia-cuda/cudnn-${CUDNN_VERSION}-LICENSE
# Fail the build if the CUDA provider can't resolve a library, or if it isn't the
# sm_61 build of the expected ONNX Runtime version.
RUN set -eu; \
    rm -rf /usr/local/cuda/lib64/stubs; \
    ldconfig; \
    missing=$(ldd /usr/lib/libonnxruntime_providers_cuda.so | grep "not found" || true); \
    if [ -n "${missing}" ]; then echo "Error: unresolved CUDA provider libraries:" >&2; echo "${missing}" >&2; exit 1; fi; \
    test -e "/usr/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}"; \
    test -e "/usr/local/cuda/lib64/libcudnn.so.${CUDNN_VERSION%.*}"
LABEL org.opencontainers.image.description="BirdNET-Go CUDA image for Pascal GPUs: ONNX Runtime ${ONNXRUNTIME_VERSION} built for sm_61, cuDNN ${CUDNN_VERSION}"
