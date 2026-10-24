# BirdNET-Go on NVIDIA Pascal GPUs (CUDA)

This branch (`pascal-cuda`, based on upstream tag `20261014`) runs all of BirdNET-Go's ONNX classifiers (BirdNET v2.4, Google Perch v2, BirdNET v3.0) on an NVIDIA GPU through ONNX Runtime's CUDA execution provider. It was built and tested on a **Quadro P1000 (Pascal, sm_61, 4 GB)**, which upstream BirdNET-Go cannot use.

**Unofficial.** Not affiliated with or supported by the BirdNET-Go project. Upstream's supported GPU path is OpenVINO on Intel graphics.

## Results (Quadro P1000 vs Intel UHD 630 vs CPU)

i5-8600T (6C/6T), 200 timed runs after 20 warm-up, same model files and the same real recording. "P1000" is this branch; "iGPU" is upstream OpenVINO on the UHD 630.

| Model (clip) | ORT CPU, 6 threads | OpenVINO CPU | iGPU (OpenVINO) | **P1000 (ORT CUDA)** |
|---|---:|---:|---:|---:|
| BirdNET v2.4 (3 s) | 13.0 ms | 7.2 ms | 10.7 ms | **7.7 ms** |
| Perch v2 (5 s) | 31.8 ms | 39.5 ms | 63.8 ms | **24.6 ms** |
| BirdNET v3.0 fp32 (5 s) | 46.5 ms | 43.3 ms | 125.5 ms | **31.2 ms** |
| BirdNET v3.0 **fp16** (5 s) | 47.9 ms | 49.9 ms | 124.3 ms | 179.9 ms ✗ |

- **Accuracy:** with fp32 models, scores on the P1000 match ONNX Runtime CPU within 1e-5, with the same top-1 and top-5 species in every window.
- **Avoid fp16 models on Pascal.** BirdNET v3.0 fp16 on the P1000 is slower than the CPU and disagreed on the top-1 species in 64% of windows. Install the gallery's `fp32` variants.
- **CPU cost:** CPU time per inference on the P1000 is roughly its latency (the CUDA driver waits on the GPU). That is still 4–9× less than CPU inference: 8 / 25 / 31 ms vs 47 / 145 / 225 ms.
- **Live:** with 4 camera streams and all three models, host CPU was 1.5% busy on the P1000 vs 3.5% on the iGPU. Peak GPU temperature under a 200 s full-load test was 77 °C with no throttling.

### CUDA options tested (no gain)

| `BIRDNET_ORT_CUDA_OPTIONS` | Effect |
|---|---|
| `cudnn_conv_algo_search=HEURISTIC` | same as the default (EXHAUSTIVE) |
| `prefer_nhwc=1` | slower (v3.0: 52 ms vs 31 ms) |
| `cudnn_conv_algo_search=DEFAULT` | about 60× slower |
| `enable_cuda_graph=1` | **don't**: wrong scores on v2.4 (replays stale inputs), and Perch/v3.0 fail to load (not every node is on CUDA) |

## What the patches change

| Commit | Change |
|---|---|
| `feat(inference): ONNX Runtime CUDA/TensorRT execution provider (BIRDNET_ORT_EP) with UI status` | `BIRDNET_ORT_EP=cuda` (or `tensorrt`) attaches the GPU execution provider to every ONNX classifier session. If attaching fails it logs a warning and stays on CPU, so a missing GPU never stops a model from loading. `BIRDNET_ORT_DEVICE_ID` picks the GPU; `BIRDNET_ORT_CUDA_OPTIONS=key=value,…` passes extra CUDA provider options. Models report `device=GPU`. The status API gains `backends.cuda`; **System → Inference** shows a "CUDA (ONNX Runtime)" row, and the hardware card stops saying NVIDIA cards "will not be used" while CUDA is active (it also names them from `/proc/driver/nvidia`). |
| `feat: ep-benchmark backend comparison tool` | `cmd/ep-benchmark`: runs one model on ORT CPU / ORT CUDA / OpenVINO CPU / OpenVINO GPU through BirdNET-Go's own inference package, on real audio. Reports load time, latency percentiles, CPU time per inference, and score agreement with ORT CPU. Writes JSON lines. |
| `fix(system): skip single-file bind mounts in the disk list` | The NVIDIA container toolkit bind-mounts single files (`/usr/bin/nvidia-*`, driver libraries); the System page listed them as disks. |

The BirdNET range-filter geomodel stays on CPU.

## Why Pascal needs a custom build

| Blocker | Workaround |
|---|---|
| Official ONNX Runtime GPU releases contain kernels for **sm_70+** only. On a P1000: `no kernel image is available for execution on the device` | `build-ort-sm61.sh`: ONNX Runtime 1.25.1 (the version this BirdNET-Go release pins) built from source with `CMAKE_CUDA_ARCHITECTURES=61`. About 80 min on 5 cores |
| cuDNN **≥ 9.5** has no Pascal kernels | cuDNN **9.1.0.70** (cuda12 build) |
| nvcc 12.4 can't compile the abseil that ORT 1.25 bundles | Build with the CUDA **12.9** toolkit, which still targets sm_61 |
| CUDA 13 and NVIDIA drivers 590+ dropped Pascal | Host driver **580** branch (pin it so `dnf`/`apt` can't upgrade it); CUDA 12.9 runtime in the image |

## Build

On a Linux host with Podman and the NVIDIA driver:

```sh
# 1. BirdNET-Go from this branch (upstream Dockerfile, ~10 min)
git clone -b pascal-cuda https://github.com/ParkWardRR/birdnet-go.git && cd birdnet-go
podman build --build-arg BUILD_VERSION=20261014-cudaep -t localhost/birdnet-go:cudaep-base .

# 2. ONNX Runtime 1.25.1 for sm_61 (~80 min). Needs cuDNN 9.1 headers/libs.
cd contrib/pascal-cuda
mkdir -p ort-build && cp build-ort-sm61.sh ort-build/
sh fetch-cuda-libs.sh          # CUDA 12.9.1 runtime + cuDNN 9.1.0.70 -> ./cuda, sha256-checked
podman run --rm --cpus 5 -v "$PWD/ort-build":/work:Z \
  -v "$PWD/dl/cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz":/cudnn.tar.xz:ro,z \
  docker.io/nvidia/cuda:12.9.1-devel-ubuntu22.04 sh /work/build-ort-sm61.sh
mkdir -p ort && cp -a ort-build/out/. ort/

# 3. GPU image
podman build -f Containerfile --build-arg BASE=localhost/birdnet-go:cudaep-base -t localhost/birdnet-go:p1000 .
```

To target a different pre-Volta GPU, change `CMAKE_CUDA_ARCHITECTURES` (e.g. `60` for P100, `52` for Maxwell). Untested.

## Run

1. Host: NVIDIA driver (580 for Pascal), NVIDIA Container Toolkit, CDI spec (`nvidia-cdi-refresh.service` regenerates `/var/run/cdi/nvidia.yaml` at boot). Don't keep a hand-made copy in `/etc/cdi`: DRM node numbers can change between boots and a stale spec maps the wrong device. On SELinux hosts: `setsebool -P container_use_devices on` (otherwise `NVML: Insufficient Permissions`).
2. Container: `--device nvidia.com/gpu=all -e BIRDNET_ORT_EP=cuda`. Example Quadlet: `birdnet-go.container`.
3. `config.yaml`: `birdnet.backend: onnx` (keeps OpenVINO out), and install the **fp32** gallery variants (v2.4 `fp32-dfttrunc`, Perch `no-dft-fp32@<region>`, v3.0 `fp32@<region>`).
4. Check: the log shows `ONNX Runtime execution provider attached provider=cuda` per model; `curl -s localhost:8080/api/v2/system/inference | jq '.backends.cuda, (.models[] | {id, backend, device})'`.

Give a container one GPU type. With both the Intel render node and the NVIDIA device in one container, OpenVINO's iGPU path ran about 3.5× slower in testing.

## Benchmark

`bench.sh` runs `cmd/ep-benchmark` for each model on every backend (build it with `go build -tags noembed,skipfrontend,openvino ./cmd/ep-benchmark` against the OpenVINO headers). Stop the live container first for clean numbers.
