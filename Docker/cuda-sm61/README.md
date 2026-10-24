# BirdNET-Go CUDA on Pascal GPUs (sm_61)

This branch is **[Steveiwonder/birdnet-go](https://github.com/Steveiwonder/birdnet-go)'s CUDA implementation**, merged with the upstream release, plus a small adaptation for NVIDIA GPUs older than Volta (compute capability 6.x: GTX 10xx, Quadro P-series). Tested on a Quadro P1000 (sm_61, 4 GB).

Unofficial; not affiliated with the BirdNET-Go project.

## Why Pascal needs this

Steveiwonder's CUDA image supports compute capability 7.0+. On a Pascal card the CUDA provider registers, then the first inference fails with `no kernel image is available for execution on the device`. The causes:

| Component | Standard CUDA image | Here |
|---|---|---|
| ONNX Runtime 1.25.1 GPU | official release: sm_70+ kernels only | built from source with `CMAKE_CUDA_ARCHITECTURES=61` |
| cuDNN | 9.x from `nvidia/cuda:12.9.1-cudnn-runtime` (9.5+ has no Pascal kernels) | **9.1.0.70** (cuda12), verified on the P1000 |
| CUDA runtime | 12.9 | 12.9 (unchanged) |
| Host driver | 525.60+ | **580 branch** (the last with Pascal; 590+ and CUDA 13 dropped it) |

## What this branch changes, on top of the two parents

| Change | Where | Why |
|---|---|---|
| Pascal image layer | `Docker/cuda-sm61/` (new) | swaps in the sm_61 ONNX Runtime + cuDNN 9.1 on the standard CUDA image; the main `Dockerfile` is untouched |
| Architecture hint | `internal/inference/cuda_pascal.go` + 1 line in `cudaHint` | the "no kernel image" failure now says the GPU is too old for this build and points at the sm61 image, instead of the generic hint |
| FP16 warning | same file + 1 line in `NewONNXClassifier` | warns when an FP16 model loads on a card without fast FP16 (< 7.0). On the P1000, BirdNET v3.0 FP16 was ~6× slower than FP32 and disagreed with the CPU on the top species in 64% of windows |
| Friendly stream names in notifications | cherry-pick of turmacar's `a48fb8f` | alerts say "Doorbell", not `rtsp_d13dfe45` |
| Disk list | skip single-file bind mounts | the NVIDIA toolkit's file mounts showed up as disks on the System page |

Everything else (the `birdnet.onnxprovider` setting, hot reload, the load-time operator placement probe, UI/API status, `cuda-benchmark`) is Steveiwonder's code, unchanged.

## Results (Quadro P1000, i5-8600T)

ms per inference, 200 runs, real recording. ORT CPU = 6 threads; iGPU = UHD 630 via upstream OpenVINO.

| Model | ORT CPU | iGPU (OpenVINO) | **P1000 (this image)** |
|---|---:|---:|---:|
| BirdNET v2.4 (3 s) | 13.0 | 10.7 | **7.7** |
| Perch v2 (5 s) | 31.8 | 63.8 | **24.6** |
| BirdNET v3.0 **FP32** (5 s) | 46.5 | 125.5 | **31.2** |
| BirdNET v3.0 FP16 (5 s) | 47.9 | 124.3 | 179.9 ✗ |

FP32 scores on the P1000 match ORT CPU (max diff < 1e-5, same top-1/top-5 species in every window). CUDA options that didn't help: `cudnn_conv_algo_search=HEURISTIC` (same), `prefer_nhwc` (slower), `DEFAULT` search (60× slower). Avoid `enable_cuda_graph` (wrong scores).

## Build

```sh
# 1. Standard CUDA image (Steveiwonder's stage, unchanged)
podman build --build-arg BUILD_VERSION=$(git describe --tags) --build-arg BASE_IMAGE=runtime-base-cuda \
  -t localhost/birdnet-go:cuda .
# 2. Pascal layer (first build compiles ONNX Runtime: ~80 min on 5 cores)
podman build -f Docker/cuda-sm61/Dockerfile --build-arg BASE=localhost/birdnet-go:cuda \
  -t localhost/birdnet-go:cuda-sm61 .
```

To reuse a compiled ONNX Runtime, keep it as an image (`podman build --target ort-sm61-build ... -t localhost/ort-sm61:1.25.1`) and pass `--build-arg ORT_SM61_FROM=localhost/ort-sm61:1.25.1`. Rebuild it only when the base `ONNXRUNTIME_VERSION` changes.

## Run

- Host: NVIDIA driver **580** (pin it), NVIDIA Container Toolkit with CDI; regenerate the spec each boot (`nvidia-cdi-refresh.service`), and don't keep a stale copy in `/etc/cdi`. SELinux + rootless Podman: `setsebool -P container_use_devices on`.
- Container: `--device nvidia.com/gpu=all`; config `birdnet.onnxprovider: cuda` (or `BIRDNET_ONNXPROVIDER=cuda`).
- Models: install the **FP32** gallery variants (v2.4 `fp32-dfttrunc`, Perch `no-dft-fp32@<region>`, v3.0 `fp32@<region>`).
- Check: the log shows `ONNX Runtime CUDA execution provider active ... cuda_nodes=N total_nodes=M` per model.

## Keeping up with Steveiwonder and upstream

`sh Docker/cuda-sm61/sync-upstreams.sh [tag]` merges `Steveiwonder/main` and the newest (or given) upstream release tag into this branch. It doesn't rebase, so this branch's history is never rewritten. The Pascal changes touch two lines of Steveiwonder's files; everything else is in new files, so merges are normally conflict-free.
