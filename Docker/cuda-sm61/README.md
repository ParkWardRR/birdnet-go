# BirdNET-Go CUDA on Pascal GPUs (sm_61)

This branch is **[Steveiwonder/birdnet-go](https://github.com/Steveiwonder/birdnet-go)'s CUDA implementation**, merged with the upstream release, plus a small adaptation for NVIDIA GPUs older than Volta (compute capability 6.x: GTX 10xx, Quadro P-series). Tested on a Quadro P1000 (sm_61, 4 GB).

Unofficial; not affiliated with the BirdNET-Go project.

## Does Pascal need anything special?

**Measured on a Quadro P1000: the standard CUDA image works.** Its docs say compute capability 7.0+, but the official ONNX Runtime 1.25.1 GPU build carries PTX that the driver JIT-compiles for sm_61 when each model loads. cuDNN **9.10.2** (from the pinned `nvidia/cuda:12.9.1-cudnn-runtime` image) reaches Pascal through its runtime-compiled (NVRTC) engines.

That is luck rather than a guarantee: it depends on the next ONNX Runtime / cuDNN bump still shipping PTX and Pascal-capable engines. In earlier testing, a cuDNN **9.27** layer without nvJitLink failed on the first inference with `no kernel image is available for execution on the device`. So this branch adds an optional **`cuda-sm61`** image that pins a known-good Pascal stack.

| | Standard CUDA image | `cuda-sm61` variant |
|---|---|---|
| ONNX Runtime 1.25.1 | official GPU release (PTX JIT on Pascal) | built from source for sm_61 (native kernels) |
| cuDNN | 9.10.2 (whatever the pinned NVIDIA image ships) | **9.1.0.70**, pinned |
| CUDA runtime | 12.9 | 12.9 |
| Model load (v2.4 / Perch / v3.0) | 1.1 / 2.2 / 3.7 s | **0.6 / 0.9 / 1.4 s** |
| Inference (v2.4 / Perch / v3.0 FP32) | 8.6 / 24.8 / 31.3 ms | 7.7 / 24.7 / 31.2 ms |
| Accuracy vs CPU | identical top-1, max confidence diff ≤ 2e-6 | identical |
| Host driver | 580 branch (the last with Pascal; 590+ and CUDA 13 dropped it) | same |

## What this branch changes, on top of the two parents

| Change | Where | Why |
|---|---|---|
| Pascal image layer (optional) | `Docker/cuda-sm61/` (new) | native sm_61 ONNX Runtime + cuDNN 9.1 on top of the standard CUDA image; the main `Dockerfile` is untouched |
| Architecture hint | `internal/inference/cuda_pascal.go` + 1 line in `cudaHint` | a "no kernel image" failure says the build has no kernels for this GPU and points at the sm61 image, instead of the generic hint |
| FP16 warning | same file + 1 line in `NewONNXClassifier` | warns when an FP16 model loads on a card without fast FP16 (< 7.0). On the P1000, BirdNET v3.0 FP16 took 180 ms vs 31 ms FP32 and matched the CPU's top species in only 1 of 3 windows |
| Friendly stream names in notifications | cherry-pick of turmacar's `a48fb8f` | alerts say "Doorbell", not `rtsp_d13dfe45` |
| Disk list | skip single-file bind mounts | the NVIDIA toolkit's file mounts showed up as disks on the System page |

Everything else (the `birdnet.onnxprovider` setting, hot reload, the load-time operator placement probe, UI/API status, `cuda-benchmark`) is Steveiwonder's code, unchanged.

## Results (Quadro P1000, i5-8600T)

`birdnet-go cuda-benchmark`, 5 passes over a real recording, nothing else on the GPU:

| Model | CPU (6 threads) | P1000, `cuda-sm61` | Operators on GPU |
|---|---:|---:|---:|
| BirdNET v2.4 `fp32-dfttrunc` | 12.8 ms | **7.7 ms** | 246 / 276 |
| Perch v2 `no-dft-fp32` | 33–37 ms | **24.7 ms** | 725 / 725 |
| BirdNET v3.0 `fp32` | 47–56 ms | **31.2 ms** | 600 / 701 |
| BirdNET v3.0 `fp16` | — | 179 ms ✗ | 714 / 736 |

The CPU-side operators are small shape and arithmetic ops (Add, Concat, Gather, Reshape, …) that ONNX Runtime normally keeps on the host.

## Build

```sh
# 1. Standard CUDA image (Steveiwonder's stage, unchanged). With Podman, fully qualify the
#    NVIDIA image (Podman refuses Docker-Hub short names when it can't prompt):
IMG=$(sed -n 's/^ARG CUDA_RUNTIME_IMAGE=//p' Dockerfile)
podman build --build-arg BUILD_VERSION=$(git describe --tags) --build-arg BASE_IMAGE=runtime-base-cuda \
  --build-arg CUDA_RUNTIME_IMAGE=docker.io/$IMG -t localhost/birdnet-go:cuda .
# 2. Optional Pascal layer (the first build compiles ONNX Runtime: ~80 min on 5 cores)
podman build -f Docker/cuda-sm61/Dockerfile --build-arg BASE=localhost/birdnet-go:cuda \
  -t localhost/birdnet-go:cuda-sm61 .
```

To reuse a compiled ONNX Runtime, keep it as an image (`podman build -f Docker/cuda-sm61/Dockerfile --target ort-sm61-build -t localhost/ort-sm61:1.25.1 .`) and pass `--build-arg ORT_SM61_FROM=localhost/ort-sm61:1.25.1`; the layer then builds in ~3 min. Rebuild it only when the base `ONNXRUNTIME_VERSION` changes.

## Run

- Host: NVIDIA driver **580** (pin it), NVIDIA Container Toolkit with CDI; regenerate the spec each boot (`nvidia-cdi-refresh.service`), and don't keep a stale copy in `/etc/cdi`. SELinux + rootless Podman: `setsebool -P container_use_devices on`.
- Container: `--device nvidia.com/gpu=all`; config `birdnet.onnxprovider: cuda` (or `BIRDNET_ONNXPROVIDER=cuda`).
- Models: install the **FP32** gallery variants (v2.4 `fp32-dfttrunc`, Perch `no-dft-fp32@<region>`, v3.0 `fp32@<region>`).
- Check: the log shows `ONNX Runtime CUDA execution provider active ... cuda_nodes=N total_nodes=M` per model; `birdnet-go cuda-benchmark --model <file> --audio <wav>` compares CPU and CUDA.

## Keeping up with Steveiwonder and upstream

`sh Docker/cuda-sm61/sync-upstreams.sh [tag]` merges `Steveiwonder/main` and the newest (or given) upstream release tag into this branch. It doesn't rebase, so history is never rewritten. The Pascal changes touch two lines of Steveiwonder's files; everything else is in new files, so merges are normally conflict-free. After a sync, rebuild and rerun `cuda-benchmark`: a "no kernel image" error on the standard image means the bump dropped Pascal, and the `cuda-sm61` image is the fix.
