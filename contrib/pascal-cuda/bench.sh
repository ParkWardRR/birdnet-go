#!/bin/sh
# Compare backends with cmd/ep-benchmark (built from this branch, with -tags openvino).
#   MODELS=<BirdNET-Go models dir> BENCH=<dir with ep-benchmark, audio, labels> sh bench.sh <image> <out.jsonl>
# BENCH must contain: ep-benchmark, clip48k.f32 / clip32k.f32 (raw float32 mono at 48/32 kHz,
# e.g. ffmpeg -i clip.flac -ac 1 -ar 48000 -f f32le clip48k.f32), v24_labels.txt.
# IGPU is the host render node of the Intel GPU (optional). The iGPU and the NVIDIA GPU
# are benchmarked in separate containers: with both in one container OpenVINO's iGPU
# path ran ~3.5x slower in testing.
set -eu
IMG="$1"; OUT="$2"
M="${MODELS:?set MODELS}"; B="${BENCH:?set BENCH}"; IGPU_NODE="${IGPU:-}"
: > "$B/$OUT"
run() {
  dev="$1"; backends="$2"; shift 2
  # shellcheck disable=SC2086
  podman run --rm --entrypoint /b/ep-benchmark $dev --group-add keep-groups \
    --userns keep-id -v "$B":/b:Z -v "$M":/m:ro,z "$IMG" \
    -backends "$backends" -warmup 20 -iters 200 -out "/b/$OUT" "$@"
}
both() {
  [ -n "$IGPU_NODE" ] && run "--device $IGPU_NODE:/dev/dri/renderD128" ort-cpu,ov-cpu,ov-gpu "$@"
  run "--device nvidia.com/gpu=all" ort-cpu,ort-cuda "$@"
}
both -name "BirdNET v2.4" -model /m/birdnet-v2.4/BirdNET_v2.4_fp32_dfttrunc.onnx \
  -labels /b/v24_labels.txt -audio /b/clip48k.f32 -rate 48000 -seconds 3
both -name "Perch v2" -model /m/perch-v2/perch_v2_north-america-west_no_dft.onnx -ov-output-index 3 \
  -labels /m/perch-v2/perch_v2_north-america-west_labels.txt -audio /b/clip32k.f32 -rate 32000 -seconds 5
both -name "BirdNET v3.0 fp32" -model /m/birdnet-v3.0/birdnet_v3.0_north-america-west_fp32.onnx \
  -labels /m/birdnet-v3.0/birdnet_v3.0_north-america-west_labels.txt -audio /b/clip32k.f32 -rate 32000 -seconds 5
