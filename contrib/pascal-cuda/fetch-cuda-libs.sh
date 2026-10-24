#!/bin/sh
# Download and unpack the CUDA 12.9.1 runtime libraries and cuDNN 9.1.0.70 (cuda12)
# from NVIDIA's redist server into ./cuda (layout expected by Containerfile).
# Every archive is checked against the sha256 in NVIDIA's redistrib manifest.
set -eu
B=https://developer.download.nvidia.com/compute/cuda/redist
C=https://developer.download.nvidia.com/compute/cudnn/redist
mkdir -p dl cuda/opt/cuda/lib cuda/etc/ld.so.conf.d
curl -fsS "$B/redistrib_12.9.1.json" > dl/cuda.json
curl -fsS "$C/redistrib_9.1.0.json" > dl/cudnn.json
{
  for c in cuda_cudart libcublas libcufft libcurand cuda_nvrtc; do
    jq -r --arg c "$c" --arg b "$B" '.[$c]["linux-x86_64"] | "\($b)/\(.relative_path) \(.sha256)"' dl/cuda.json
  done
  jq -r --arg b "$C" '.cudnn["linux-x86_64"] | to_entries[] | select(.key|test("cuda12")) | "\($b)/\(.value.relative_path) \(.value.sha256)"' dl/cudnn.json
} | while read -r url sum; do
  f="dl/$(basename "$url")"
  [ -f "$f" ] || curl -fsS -o "$f" "$url"
  echo "$sum  $f" | sha256sum -c --quiet
  tar -xJf "$f" -C cuda/opt/cuda/lib --strip-components=2 --wildcards '*/lib/lib*.so*'
done
rm -rf cuda/opt/cuda/lib/stubs
echo /opt/cuda/lib > cuda/etc/ld.so.conf.d/cuda.conf
ls cuda/opt/cuda/lib | sed 's/\.so.*//' | sort -u | tr '\n' ' '; echo
