#!/bin/sh
# Bring this branch up to date with both of its parents, by merging (no rebase),
# so history is never rewritten and the Pascal commits stay on top:
#   steve    = https://github.com/Steveiwonder/birdnet-go  (CUDA implementation, main)
#   upstream = https://github.com/tphakala/birdnet-go       (releases, by tag)
# Usage: sh Docker/cuda-sm61/sync-upstreams.sh [upstream-tag]
# Then rebuild both images, run the CUDA benchmark, and push.
set -eu
git remote get-url steve >/dev/null 2>&1 || git remote add steve https://github.com/Steveiwonder/birdnet-go.git
git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/tphakala/birdnet-go.git
git fetch -q steve main
git fetch -q --tags upstream
TAG="${1:-$(git tag --list '20[0-9][0-9][0-9][0-9][0-9][0-9]' --sort=-v:refname | head -1)}"
echo "steve/main: $(git rev-list --count HEAD..steve/main) new commits; upstream $TAG: $(git rev-list --count HEAD.."$TAG") new commits"
git merge --no-ff -m "merge Steveiwonder/birdnet-go main" steve/main
git merge --no-ff -m "merge upstream $TAG" "$TAG"
# Files this branch adds or changes relative to the two parents (should stay small):
git diff --stat "$(git merge-base HEAD steve/main)" HEAD -- Docker/cuda-sm61 internal/inference/cuda_pascal.go \
  internal/inference/cuda_pascal_test.go internal/inference/execution_provider.go internal/inference/onnx.go | tail -1
