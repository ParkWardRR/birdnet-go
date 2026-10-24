package inference

// Support for NVIDIA GPUs older than Volta (compute capability < 7.0, e.g.
// Pascal: GTX 10xx, Quadro P-series). Official ONNX Runtime GPU releases only
// contain kernels for sm_70+, and cuDNN 9.5+ has no Pascal kernels, so on these
// cards the CUDA provider registers fine and then fails on the first inference
// with "no kernel image is available for execution on the device". The fix is a
// different image (Docker/cuda-sm61: ONNX Runtime built for sm_61 + cuDNN 9.1),
// not a code change; this file only makes the failure explain itself and warns
// about FP16 models, which Pascal runs slowly and with degraded accuracy.

import (
	"context"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tphakala/birdnet-go/internal/logger"
)

// cudaHintArch is the troubleshooting hint for a GPU older than the kernels in
// the ONNX Runtime / cuDNN build.
const cudaHintArch = "this GPU is older than the kernels in the ONNX Runtime/cuDNN build " +
	"(official ONNX Runtime GPU releases need compute capability 7.0+, cuDNN 9.5+ dropped Pascal): " +
	"for Pascal cards (GTX 10xx, Quadro P-series, compute capability 6.x) use the cuda-sm61 image " +
	"(Docker/cuda-sm61: ONNX Runtime built for sm_61 with cuDNN 9.1), or set birdnet.onnxprovider to cpu"

// isCUDAArchMismatch reports whether a lower-cased ONNX Runtime error says the
// build has no kernels for this GPU's architecture.
func isCUDAArchMismatch(m string) bool {
	return strings.Contains(m, "no kernel image is available") ||
		strings.Contains(m, "cudaerrornokernelimagefordevice") ||
		strings.Contains(m, "cudnn_status_arch_mismatch")
}

// fp16FastComputeCapability is the first compute capability with full-rate FP16
// arithmetic for inference (Volta). Below it FP16 runs at a fraction of FP32
// throughput, and in testing on a Quadro P1000 (6.1) the BirdNET v3.0 FP16
// model was ~6x slower than FP32 and disagreed with the CPU on the top species.
const fp16FastComputeCapability = 7.0

var (
	computeCapMu    sync.Mutex
	computeCapCache = map[int]float64{}
)

// cudaComputeCapability returns the compute capability of the given GPU as
// reported by nvidia-smi (0 when it cannot be determined). The NVIDIA Container
// Toolkit injects nvidia-smi with the "utility" driver capability, which the
// CUDA image requests. Results are cached per device.
func cudaComputeCapability(deviceID int) float64 {
	computeCapMu.Lock()
	defer computeCapMu.Unlock()
	if cc, ok := computeCapCache[deviceID]; ok {
		return cc
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=compute_cap",
		"--format=csv,noheader", "-i", strconv.Itoa(deviceID)).Output()
	cc := 0.0
	if err == nil {
		if v, perr := strconv.ParseFloat(strings.TrimSpace(string(out)), 64); perr == nil {
			cc = v
		}
	}
	computeCapCache[deviceID] = cc
	return cc
}

// warnFP16OnPreVolta logs a warning when an FP16 model is loaded on the CUDA
// provider of a GPU without fast FP16 (compute capability < 7.0).
func warnFP16OnPreVolta(modelPath string, deviceID int) {
	name := strings.ToLower(filepath.Base(modelPath))
	if !strings.Contains(name, "fp16") {
		return
	}
	cc := cudaComputeCapability(deviceID)
	if cc == 0 || cc >= fp16FastComputeCapability {
		return
	}
	logger.Global().Module("inference").Warn(
		"FP16 model on a GPU without fast FP16: expect slow inference and reduced accuracy; install the FP32 variant of this model",
		logger.String("model", filepath.Base(modelPath)),
		logger.String("device", CUDADeviceLabel(deviceID)),
		logger.String("compute_capability", strconv.FormatFloat(cc, 'f', 1, 64)))
}
