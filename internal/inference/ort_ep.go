package inference

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/tphakala/birdnet-go/internal/logger"
	ortlib "github.com/yalue/onnxruntime_go"
)

// ONNX Runtime execution-provider selection for the classifier sessions.
//
// BIRDNET_ORT_EP picks the provider: "" or "cpu" (default, upstream behaviour),
// "cuda" (NVIDIA CUDA EP) or "tensorrt" (TensorRT EP with CUDA as the fallback).
// BIRDNET_ORT_DEVICE_ID selects the GPU (default "0"). The GPU providers need the
// GPU build of libonnxruntime plus the CUDA/cuDNN runtime libraries on the
// library path. If a provider cannot be attached the session stays on the CPU EP
// and a warning is logged, so a broken GPU never stops a model from loading.
const (
	ORTProviderCPU      = "cpu"
	ORTProviderCUDA     = "cuda"
	ORTProviderTensorRT = "tensorrt"

	envORTProvider = "BIRDNET_ORT_EP"
	envORTDeviceID = "BIRDNET_ORT_DEVICE_ID"
	// envORTCUDAOptions passes extra CUDA EP options as "key=value,key=value"
	// (e.g. "cudnn_conv_algo_search=HEURISTIC").
	envORTCUDAOptions = "BIRDNET_ORT_CUDA_OPTIONS"
)

// cudaOptions returns the CUDA EP options: device_id plus BIRDNET_ORT_CUDA_OPTIONS.
func cudaOptions() map[string]string {
	opts := map[string]string{"device_id": ortDeviceID()}
	for _, kv := range strings.Split(os.Getenv(envORTCUDAOptions), ",") {
		if k, v, ok := strings.Cut(strings.TrimSpace(kv), "="); ok && k != "" {
			opts[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
	}
	return opts
}

// RequestedORTProvider returns the provider requested via BIRDNET_ORT_EP.
func RequestedORTProvider() string {
	switch v := strings.ToLower(strings.TrimSpace(os.Getenv(envORTProvider))); v {
	case ORTProviderCUDA, ORTProviderTensorRT:
		return v
	default:
		return ORTProviderCPU
	}
}

func ortDeviceID() string {
	if v := strings.TrimSpace(os.Getenv(envORTDeviceID)); v != "" {
		return v
	}
	return "0"
}

// appendORTProvider attaches the requested GPU provider to so and returns the
// provider actually bound. On any error it returns ORTProviderCPU and the error;
// the session options are then left on the default CPU EP.
func appendORTProvider(so *ortlib.SessionOptions) (string, error) {
	provider := RequestedORTProvider()
	if provider == ORTProviderCPU {
		return ORTProviderCPU, nil
	}
	if provider == ORTProviderTensorRT {
		trt, err := ortlib.NewTensorRTProviderOptions()
		if err != nil {
			return ORTProviderCPU, err
		}
		defer func() { _ = trt.Destroy() }()
		if err := trt.Update(map[string]string{"device_id": ortDeviceID()}); err != nil {
			return ORTProviderCPU, err
		}
		if err := so.AppendExecutionProviderTensorRT(trt); err != nil {
			return ORTProviderCPU, err
		}
	}
	cuda, err := ortlib.NewCUDAProviderOptions()
	if err != nil {
		return ORTProviderCPU, err
	}
	defer func() { _ = cuda.Destroy() }()
	if err := cuda.Update(cudaOptions()); err != nil {
		return ORTProviderCPU, err
	}
	if err := so.AppendExecutionProviderCUDA(cuda); err != nil {
		return ORTProviderCPU, err
	}
	return provider, nil
}

// logORTProvider records which provider a classifier session ended up on.
func logORTProvider(modelPath, provider string, err error) {
	log := logger.Global().Module("inference")
	if err != nil {
		log.Warn("ONNX Runtime GPU execution provider unavailable, using CPU",
			logger.String("requested", RequestedORTProvider()),
			logger.String("model", modelPath),
			logger.Error(err))
		return
	}
	if provider != ORTProviderCPU {
		log.Info("ONNX Runtime execution provider attached",
			logger.String("provider", provider),
			logger.String("device_id", ortDeviceID()),
			logger.String("model", modelPath))
	}
}

// ExecutionProviderReporter is implemented by ONNX Runtime backed classifiers.
type ExecutionProviderReporter interface {
	ExecutionProvider() string
}

// ORTDevice maps a classifier's bound ONNX Runtime provider to the runtime
// device string shown in the UI/API: "GPU" for CUDA/TensorRT, else "CPU".
func ORTDevice(c Classifier) string {
	if r, ok := c.(ExecutionProviderReporter); ok {
		switch r.ExecutionProvider() {
		case ORTProviderCUDA, ORTProviderTensorRT:
			return "GPU"
		}
	}
	return "CPU"
}

// gpuSessions counts live ONNX Runtime classifier sessions bound to a GPU
// execution provider, so the status API can report CUDA as active.
var gpuSessions atomic.Int64

// CUDAStatusInfo describes the ONNX Runtime GPU execution provider for the UI.
type CUDAStatusInfo struct {
	// Requested reports that BIRDNET_ORT_EP asks for a GPU provider.
	Requested bool
	// Provider is the requested provider ("cuda" or "tensorrt"), empty for CPU.
	Provider string
	// Active reports that at least one classifier session runs on the GPU.
	Active bool
	// Sessions is the number of classifier sessions bound to the GPU.
	Sessions int
	// Devices lists the NVIDIA GPU models reported by the driver.
	Devices []string
}

// CUDAStatus returns the current ONNX Runtime GPU execution-provider status.
func CUDAStatus() CUDAStatusInfo {
	s := CUDAStatusInfo{Sessions: int(gpuSessions.Load())}
	if p := RequestedORTProvider(); p != ORTProviderCPU {
		s.Requested = true
		s.Provider = p
	}
	s.Active = s.Sessions > 0
	if s.Requested {
		s.Devices = NVIDIAGPUModels()
	}
	return s
}

// NVIDIAGPUModels returns the GPU model names from /proc/driver/nvidia, which
// the NVIDIA container toolkit maps into the container. Empty when unavailable.
func NVIDIAGPUModels() []string {
	files, _ := filepath.Glob("/proc/driver/nvidia/gpus/*/information")
	sort.Strings(files)
	var models []string
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			if k, v, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(k) == "Model" {
				if m := strings.TrimSpace(v); m != "" {
					models = append(models, m)
				}
				break
			}
		}
	}
	return models
}
