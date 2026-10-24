package inference

import "testing"

func TestCUDAHintArchMismatch(t *testing.T) {
	// Error text ONNX Runtime 1.25.1 (official GPU build) produced on a Quadro P1000.
	cases := []string{
		"Non-zero status code returned while running Conv node. Name:'node_Conv_7' Status Message: " +
			"CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED ; err 209 no kernel image is available for execution on the device",
		"CUDA failure 209: cudaErrorNoKernelImageForDevice ; GPU=0",
		"CUDNN_STATUS_ARCH_MISMATCH",
	}
	for _, msg := range cases {
		if got := cudaHint(msg); got != cudaHintArch {
			t.Errorf("cudaHint(%q) = %q, want the architecture hint", msg, got)
		}
	}
	// Unrelated errors keep their existing hints.
	if got := cudaHint("libcudnn.so.9: cannot open shared object file"); got != cudaHintLibraries {
		t.Errorf("library error routed to %q", got)
	}
	if got := cudaHint("no CUDA-capable device is detected"); got != cudaHintDevice {
		t.Errorf("device error routed to %q", got)
	}
}

func TestWarnFP16OnPreVoltaSkipsFP32(t *testing.T) {
	// Must not shell out for non-FP16 models.
	computeCapMu.Lock()
	computeCapCache[99] = 6.1
	computeCapMu.Unlock()
	warnFP16OnPreVolta("/m/birdnet_v3.0_north-america-west_fp32.onnx", 99)
}
