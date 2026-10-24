package inference

import ortlib "github.com/yalue/onnxruntime_go"

// AppendCUDAProvider registers the CUDA execution provider for deviceID on so,
// exactly as the species classifiers do. It is for ONNX sessions built outside
// this package, such as the privacy-filter VAD.
func AppendCUDAProvider(so *ortlib.SessionOptions, deviceID int) error {
	return appendCUDAProvider(so, deviceID)
}
