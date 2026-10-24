package processor

import (
	"strconv"

	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/inference"
	"github.com/tphakala/birdnet-go/internal/inference/vad"
)

// vadCUDAKeySep separates the model-source cache key from the CUDA device the
// VAD session runs on, so switching birdnet.onnxprovider reloads the session.
const vadCUDAKeySep = "#cuda:"

// withVADProvider puts the VAD on the GPU the species classifiers use when
// birdnet.onnxprovider is "cuda", and returns the cache key extended with the
// device. Any other provider (or an invalid one) leaves the VAD on the CPU and
// the key unchanged. An empty key (no model) stays empty.
func withVADProvider(cfg *vad.Config, cacheKey string, bn *conf.BirdNETConfig) string {
	if cacheKey == "" {
		return ""
	}
	provider, err := inference.NormalizeExecutionProvider(bn.ONNXProvider)
	if err != nil || provider != inference.ExecutionProviderCUDA {
		return cacheKey
	}
	cfg.Provider = provider
	cfg.CUDADeviceID = bn.CUDADeviceID
	return cacheKey + vadCUDAKeySep + strconv.Itoa(bn.CUDADeviceID)
}
