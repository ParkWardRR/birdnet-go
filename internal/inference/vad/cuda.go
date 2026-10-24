package vad

import (
	ort "github.com/yalue/onnxruntime_go"

	"github.com/tphakala/birdnet-go/internal/inference"
	"github.com/tphakala/birdnet-go/internal/logger"
)

// newSessionForProvider builds the session on cfg's execution provider. On
// CUDA it falls back to the CPU with a warning instead of failing like the
// species classifiers do: the VAD gates the privacy filter, which must keep
// working when the GPU is unavailable. Silero VAD is small, so the GPU gain is
// modest (measured on a Quadro P1000: 1.38 ms vs 1.80 ms per one-second flush,
// identical probabilities), but it also takes that work off the CPU.
func newSessionForProvider(cfg *Config) (*session, error) {
	if cfg.Provider != inference.ExecutionProviderCUDA {
		return newSession(cfg.ModelPath, cfg.ModelData, nil)
	}
	log := logger.Global().Module("inference")
	device := inference.CUDADeviceLabel(cfg.CUDADeviceID)

	var regErr error
	s, err := newSession(cfg.ModelPath, cfg.ModelData, func(so *ort.SessionOptions) {
		regErr = inference.AppendCUDAProvider(so, cfg.CUDADeviceID)
	})
	switch {
	case err == nil && regErr == nil:
		log.Info("privacy VAD running on the CUDA execution provider",
			logger.String("device", device))
		return s, nil
	case err == nil:
		// Registration failed, so ORT built the session on the CPU provider alone.
		log.Warn("privacy VAD could not use CUDA; running on CPU",
			logger.String("device", device), logger.Error(regErr))
		return s, nil
	default:
		log.Warn("privacy VAD could not use CUDA; running on CPU",
			logger.String("device", device), logger.Error(err))
		return newSession(cfg.ModelPath, cfg.ModelData, nil)
	}
}
