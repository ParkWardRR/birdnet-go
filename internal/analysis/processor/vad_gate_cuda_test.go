package processor

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/tphakala/birdnet-go/internal/conf"
	"github.com/tphakala/birdnet-go/internal/inference"
	"github.com/tphakala/birdnet-go/internal/inference/vad"
)

func TestWithVADProvider(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		bn       conf.BirdNETConfig
		key      string
		wantKey  string
		wantProv string
		wantDev  int
	}{
		{name: "default provider stays on CPU", key: "embedded", wantKey: "embedded"},
		{name: "explicit cpu", bn: conf.BirdNETConfig{ONNXProvider: "cpu"}, key: "embedded", wantKey: "embedded"},
		{name: "cuda device 0", bn: conf.BirdNETConfig{ONNXProvider: "cuda"}, key: "embedded", wantKey: "embedded#cuda:0", wantProv: inference.ExecutionProviderCUDA},
		{name: "cuda device 1, path model", bn: conf.BirdNETConfig{ONNXProvider: " CUDA ", CUDADeviceID: 1}, key: "path:/m/silero.onnx", wantKey: "path:/m/silero.onnx#cuda:1", wantProv: inference.ExecutionProviderCUDA, wantDev: 1},
		{name: "invalid provider stays on CPU", bn: conf.BirdNETConfig{ONNXProvider: "rocm"}, key: "embedded", wantKey: "embedded"},
		{name: "no model stays empty", bn: conf.BirdNETConfig{ONNXProvider: "cuda"}, key: "", wantKey: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var cfg vad.Config
			assert.Equal(t, tt.wantKey, withVADProvider(&cfg, tt.key, &tt.bn))
			assert.Equal(t, tt.wantProv, cfg.Provider)
			assert.Equal(t, tt.wantDev, cfg.CUDADeviceID)
		})
	}
}

func TestVADSourceLabelIgnoresCUDADevice(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "embedded", vadSourceLabel("embedded#cuda:0"))
	assert.Equal(t, "path", vadSourceLabel("path:/m/silero.onnx#cuda:1"))
}
