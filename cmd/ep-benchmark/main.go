// ep-benchmark compares BirdNET-Go inference backends on one model, using
// BirdNET-Go's own inference package so every backend goes through the same
// code paths the app uses: ONNX Runtime CPU, ONNX Runtime CUDA (local patch,
// BIRDNET_ORT_EP), OpenVINO CPU and OpenVINO GPU.
//
// It feeds real audio (raw little-endian float32 mono at the model's rate),
// measures model load time, per-inference latency and process CPU time per
// inference, and checks each backend's scores against ONNX Runtime CPU.
// Results are printed as a table and written as JSON lines.
//
// Lives in the birdnet-go tree as cmd/ep-benchmark (internal packages).
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/tphakala/birdnet-go/internal/inference"
)

type result struct {
	Model       string  `json:"model"`
	Backend     string  `json:"backend"`
	Device      string  `json:"device"`
	LoadMs      float64 `json:"load_ms"`
	Iters       int     `json:"iters"`
	MeanMs      float64 `json:"mean_ms"`
	P50Ms       float64 `json:"p50_ms"`
	P95Ms       float64 `json:"p95_ms"`
	P99Ms       float64 `json:"p99_ms"`
	MaxMs       float64 `json:"max_ms"`
	CPUMsPerInf float64 `json:"cpu_ms_per_inference"`
	RTF         float64 `json:"real_time_factor"`
	Windows     int     `json:"compare_windows"`
	MaxAbsDiff  float64 `json:"max_abs_score_diff"`
	Top1Agree   float64 `json:"top1_agreement"`
	Top5Overlap float64 `json:"top5_overlap"`
	Error       string  `json:"error,omitempty"`
}

func main() {
	var (
		name     = flag.String("name", "", "model display name")
		model    = flag.String("model", "", "ONNX model path")
		labels   = flag.String("labels", "", "labels file (one per line)")
		audio    = flag.String("audio", "", "raw float32 LE mono audio at the model rate")
		rate     = flag.Int("rate", 48000, "model sample rate")
		seconds  = flag.Float64("seconds", 3, "model clip length in seconds")
		backends = flag.String("backends", "ort-cpu,ort-cuda,ov-cpu,ov-gpu", "backends to run")
		warmup   = flag.Int("warmup", 20, "warmup inferences")
		iters    = flag.Int("iters", 200, "timed inferences")
		threads  = flag.Int("threads", 0, "CPU threads for ORT (0 = all)")
		ovLib    = flag.String("ov-lib", "/usr/lib/libopenvino_c.so", "libopenvino_c path")
		ortLib   = flag.String("ort-lib", "", "libonnxruntime path (auto if empty)")
		out      = flag.String("out", "", "append JSON lines here")
		ovOutIdx = flag.Int("ov-output-index", 0, "OpenVINO logits output port (3 for Perch v2)")
	)
	flag.Parse()

	lbl, err := readLines(*labels)
	must(err)
	pcm, err := readF32(*audio)
	must(err)
	clip := int(float64(*rate) * *seconds)
	windows := splitWindows(pcm, clip, *rate) // 1 s hop across the whole file
	if len(windows) == 0 {
		must(fmt.Errorf("audio shorter than one %d-sample window", clip))
	}
	must(inference.InitONNXRuntime(*ortLib))

	var ref [][]float32 // ORT CPU scores per window, the reference
	var results []result
	for _, b := range strings.Split(*backends, ",") {
		r := result{Model: *name, Backend: b, Iters: *iters, Windows: len(windows)}
		c, dev, loadDur, err := newClassifier(b, *model, lbl, *threads, *ovLib, *ovOutIdx)
		r.LoadMs = ms(loadDur)
		r.Device = dev
		if err != nil {
			r.Error = err.Error()
			results = append(results, r)
			report(r, *out)
			continue
		}
		// Scores for every window (accuracy), using the backend under test.
		scores := make([][]float32, len(windows))
		for i, w := range windows {
			logits, perr := c.Predict(w)
			if perr != nil {
				r.Error = "predict: " + perr.Error()
				break
			}
			scores[i] = sigmoid(logits)
		}
		if r.Error == "" {
			if b == "ort-cpu" {
				ref = scores
			}
			if ref != nil {
				r.MaxAbsDiff, r.Top1Agree, r.Top5Overlap = compare(ref, scores)
			}
			in := windows[len(windows)/2]
			for range *warmup {
				_, _ = c.Predict(in)
			}
			lat := make([]time.Duration, 0, *iters)
			cpu0 := cpuTime()
			for range *iters {
				t := time.Now()
				if _, perr := c.Predict(in); perr != nil {
					r.Error = "predict: " + perr.Error()
					break
				}
				lat = append(lat, time.Since(t))
			}
			cpuUsed := cpuTime() - cpu0
			if len(lat) > 0 {
				fill(&r, lat, cpuUsed, *seconds)
			}
		}
		c.Close()
		results = append(results, r)
		report(r, *out)
	}
	printTable(results)
}

func newClassifier(backend, model string, labels []string, threads int, ovLib string, ovOutIdx int) (inference.Classifier, string, time.Duration, error) {
	t := time.Now()
	switch backend {
	case "ort-cpu", "ort-cuda", "ort-tensorrt":
		ep := strings.TrimPrefix(backend, "ort-")
		must(os.Setenv("BIRDNET_ORT_EP", ep))
		c, err := inference.NewONNXClassifier(model, inference.ONNXClassifierOptions{Labels: labels, Threads: threads})
		if err != nil {
			return nil, "", time.Since(t), err
		}
		dev := inference.ORTDevice(c)
		if ep != "cpu" && dev != "GPU" {
			c.Close()
			return nil, dev, time.Since(t), fmt.Errorf("%s requested but the session fell back to CPU", ep)
		}
		return c, dev, time.Since(t), nil
	case "ov-cpu", "ov-gpu", "ov-gpu-f16":
		if err := inference.InitOpenVINO(ovLib); err != nil {
			return nil, "", time.Since(t), err
		}
		dev, prec := inference.OVDeviceCPU, inference.OVPrecisionF32
		if strings.HasPrefix(backend, "ov-gpu") {
			dev = inference.OVDeviceGPU
		}
		if backend == "ov-gpu-f16" {
			prec = "" // OpenVINO default (f16)
		}
		c, err := inference.NewOpenVINOClassifier(model, inference.OpenVINOClassifierOptions{
			Labels: labels, Device: dev, PrecisionHint: prec, OutputIndex: ovOutIdx,
		})
		return c, dev, time.Since(t), err
	}
	return nil, "", 0, fmt.Errorf("unknown backend %q", backend)
}

func fill(r *result, lat []time.Duration, cpu time.Duration, clipSec float64) {
	s := slices.Clone(lat)
	slices.Sort(s)
	var sum time.Duration
	for _, d := range s {
		sum += d
	}
	pct := func(p float64) float64 { return ms(s[min(len(s)-1, int(math.Ceil(p*float64(len(s))))-1)]) }
	r.Iters = len(s)
	r.MeanMs = ms(sum) / float64(len(s))
	r.P50Ms, r.P95Ms, r.P99Ms, r.MaxMs = pct(0.50), pct(0.95), pct(0.99), ms(s[len(s)-1])
	r.CPUMsPerInf = ms(cpu) / float64(len(s))
	r.RTF = r.MeanMs / (clipSec * 1000)
}

// compare returns the max |score diff|, top-1 agreement and mean top-5 overlap.
func compare(ref, got [][]float32) (maxDiff, top1, top5 float64) {
	n := 0
	for i := range ref {
		if got[i] == nil || len(got[i]) != len(ref[i]) {
			continue
		}
		n++
		for j := range ref[i] {
			maxDiff = math.Max(maxDiff, math.Abs(float64(ref[i][j]-got[i][j])))
		}
		a, b := topK(ref[i], 5), topK(got[i], 5)
		if a[0] == b[0] {
			top1++
		}
		hit := 0
		for _, x := range a {
			if slices.Contains(b, x) {
				hit++
			}
		}
		top5 += float64(hit) / 5
	}
	if n > 0 {
		top1 /= float64(n)
		top5 /= float64(n)
	}
	return maxDiff, top1, top5
}

func topK(v []float32, k int) []int {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int {
		switch {
		case v[a] > v[b]:
			return -1
		case v[a] < v[b]:
			return 1
		}
		return a - b
	})
	return idx[:k]
}

func sigmoid(l []float32) []float32 {
	o := make([]float32, len(l))
	for i, x := range l {
		o[i] = float32(1 / (1 + math.Exp(-float64(x))))
	}
	return o
}

func splitWindows(pcm []float32, clip, hop int) [][]float32 {
	var w [][]float32
	for off := 0; off+clip <= len(pcm); off += hop {
		w = append(w, pcm[off:off+clip])
	}
	return w
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func readF32(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out, nil
}

func readLines(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, l := range strings.Split(strings.TrimRight(string(b), "\n"), "\n") {
		out = append(out, strings.TrimRight(l, "\r"))
	}
	return out, nil
}

func report(r result, path string) {
	if path == "" {
		return
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	must(err)
	defer f.Close()
	must(json.NewEncoder(f).Encode(r))
}

func printTable(rs []result) {
	fmt.Printf("\n%-14s %-12s %-4s %8s %8s %8s %8s %8s %9s %8s %7s %7s\n",
		"model", "backend", "dev", "load_ms", "mean_ms", "p50", "p95", "max", "cpu_ms/i", "maxdiff", "top1", "top5")
	for _, r := range rs {
		if r.Error != "" {
			fmt.Printf("%-14s %-12s %-4s ERROR: %s\n", r.Model, r.Backend, r.Device, r.Error)
			continue
		}
		fmt.Printf("%-14s %-12s %-4s %8.0f %8.2f %8.2f %8.2f %8.2f %9.2f %8.5f %7.2f %7.2f\n",
			r.Model, r.Backend, r.Device, r.LoadMs, r.MeanMs, r.P50Ms, r.P95Ms, r.MaxMs,
			r.CPUMsPerInf, r.MaxAbsDiff, r.Top1Agree, r.Top5Overlap)
	}
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
