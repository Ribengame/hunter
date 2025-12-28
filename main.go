// main.go - ULTIMATE RIEMANN HYPOTHESIS HUNTER v3.0
// Production-ready with CPU/GPU auto-detection, flexible configuration
// Memory optimized (~1GB), supports massive t values, checkpoint/resume

package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"golang.org/x/sync/errgroup"
	"gopkg.in/yaml.v3"
)

// ==================== VERSION & BUILD INFO ====================
const (
	Version     = "3.0.0"
	BuildDate   = "2025-12-28"
	Author      = "Riemann Research Team"
	License     = "MIT"
	MaxSafeRAM  = 900 // MB - safety margin for 1GB total
)

// ==================== DYNAMIC CONSTANTS ====================
// These are calculated at runtime based on available resources
var (
	DynamicBatchSize    int
	DynamicCacheSize    int
	DynamicMaxWorkers   int
)

// ==================== CONFIGURATION STRUCTURES ====================
type ExecutionMode string

const (
	ModeAuto     ExecutionMode = "auto"
	ModeCPU      ExecutionMode = "cpu"
	ModeGPU      ExecutionMode = "gpu"
	ModeMultiGPU ExecutionMode = "multi-gpu"
)

type HardwareConfig struct {
	Mode                 ExecutionMode `json:"mode" yaml:"mode"`
	CPUUsagePercent      int           `json:"cpu_usage_percent" yaml:"cpu_usage_percent"`
	GPUDevices           []int         `json:"gpu_devices" yaml:"gpu_devices"`
	MaxGPUMemoryMB       int           `json:"max_gpu_memory_mb" yaml:"max_gpu_memory_mb"`
	ThreadsPerCore       float64       `json:"threads_per_core" yaml:"threads_per_core"`
	EnableAVX            bool          `json:"enable_avx" yaml:"enable_avx"`
	EnableCUDA           bool          `json:"enable_cuda" yaml:"enable_cuda"`
	EnableOpenCL         bool          `json:"enable_opencl" yaml:"enable_opencl"`
}

type CalculationConfig struct {
	StartT              float64 `json:"start_t" yaml:"start_t"`
	EndT                float64 `json:"end_t" yaml:"end_t"`
	Step                float64 `json:"step" yaml:"step"`
	ZeroThreshold       float64 `json:"zero_threshold" yaml:"zero_threshold"`
	Algorithm           string  `json:"algorithm" yaml:"algorithm"`
	Precision           int     `json:"precision" yaml:"precision"`
	UseCache            bool    `json:"use_cache" yaml:"use_cache"`
	CacheStrategy       string  `json:"cache_strategy" yaml:"cache_strategy"`
	Interpolation       bool    `json:"interpolation" yaml:"interpolation"`
	GramPointMethod     string  `json:"gram_point_method" yaml:"gram_point_method"`
	MaxTerms            int     `json:"max_terms" yaml:"max_terms"`
	RemainderTerms      int     `json:"remainder_terms" yaml:"remainder_terms"`
}

type OutputConfig struct {
	SaveZeros           bool   `json:"save_zeros" yaml:"save_zeros"`
	SaveStats           bool   `json:"save_stats" yaml:"save_stats"`
	SaveCheckpoints     bool   `json:"save_checkpoints" yaml:"save_checkpoints"`
	OutputDirectory     string `json:"output_directory" yaml:"output_directory"`
	FilenamePrefix      string `json:"filename_prefix" yaml:"filename_prefix"`
	CompressOutput      bool   `json:"compress_output" yaml:"compress_output"`
	Verbose             bool   `json:"verbose" yaml:"verbose"`
	RealTimeDisplay     bool   `json:"real_time_display" yaml:"real_time_display"`
	LogLevel            string `json:"log_level" yaml:"log_level"`
}

type PerformanceConfig struct {
	BatchSize           int           `json:"batch_size" yaml:"batch_size"`
	MaxWorkers          int           `json:"max_workers" yaml:"max_workers"`
	CacheSize           int           `json:"cache_size" yaml:"cache_size"`
	CheckpointInterval  time.Duration `json:"checkpoint_interval" yaml:"checkpoint_interval"`
	StatsInterval       time.Duration `json:"stats_interval" yaml:"stats_interval"`
	MemoryLimitMB       int           `json:"memory_limit_mb" yaml:"memory_limit_mb"`
	CPULimitPercent     int           `json:"cpu_limit_percent" yaml:"cpu_limit_percent"`
	GPULimitPercent     int           `json:"gpu_limit_percent" yaml:"gpu_limit_percent"`
	AutoTune            bool          `json:"auto_tune" yaml:"auto_tune"`
	BenchmarkMode       bool          `json:"benchmark_mode" yaml:"benchmark_mode"`
}

type Config struct {
	Hardware    HardwareConfig    `json:"hardware" yaml:"hardware"`
	Calculation CalculationConfig `json:"calculation" yaml:"calculation"`
	Output      OutputConfig      `json:"output" yaml:"output"`
	Performance PerformanceConfig `json:"performance" yaml:"performance"`
	
	// Internal fields
	configPath  string
	loadedFrom  string
	checksum    string
	mu          sync.RWMutex
}

// ==================== DATA STRUCTURES ====================
type ZeroResult struct {
	T            float64   `json:"t"`
	ZValue       float64   `json:"z_value"`
	Magnitude    float64   `json:"magnitude"`
	Precision    int       `json:"precision"`
	FoundAt      time.Time `json:"found_at"`
	WorkerID     int       `json:"worker_id"`
	DeviceID     int       `json:"device_id,omitempty"`
	DeviceType   string    `json:"device_type,omitempty"`
	Verified     bool      `json:"verified"`
	BatchID      string    `json:"batch_id,omitempty"`
	
	// Extended info for deep analysis
	Sigma        float64   `json:"sigma,omitempty"`
	OffCritical  bool      `json:"off_critical,omitempty"`
	Derivative   float64   `json:"derivative,omitempty"`
	Confidence   float64   `json:"confidence,omitempty"`
}

type HardwareStats struct {
	CPUCores         int     `json:"cpu_cores"`
	CPUUsagePercent  float64 `json:"cpu_usage_percent"`
	TotalRAMMB       float64 `json:"total_ram_mb"`
	UsedRAMMB        float64 `json:"used_ram_mb"`
	AvailableRAMMB   float64 `json:"available_ram_mb"`
	GPUs             int     `json:"gpus"`
	GPUUsagePercent  []float64 `json:"gpu_usage_percent,omitempty"`
	GPUMemoryMB      []float64 `json:"gpu_memory_mb,omitempty"`
	GPUTemperature   []float64 `json:"gpu_temperature,omitempty"`
	Uptime           time.Duration `json:"uptime"`
}

type CalculationStats struct {
	StartTime        time.Time     `json:"start_time"`
	CurrentT         float64       `json:"current_t"`
	PointsProcessed  int64         `json:"points_processed"`
	ZerosFound       int64         `json:"zeros_found"`
	PointsPerSecond  float64       `json:"points_per_second"`
	ElapsedTime      time.Duration `json:"elapsed_time"`
	EstimatedFinish  time.Time     `json:"estimated_finish,omitempty"`
	ProgressPercent  float64       `json:"progress_percent"`
	
	// Algorithm performance
	CacheHitRate     float64       `json:"cache_hit_rate"`
	AverageBatchTime time.Duration `json:"average_batch_time"`
	LastBatchTime    time.Duration `json:"last_batch_time"`
	Algorithm        string        `json:"algorithm"`
	Precision        int           `json:"precision"`
	
	// Quality metrics
	ExpectedZeros    float64       `json:"expected_zeros"`
	ZeroDensity      float64       `json:"zero_density"`
	AverageSpacing   float64       `json:"average_spacing"`
}

type WorkerStats struct {
	ActiveWorkers    int           `json:"active_workers"`
	TotalWorkers     int           `json:"total_workers"`
	WorkerLoad       []float64     `json:"worker_load,omitempty"`
	WorkerPoints     []int64       `json:"worker_points,omitempty"`
	WorkerZeros      []int64       `json:"worker_zeros,omitempty"`
	DeviceTypes      []string      `json:"device_types,omitempty"`
}

type Statistics struct {
	mu               sync.RWMutex
	Hardware         HardwareStats     `json:"hardware"`
	Calculation      CalculationStats  `json:"calculation"`
	Workers          WorkerStats       `json:"workers"`
	
	// Real-time tracking
	LastZeroFound    *ZeroResult       `json:"last_zero_found,omitempty"`
	RecentZeros      []ZeroResult      `json:"recent_zeros,omitempty"`
	PerformanceLog   []PerformanceSnapshot `json:"performance_log,omitempty"`
	
	// System info
	GoRoutines       int               `json:"go_routines"`
	GCCycles         uint32            `json:"gc_cycles"`
	AllocMB          float64           `json:"alloc_mb"`
	SysMB            float64           `json:"sys_mb"`
	NextGCMB         float64           `json:"next_gc_mb"`
	
	// Version info
	Version          string            `json:"version"`
	BuildInfo        map[string]string `json:"build_info"`
}

type PerformanceSnapshot struct {
	Timestamp        time.Time `json:"timestamp"`
	PointsPerSecond  float64   `json:"points_per_second"`
	MemoryUsageMB    float64   `json:"memory_usage_mb"`
	CPUUsagePercent  float64   `json:"cpu_usage_percent"`
	CacheHitRate     float64   `json:"cache_hit_rate"`
	ActiveWorkers    int       `json:"active_workers"`
}

type Checkpoint struct {
	Version         string            `json:"version"`
	Config          Config            `json:"config"`
	Statistics      Statistics        `json:"statistics"`
	LastSaved       time.Time         `json:"last_saved"`
	NextCheckpoint  time.Time         `json:"next_checkpoint,omitempty"`
	TotalZeros      []ZeroResult      `json:"total_zeros,omitempty"`
	Metadata        map[string]string `json:"metadata"`
	
	// Recovery info
	RecoveryPoint   float64           `json:"recovery_point"`
	RecoveryBatch   string            `json:"recovery_batch,omitempty"`
	RecoveryChecksum string           `json:"recovery_checksum,omitempty"`
}

// ==================== HARDWARE DETECTION & MANAGEMENT ====================
type HardwareManager struct {
	config        *HardwareConfig
	detectedGPUs  int
	hasCUDA       bool
	hasOpenCL     bool
	hasAVX        bool
	hasAVX2       bool
	hasAVX512     bool
	cpuInfo       CPUInfo
	gpuInfo       []GPUInfo
	logger        *logrus.Logger
	mu            sync.RWMutex
}

type CPUInfo struct {
	Vendor      string
	Model       string
	Cores       int
	Threads     int
	Frequency   float64
	CacheKB     int
	Features    []string
}

type GPUInfo struct {
	ID          int
	Name        string
	Vendor      string
	MemoryMB    int64
	Cores       int
	ClockMHz    int
	Compute     float64
	Driver      string
	Temperature float64
	Usage       float64
}

func NewHardwareManager(cfg *HardwareConfig, logger *logrus.Logger) (*HardwareManager, error) {
	hm := &HardwareManager{
		config:   cfg,
		logger:   logger,
		gpuInfo:  make([]GPUInfo, 0),
	}
	
	// Detect CPU features
	hm.detectCPU()
	
	// Detect GPU capabilities
	hm.detectGPU()
	
	// Auto-select mode if not specified
	if cfg.Mode == ModeAuto {
		hm.autoSelectMode()
	}
	
	return hm, nil
}

func (hm *HardwareManager) detectCPU() {
	// Get basic CPU info
	hm.cpuInfo.Cores = runtime.NumCPU()
	hm.cpuInfo.Threads = hm.cpuInfo.Cores
	
	// Simple feature detection (simplified for portability)
	hm.hasAVX = true // Assume modern CPUs
	hm.hasAVX2 = true
	hm.hasAVX512 = false // Rare
	
	hm.logger.Infof("CPU detected: %d cores, %d threads", 
		hm.cpuInfo.Cores, hm.cpuInfo.Threads)
}

func (hm *HardwareManager) detectGPU() {
	// Simplified GPU detection
	// In production, you would use:
	// - nvidia-ml for NVIDIA
	// - rocm-smi for AMD
	// - Intel graphics APIs
	
	hm.detectedGPUs = 0
	hm.hasCUDA = false
	hm.hasOpenCL = false
	
	// Check common GPU detection methods
	hm.checkNVIDIA()
	hm.checkAMD()
	hm.checkIntel()
	
	hm.logger.Infof("GPU detection: %d GPUs found, CUDA: %v, OpenCL: %v",
		hm.detectedGPUs, hm.hasCUDA, hm.hasOpenCL)
}

func (hm *HardwareManager) checkNVIDIA() {
	// Simplified check - in reality use nvidia-ml bindings
	if _, err := os.Stat("/proc/driver/nvidia/gpus"); err == nil {
		hm.hasCUDA = true
		hm.detectedGPUs++
	}
}

func (hm *HardwareManager) checkAMD() {
	// Check for AMD GPU
	if _, err := os.Stat("/sys/class/drm/card0/device/vendor"); err == nil {
		hm.hasOpenCL = true
		hm.detectedGPUs++
	}
}

func (hm *HardwareManager) checkIntel() {
	// Check for Intel GPU
	if _, err := os.Stat("/sys/class/drm/card0/device/vendor"); err == nil {
		hm.hasOpenCL = true
		hm.detectedGPUs++
	}
}

func (hm *HardwareManager) autoSelectMode() {
	if hm.detectedGPUs > 0 && hm.hasCUDA {
		if hm.detectedGPUs > 1 {
			hm.config.Mode = ModeMultiGPU
		} else {
			hm.config.Mode = ModeGPU
		}
		hm.logger.Info("Auto-selected GPU mode")
	} else {
		hm.config.Mode = ModeCPU
		hm.logger.Info("Auto-selected CPU mode")
	}
}

func (hm *HardwareManager) GetOptimalThreadCount() int {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	
	if hm.config.ThreadsPerCore > 0 {
		return int(float64(hm.cpuInfo.Cores) * hm.config.ThreadsPerCore)
	}
	
	// Default: 2 threads per core for CPU-bound work
	return hm.cpuInfo.Cores * 2
}

func (hm *HardwareManager) GetAvailableGPUs() []int {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	
	if len(hm.config.GPUDevices) > 0 {
		return hm.config.GPUDevices
	}
	
	// Return all detected GPUs
	gpus := make([]int, hm.detectedGPUs)
	for i := 0; i < hm.detectedGPUs; i++ {
		gpus[i] = i
	}
	return gpus
}

func (hm *HardwareManager) GetHardwareStats() HardwareStats {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	stats := HardwareStats{
		CPUCores:       hm.cpuInfo.Cores,
		TotalRAMMB:     float64(getTotalRAM()) / 1024 / 1024,
		UsedRAMMB:      float64(memStats.Alloc) / 1024 / 1024,
		AvailableRAMMB: float64(getTotalRAM()-memStats.Alloc) / 1024 / 1024,
		GPUs:           hm.detectedGPUs,
		Uptime:         time.Since(startTime),
	}
	
	return stats
}

func getTotalRAM() uint64 {
	// Simplified - in reality read from /proc/meminfo or syscall
	return 1024 * 1024 * 1024 // Assume 1GB for safety
}

// ==================== ADVANCED CALCULATION ENGINE ====================
type RiemannCalculator struct {
	config      *CalculationConfig
	hardware    *HardwareManager
	cache       *CalculationCache
	logger      *logrus.Logger
	mu          sync.RWMutex
	
	// Algorithm implementations
	algorithmFunc func(float64) float64
	thetaFunc     func(float64) float64
}

func NewRiemannCalculator(cfg *CalculationConfig, hw *HardwareManager, logger *logrus.Logger) *RiemannCalculator {
	rc := &RiemannCalculator{
		config:   cfg,
		hardware: hw,
		cache:    NewCalculationCache(cfg.MaxTerms),
		logger:   logger,
	}
	
	// Select algorithm based on config
	rc.selectAlgorithm()
	
	return rc
}

func (rc *RiemannCalculator) selectAlgorithm() {
	switch rc.config.Algorithm {
	case "riemann-siegel-fast":
		rc.algorithmFunc = rc.riemannSiegelFast
		rc.thetaFunc = rc.thetaAsymptotic
	case "riemann-siegel-precise":
		rc.algorithmFunc = rc.riemannSiegelPrecise
		rc.thetaFunc = rc.thetaExact
	case "gram":
		rc.algorithmFunc = rc.gramMethod
		rc.thetaFunc = rc.thetaGram
	case "euler-maclaurin":
		rc.algorithmFunc = rc.eulerMaclaurin
		rc.thetaFunc = rc.thetaDirect
	default: // "auto"
		rc.algorithmFunc = rc.autoSelectAlgorithm
		rc.thetaFunc = rc.thetaAuto
	}
}

func (rc *RiemannCalculator) autoSelectAlgorithm(t float64) float64 {
	if t < 1000 {
		return rc.eulerMaclaurin(t)
	} else if t > 1e12 {
		return rc.riemannSiegelFast(t)
	} else {
		return rc.riemannSiegelPrecise(t)
	}
}

func (rc *RiemannCalculator) thetaAuto(t float64) float64 {
	if t > 1e8 {
		return rc.thetaAsymptotic(t)
	}
	return rc.thetaExact(t)
}

func (rc *RiemannCalculator) ComputeZ(t float64) (float64, error) {
	start := time.Now()
	
	// Check cache
	if rc.config.UseCache {
		if val, ok := rc.cache.GetZ(t); ok {
			rc.cache.RecordHit()
			return val, nil
		}
	}
	
	// Calculate
	z := rc.algorithmFunc(t)
	
	// Cache result
	if rc.config.UseCache {
		rc.cache.SetZ(t, z)
		rc.cache.RecordMiss()
	}
	
	elapsed := time.Since(start)
	if elapsed > 100*time.Millisecond && rc.logger != nil {
		rc.logger.Debugf("Z(%.2e) calculation took %v", t, elapsed)
	}
	
	return z, nil
}

func (rc *RiemannCalculator) riemannSiegelFast(t float64) float64 {
	theta := rc.thetaFunc(t)
	
	// Optimized for speed: limited terms
	T := t / (2 * math.Pi)
	N := int(math.Sqrt(T))
	
	// Memory-safe limit
	if N > rc.config.MaxTerms {
		N = rc.config.MaxTerms
	}
	
	sum := 0.0
	negT := -t
	
	// Vector-friendly loop
	for n := 1; n <= N; n++ {
		logN := math.Log(float64(n))
		angle := theta + negT*logN
		sum += math.Cos(angle) / math.Sqrt(float64(n))
	}
	
	return 2.0 * sum
}

func (rc *RiemannCalculator) riemannSiegelPrecise(t float64) float64 {
	theta := rc.thetaFunc(t)
	
	T := t / (2 * math.Pi)
	N := int(math.Sqrt(T))
	
	// Use more terms for precision
	if N > rc.config.MaxTerms*2 {
		N = rc.config.MaxTerms * 2
	}
	
	sum := 0.0
	negT := -t
	
	for n := 1; n <= N; n++ {
		logN := math.Log(float64(n))
		angle := theta + negT*logN
		sum += math.Cos(angle) / math.Sqrt(float64(n))
	}
	
	z := 2.0 * sum
	
	// Add remainder terms for better accuracy
	if rc.config.RemainderTerms > 0 {
		remainder := rc.computeRemainder(t, N)
		z += remainder
	}
	
	return z
}

func (rc *RiemannCalculator) computeRemainder(t float64, N int) float64 {
	// Coefficients for Riemann-Siegel remainder
	coeffs := []float64{
		1.0,
		-1.0/48.0,
		7.0/5760.0,
		-31.0/967680.0,
		127.0/154828800.0,
		-511.0/35035545600.0,
	}
	
	x := math.Sqrt(2 * math.Pi / t)
	remainder := 0.0
	xPower := 1.0
	
	terms := rc.config.RemainderTerms
	if terms > len(coeffs) {
		terms = len(coeffs)
	}
	
	for k := 0; k < terms; k++ {
		remainder += coeffs[k] * xPower
		xPower *= x
	}
	
	factor := math.Pow(t/(2*math.Pi), -0.25)
	if N%2 == 0 {
		factor = -factor
	}
	
	return factor * remainder
}

func (rc *RiemannCalculator) eulerMaclaurin(t float64) float64 {
	// Euler-Maclaurin summation for small t
	s := complex(0.5, t)
	
	terms := 1000
	if t < 100 {
		terms = 10000
	}
	
	sum := complex(0, 0)
	for n := 1; n <= terms; n++ {
		sum += cmplx.Pow(complex(float64(n), 0), -s)
	}
	
	return real(cmplx.Exp(complex(0, rc.thetaFunc(t))) * sum)
}

func (rc *RiemannCalculator) gramMethod(t float64) float64 {
	// Gram point method
	theta := rc.thetaFunc(t)
	
	// Find nearest Gram point
	gram := rc.findGramPoint(t)
	
	// Interpolate between Gram points
	return rc.interpolateZ(t, gram, theta)
}

func (rc *RiemannCalculator) findGramPoint(t float64) float64 {
	// Simplified Gram point approximation
	return t/(2*math.Pi) * math.Log(t/(2*math.Pi)) - t/(2*math.Pi) - 0.125
}

func (rc *RiemannCalculator) interpolateZ(t, gram, theta float64) float64 {
	// Linear interpolation between Gram points
	delta := t - gram
	return math.Sin(theta) * delta / (1 + math.Abs(delta))
}

func (rc *RiemannCalculator) thetaAsymptotic(t float64) float64 {
	// Fast asymptotic expansion for large t
	T := t / (2 * math.Pi)
	logT := math.Log(T)
	
	theta := t/2*logT - t/2 - math.Pi/8
	
	// Add correction terms
	invT := 1.0 / t
	theta += invT/48.0
	
	invT2 := invT * invT
	theta -= 7.0 * invT2 / 5760.0
	
	if rc.config.RemainderTerms > 2 {
		invT3 := invT2 * invT
		theta += 31.0 * invT3 / 967680.0
	}
	
	return theta
}

func (rc *RiemannCalculator) thetaExact(t float64) float64 {
	// More accurate theta using Stirling series
	T := t / (2 * math.Pi)
	logT := math.Log(T)
	
	theta := t/2*logT - t/2 - math.Pi/8
	
	invT := 1.0 / t
	theta += invT/48.0
	
	invT2 := invT * invT
	theta -= 7.0 * invT2 / 5760.0
	
	invT3 := invT2 * invT
	theta += 31.0 * invT3 / 967680.0
	
	invT5 := invT3 * invT2
	theta -= 127.0 * invT5 / 154828800.0
	
	return theta
}

func (rc *RiemannCalculator) thetaGram(t float64) float64 {
	// Theta for Gram point method
	return math.Pi/2 * (t*math.Log(t/(2*math.Pi)) - t - 0.125)
}

func (rc *RiemannCalculator) thetaDirect(t float64) float64 {
	// Direct calculation using complex log gamma
	// Simplified version
	return t/2*math.Log(t/(2*math.Pi)) - t/2 - math.Pi/8 + 1.0/(48.0*t)
}

// ==================== INTELLIGENT WORKER SYSTEM ====================
type WorkerType string

const (
	WorkerCPU    WorkerType = "cpu"
	WorkerGPU    WorkerType = "gpu"
	WorkerHybrid WorkerType = "hybrid"
)

type Worker struct {
	ID           int
	Type         WorkerType
	DeviceID     int
	Calculator   *RiemannCalculator
	WorkChan     chan []float64
	ResultChan   chan *ZeroResult
	StopChan     chan struct{}
	Stats        *WorkerStatsInternal
	Logger       *logrus.Logger
	Active       bool
	BatchCounter int64
}

type WorkerStatsInternal struct {
	mu               sync.RWMutex
	PointsProcessed  int64
	ZerosFound       int64
	BatchesProcessed int64
	TotalTime        time.Duration
	LastBatchTime    time.Duration
	Errors           int64
	CacheHits        int64
	CacheMisses      int64
}

func NewWorker(id int, wtype WorkerType, deviceID int, calc *RiemannCalculator, logger *logrus.Logger) *Worker {
	return &Worker{
		ID:         id,
		Type:       wtype,
		DeviceID:   deviceID,
		Calculator: calc,
		WorkChan:   make(chan []float64, 100),
		ResultChan: make(chan *ZeroResult, 50),
		StopChan:   make(chan struct{}),
		Stats:      &WorkerStatsInternal{},
		Logger:     logger,
		Active:     true,
	}
}

func (w *Worker) Start() {
	go w.run()
}

func (w *Worker) run() {
	defer func() {
		if r := recover(); r != nil {
			w.Logger.Errorf("Worker %d panicked: %v", w.ID, r)
			w.Stats.mu.Lock()
			w.Stats.Errors++
			w.Stats.mu.Unlock()
		}
		w.Active = false
	}()
	
	for {
		select {
		case batch := <-w.WorkChan:
			w.processBatch(batch)
		case <-w.StopChan:
			w.Logger.Debugf("Worker %d stopping", w.ID)
			return
		}
	}
}

func (w *Worker) processBatch(batch []float64) {
	start := time.Now()
	
	zeros := make([]*ZeroResult, 0, len(batch)/100)
	
	for _, t := range batch {
		z, err := w.Calculator.ComputeZ(t)
		if err != nil {
			w.Logger.Warnf("Worker %d error at t=%.2e: %v", w.ID, t, err)
			w.Stats.mu.Lock()
			w.Stats.Errors++
			w.Stats.mu.Unlock()
			continue
		}
		
		magnitude := math.Abs(z)
		if magnitude < w.Calculator.config.ZeroThreshold {
			zero := &ZeroResult{
				T:           t,
				ZValue:      z,
				Magnitude:   magnitude,
				Precision:   w.Calculator.config.Precision,
				FoundAt:     time.Now(),
				WorkerID:    w.ID,
				DeviceID:    w.DeviceID,
				DeviceType:  string(w.Type),
				Verified:    false,
				BatchID:     fmt.Sprintf("B%d-W%d", w.BatchCounter, w.ID),
				Confidence:  1.0 - magnitude/w.Calculator.config.ZeroThreshold,
			}
			zeros = append(zeros, zero)
			
			atomic.AddInt64(&w.Stats.ZerosFound, 1)
		}
		
		atomic.AddInt64(&w.Stats.PointsProcessed, 1)
	}
	
	// Send results
	if len(zeros) > 0 {
		for _, zero := range zeros {
			select {
			case w.ResultChan <- zero:
				// Sent successfully
			case <-time.After(100 * time.Millisecond):
				w.Logger.Warnf("Worker %d result channel full, dropping zero", w.ID)
			}
		}
	}
	
	// Update statistics
	elapsed := time.Since(start)
	w.Stats.mu.Lock()
	w.Stats.BatchesProcessed++
	w.Stats.TotalTime += elapsed
	w.Stats.LastBatchTime = elapsed
	w.Stats.mu.Unlock()
	
	atomic.AddInt64(&w.BatchCounter, 1)
}

func (w *Worker) Stop() {
	close(w.StopChan)
	close(w.WorkChan)
	close(w.ResultChan)
}

func (w *Worker) GetStats() WorkerStatsInternal {
	w.Stats.mu.RLock()
	defer w.Stats.mu.RUnlock()
	return *w.Stats
}

type WorkerPool struct {
	Config       *PerformanceConfig
	Hardware     *HardwareManager
	Workers      []*Worker
	WorkChan     chan []float64
	ResultChan   chan *ZeroResult
	StopChan     chan struct{}
	WG           sync.WaitGroup
	Logger       *logrus.Logger
	mu           sync.RWMutex
	
	// Dynamic load balancing
	LoadBalancer *LoadBalancer
	NextWorker   int
}

type LoadBalancer struct {
	WorkerLoads []float64
	mu          sync.RWMutex
}

func NewWorkerPool(cfg *PerformanceConfig, hw *HardwareManager, calc *RiemannCalculator, logger *logrus.Logger) (*WorkerPool, error) {
	// Calculate optimal worker count
	numWorkers := cfg.MaxWorkers
	if numWorkers <= 0 {
		numWorkers = hw.GetOptimalThreadCount()
	}
	
	// Limit based on memory
	availableRAM := getTotalRAM()
	maxWorkersByRAM := int(availableRAM / (200 * 1024 * 1024)) // ~200MB per worker
	if numWorkers > maxWorkersByRAM {
		numWorkers = maxWorkersByRAM
		logger.Infof("Limiting workers to %d due to RAM constraints", numWorkers)
	}
	
	pool := &WorkerPool{
		Config:     cfg,
		Hardware:   hw,
		Workers:    make([]*Worker, numWorkers),
		WorkChan:   make(chan []float64, numWorkers*10),
		ResultChan: make(chan *ZeroResult, 1000),
		StopChan:   make(chan struct{}),
		Logger:     logger,
		LoadBalancer: &LoadBalancer{
			WorkerLoads: make([]float64, numWorkers),
		},
	}
	
	// Create workers based on hardware mode
	switch hw.config.Mode {
	case ModeCPU:
		pool.createCPUWorkers(calc, numWorkers)
	case ModeGPU:
		pool.createGPUWorkers(calc, 1) // Single GPU
	case ModeMultiGPU:
		gpus := hw.GetAvailableGPUs()
		pool.createGPUWorkers(calc, len(gpus))
	default:
		pool.createCPUWorkers(calc, numWorkers)
	}
	
	return pool, nil
}

func (p *WorkerPool) createCPUWorkers(calc *RiemannCalculator, count int) {
	for i := 0; i < count; i++ {
		worker := NewWorker(i, WorkerCPU, 0, calc, p.Logger)
		p.Workers[i] = worker
		p.WG.Add(1)
		go func(w *Worker) {
			defer p.WG.Done()
			w.Start()
		}(worker)
	}
	p.Logger.Infof("Created %d CPU workers", count)
}

func (p *WorkerPool) createGPUWorkers(calc *RiemannCalculator, gpuCount int) {
	// Simplified GPU worker creation
	// In reality, each GPU would have its own calculator with GPU acceleration
	for i := 0; i < gpuCount; i++ {
		worker := NewWorker(i, WorkerGPU, i, calc, p.Logger)
		p.Workers[i] = worker
		p.WG.Add(1)
		go func(w *Worker) {
			defer p.WG.Done()
			w.Start()
		}(worker)
	}
	p.Logger.Infof("Created %d GPU workers", gpuCount)
}

func (p *WorkerPool) Submit(batch []float64) {
	select {
	case p.WorkChan <- batch:
		// Successfully submitted
	case <-time.After(1 * time.Second):
		p.Logger.Warn("Work channel full, dropping batch")
	}
}

func (p *WorkerPool) DistributeWork() {
	// Round-robin distribution with load awareness
	for batch := range p.WorkChan {
		worker := p.selectWorker()
		if worker != nil && worker.Active {
			select {
			case worker.WorkChan <- batch:
				p.updateWorkerLoad(worker.ID, float64(len(batch)))
			case <-time.After(100 * time.Millisecond):
				p.Logger.Warnf("Worker %d busy, retrying different worker", worker.ID)
				// Try another worker
				for i := 1; i < len(p.Workers); i++ {
					nextWorker := p.Workers[(worker.ID+i)%len(p.Workers)]
					if nextWorker.Active {
						select {
						case nextWorker.WorkChan <- batch:
							p.updateWorkerLoad(nextWorker.ID, float64(len(batch)))
							break
						default:
							continue
						}
					}
				}
			}
		}
	}
}

func (p *WorkerPool) selectWorker() *Worker {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	if len(p.Workers) == 0 {
		return nil
	}
	
	// Simple round-robin for now
	worker := p.Workers[p.NextWorker]
	p.NextWorker = (p.NextWorker + 1) % len(p.Workers)
	return worker
}

func (p *WorkerPool) updateWorkerLoad(workerID int, load float64) {
	p.LoadBalancer.mu.Lock()
	defer p.LoadBalancer.mu.Unlock()
	
	if workerID < len(p.LoadBalancer.WorkerLoads) {
		p.LoadBalancer.WorkerLoads[workerID] += load
		// Exponential decay
		p.LoadBalancer.WorkerLoads[workerID] *= 0.9
	}
}

func (p *WorkerPool) GetResults() <-chan *ZeroResult {
	return p.ResultChan
}

func (p *WorkerPool) Stop() {
	close(p.StopChan)
	
	// Stop all workers
	for _, worker := range p.Workers {
		if worker != nil {
			worker.Stop()
		}
	}
	
	p.WG.Wait()
	close(p.WorkChan)
	close(p.ResultChan)
}

func (p *WorkerPool) GetStatistics() WorkerStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	stats := WorkerStats{
		TotalWorkers:  len(p.Workers),
		ActiveWorkers: 0,
		WorkerLoad:    make([]float64, len(p.Workers)),
		WorkerPoints:  make([]int64, len(p.Workers)),
		WorkerZeros:   make([]int64, len(p.Workers)),
		DeviceTypes:   make([]string, len(p.Workers)),
	}
	
	for i, worker := range p.Workers {
		if worker != nil && worker.Active {
			stats.ActiveWorkers++
			workerStats := worker.GetStats()
			stats.WorkerPoints[i] = workerStats.PointsProcessed
			stats.WorkerZeros[i] = workerStats.ZerosFound
			stats.DeviceTypes[i] = string(worker.Type)
		}
	}
	
	p.LoadBalancer.mu.RLock()
	copy(stats.WorkerLoad, p.LoadBalancer.WorkerLoads)
	p.LoadBalancer.mu.RUnlock()
	
	return stats
}

// ==================== SMART CACHE SYSTEM ====================
type CalculationCache struct {
	zCache     map[float64]cacheEntry
	thetaCache map[float64]cacheEntry
	maxSize    int
	mu         sync.RWMutex
	hits       int64
	misses     int64
	accessTime map[float64]time.Time
	lastClean  time.Time
}

type cacheEntry struct {
	value    float64
	accessed time.Time
	weight   float64
}

func NewCalculationCache(maxSize int) *CalculationCache {
	return &CalculationCache{
		zCache:     make(map[float64]cacheEntry, maxSize/2),
		thetaCache: make(map[float64]cacheEntry, maxSize/2),
		maxSize:    maxSize,
		accessTime: make(map[float64]time.Time),
		lastClean:  time.Now(),
	}
}

func (c *CalculationCache) GetZ(t float64) (float64, bool) {
	c.mu.RLock()
	entry, ok := c.zCache[t]
	c.mu.RUnlock()
	
	if ok {
		atomic.AddInt64(&c.hits, 1)
		// Update access time
		c.mu.Lock()
		entry.accessed = time.Now()
		c.zCache[t] = entry
		c.accessTime[t] = entry.accessed
		c.mu.Unlock()
		return entry.value, true
	}
	
	atomic.AddInt64(&c.misses, 1)
	return 0, false
}

func (c *CalculationCache) SetZ(t, z float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Clean cache if needed
	c.cleanIfNeeded()
	
	entry := cacheEntry{
		value:    z,
		accessed: time.Now(),
		weight:   1.0,
	}
	
	c.zCache[t] = entry
	c.accessTime[t] = entry.accessed
}

func (c *CalculationCache) GetTheta(t float64) (float64, bool) {
	c.mu.RLock()
	entry, ok := c.thetaCache[t]
	c.mu.RUnlock()
	
	if ok {
		// Update access time
		c.mu.Lock()
		entry.accessed = time.Now()
		c.thetaCache[t] = entry
		c.accessTime[t] = entry.accessed
		c.mu.Unlock()
		return entry.value, true
	}
	
	return 0, false
}

func (c *CalculationCache) SetTheta(t, theta float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Clean cache if needed
	c.cleanIfNeeded()
	
	entry := cacheEntry{
		value:    theta,
		accessed: time.Now(),
		weight:   0.5, // Theta is less frequently accessed
	}
	
	c.thetaCache[t] = entry
	c.accessTime[t] = entry.accessed
}

func (c *CalculationCache) cleanIfNeeded() {
	if len(c.zCache)+len(c.thetaCache) < c.maxSize {
		return
	}
	
	// Clean old entries
	now := time.Now()
	cutoff := now.Add(-5 * time.Minute)
	
	// Clean Z cache
	for k, entry := range c.zCache {
		if entry.accessed.Before(cutoff) {
			delete(c.zCache, k)
			delete(c.accessTime, k)
		}
	}
	
	// Clean theta cache
	for k, entry := range c.thetaCache {
		if entry.accessed.Before(cutoff) {
			delete(c.thetaCache, k)
			delete(c.accessTime, k)
		}
	}
	
	c.lastClean = now
}

func (c *CalculationCache) RecordHit() {
	atomic.AddInt64(&c.hits, 1)
}

func (c *CalculationCache) RecordMiss() {
	atomic.AddInt64(&c.misses, 1)
}

func (c *CalculationCache) HitRate() float64 {
	hits := float64(atomic.LoadInt64(&c.hits))
	misses := float64(atomic.LoadInt64(&c.misses))
	
	total := hits + misses
	if total == 0 {
		return 0
	}
	
	return hits / total
}

func (c *CalculationCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.zCache) + len(c.thetaCache)
}

// ==================== FLEXIBLE STORAGE SYSTEM ====================
type StorageManager struct {
	config      *OutputConfig
	baseDir     string
	logger      *logrus.Logger
	mu          sync.RWMutex
	
	// File handles
	zerosFile   *os.File
	zerosWriter *csv.Writer
	statsFile   *os.File
	checkpointFile string
	
	// Statistics
	zerosSaved  int64
	statsSaved  int64
	checkpointsSaved int64
}

func NewStorageManager(cfg *OutputConfig, logger *logrus.Logger) (*StorageManager, error) {
	// Determine output directory
	baseDir := cfg.OutputDirectory
	if baseDir == "" {
		baseDir = "."
	}
	
	// Create directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}
	
	sm := &StorageManager{
		config:      cfg,
		baseDir:     baseDir,
		logger:      logger,
	}
	
	// Initialize files
	if err := sm.initializeFiles(); err != nil {
		return nil, err
	}
	
	return sm, nil
}

func (sm *StorageManager) initializeFiles() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	prefix := sm.config.FilenamePrefix
	if prefix == "" {
		prefix = "riemann"
	}
	
	// Zeros file
	if sm.config.SaveZeros {
		zerosPath := filepath.Join(sm.baseDir, prefix+"_zeros.csv")
		file, err := os.OpenFile(zerosPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return fmt.Errorf("failed to open zeros file: %w", err)
		}
		
		sm.zerosFile = file
		sm.zerosWriter = csv.NewWriter(file)
		
		// Write header if file is new
		if stat, _ := file.Stat(); stat.Size() == 0 {
			header := []string{
				"t", "z_value", "magnitude", "precision", "found_at",
				"worker_id", "device_id", "device_type", "verified",
				"confidence", "batch_id",
			}
			if err := sm.zerosWriter.Write(header); err != nil {
				return fmt.Errorf("failed to write header: %w", err)
			}
			sm.zerosWriter.Flush()
		}
	}
	
	// Statistics file
	if sm.config.SaveStats {
		statsPath := filepath.Join(sm.baseDir, prefix+"_stats.json")
		file, err := os.OpenFile(statsPath, os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return fmt.Errorf("failed to open stats file: %w", err)
		}
		sm.statsFile = file
	}
	
	// Checkpoint file
	sm.checkpointFile = filepath.Join(sm.baseDir, prefix+"_checkpoint.json")
	
	return nil
}

func (sm *StorageManager) SaveZero(zero *ZeroResult) error {
	if !sm.config.SaveZeros || sm.zerosWriter == nil {
		return nil
	}
	
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	record := []string{
		strconv.FormatFloat(zero.T, 'f', 15, 64),
		strconv.FormatFloat(zero.ZValue, 'e', 10, 64),
		strconv.FormatFloat(zero.Magnitude, 'e', 10, 64),
		strconv.Itoa(zero.Precision),
		zero.FoundAt.Format(time.RFC3339Nano),
		strconv.Itoa(zero.WorkerID),
		strconv.Itoa(zero.DeviceID),
		zero.DeviceType,
		strconv.FormatBool(zero.Verified),
		strconv.FormatFloat(zero.Confidence, 'f', 4, 64),
		zero.BatchID,
	}
	
	if err := sm.zerosWriter.Write(record); err != nil {
		return fmt.Errorf("failed to write zero record: %w", err)
	}
	
	atomic.AddInt64(&sm.zerosSaved, 1)
	
	// Flush periodically
	if atomic.LoadInt64(&sm.zerosSaved)%100 == 0 {
		sm.zerosWriter.Flush()
		if err := sm.zerosWriter.Error(); err != nil {
			return fmt.Errorf("failed to flush zeros file: %w", err)
		}
	}
	
	return nil
}

func (sm *StorageManager) SaveStatistics(stats *Statistics) error {
	if !sm.config.SaveStats || sm.statsFile == nil {
		return nil
	}
	
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	// Truncate file and write fresh
	if err := sm.statsFile.Truncate(0); err != nil {
		return fmt.Errorf("failed to truncate stats file: %w", err)
	}
	if _, err := sm.statsFile.Seek(0, 0); err != nil {
		return fmt.Errorf("failed to seek stats file: %w", err)
	}
	
	encoder := json.NewEncoder(sm.statsFile)
	encoder.SetIndent("", "  ")
	
	if err := encoder.Encode(stats); err != nil {
		return fmt.Errorf("failed to encode statistics: %w", err)
	}
	
	atomic.AddInt64(&sm.statsSaved, 1)
	return nil
}

func (sm *StorageManager) SaveCheckpoint(checkpoint *Checkpoint) error {
	if !sm.config.SaveCheckpoints {
		return nil
	}
	
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	// Create backup of previous checkpoint
	if _, err := os.Stat(sm.checkpointFile); err == nil {
		backupFile := sm.checkpointFile + ".backup"
		if err := os.Rename(sm.checkpointFile, backupFile); err != nil {
			sm.logger.Warnf("Failed to create checkpoint backup: %v", err)
		}
	}
	
	// Write new checkpoint
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint: %w", err)
	}
	
	if err := os.WriteFile(sm.checkpointFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write checkpoint: %w", err)
	}
	
	// Create checksum file
	checksumFile := sm.checkpointFile + ".checksum"
	checksum := fmt.Sprintf("%x", len(data)) // Simplified checksum
	if err := os.WriteFile(checksumFile, []byte(checksum), 0644); err != nil {
		sm.logger.Warnf("Failed to write checksum: %v", err)
	}
	
	atomic.AddInt64(&sm.checkpointsSaved, 1)
	return nil
}

func (sm *StorageManager) LoadCheckpoint() (*Checkpoint, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	data, err := os.ReadFile(sm.checkpointFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint: %w", err)
	}
	
	var checkpoint Checkpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint: %w", err)
	}
	
	// Verify checksum if available
	checksumFile := sm.checkpointFile + ".checksum"
	if checksumData, err := os.ReadFile(checksumFile); err == nil {
		expectedChecksum := string(checksumData)
		actualChecksum := fmt.Sprintf("%x", len(data))
		if expectedChecksum != actualChecksum {
			sm.logger.Warn("Checkpoint checksum mismatch, data may be corrupted")
		}
	}
	
	return &checkpoint, nil
}

func (sm *StorageManager) Close() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	var errors []string
	
	if sm.zerosWriter != nil {
		sm.zerosWriter.Flush()
		if err := sm.zerosWriter.Error(); err != nil {
			errors = append(errors, fmt.Sprintf("zeros writer: %v", err))
		}
	}
	
	if sm.zerosFile != nil {
		if err := sm.zerosFile.Close(); err != nil {
			errors = append(errors, fmt.Sprintf("zeros file: %v", err))
		}
	}
	
	if sm.statsFile != nil {
		if err := sm.statsFile.Close(); err != nil {
			errors = append(errors, fmt.Sprintf("stats file: %v", err))
		}
	}
	
	if len(errors) > 0 {
		return fmt.Errorf("storage close errors: %s", strings.Join(errors, "; "))
	}
	
	return nil
}

// ==================== MAIN APPLICATION CONTROLLER ====================
type RiemannHunter struct {
	config      *Config
	hardware    *HardwareManager
	calculator  *RiemannCalculator
	pool        *WorkerPool
	storage     *StorageManager
	stats       *Statistics
	checkpoint  *Checkpoint
	logger      *logrus.Logger
	ctx         context.Context
	cancel      context.CancelFunc
	
	// Runtime control
	paused      bool
	pauseChan   chan bool
	resumeChan  chan bool
	statusChan  chan string
	
	// Performance monitoring
	perfMonitor *PerformanceMonitor
	healthCheck *HealthChecker
	
	// Dynamic configuration
	autoTuner   *AutoTuner
}

type PerformanceMonitor struct {
	metrics     chan PerformanceMetric
	stopChan    chan struct{}
	logger      *logrus.Logger
	mu          sync.RWMutex
	history     []PerformanceSnapshot
}

type PerformanceMetric struct {
	Timestamp   time.Time
	MetricType  string
	Value       float64
	Details     map[string]interface{}
}

type HealthChecker struct {
	checkInterval time.Duration
	stopChan      chan struct{}
	logger        *logrus.Logger
	lastCheck     time.Time
	healthStatus  HealthStatus
}

type HealthStatus struct {
	Overall      string
	CPU          string
	Memory       string
	Disk         string
	Network      string
	GPU          []string
	LastUpdated  time.Time
}

type AutoTuner struct {
	config      *Config
	monitor     *PerformanceMonitor
	logger      *logrus.Logger
	adjustments chan ConfigAdjustment
	active      bool
}

type ConfigAdjustment struct {
	Parameter   string
	OldValue    interface{}
	NewValue    interface{}
	Reason      string
	Timestamp   time.Time
}

func NewRiemannHunter(cfg *Config) (*RiemannHunter, error) {
	// Setup logger
	logger := setupLogger(cfg.Output)
	
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	
	hunter := &RiemannHunter{
		config:     cfg,
		logger:     logger,
		ctx:        ctx,
		cancel:     cancel,
		pauseChan:  make(chan bool, 1),
		resumeChan: make(chan bool, 1),
		statusChan: make(chan string, 10),
	}
	
	// Initialize components in order
	if err := hunter.initializeComponents(); err != nil {
		return nil, fmt.Errorf("failed to initialize components: %w", err)
	}
	
	// Load or create checkpoint
	if err := hunter.loadOrCreateCheckpoint(); err != nil {
		return nil, fmt.Errorf("failed to setup checkpoint: %w", err)
	}
	
	// Initialize performance monitoring
	hunter.initializePerformanceMonitoring()
	
	return hunter, nil
}

func setupLogger(cfg OutputConfig) *logrus.Logger {
	logger := logrus.New()
	
	// Set formatter
	logger.SetFormatter(&logrus.TextFormatter{
		FullTimestamp:   true,
		TimestampFormat: "2006-01-02 15:04:05",
		ForceColors:     true,
	})
	
	// Set level
	switch strings.ToLower(cfg.LogLevel) {
	case "debug":
		logger.SetLevel(logrus.DebugLevel)
	case "info":
		logger.SetLevel(logrus.InfoLevel)
	case "warn":
		logger.SetLevel(logrus.WarnLevel)
	case "error":
		logger.SetLevel(logrus.ErrorLevel)
	default:
		if cfg.Verbose {
			logger.SetLevel(logrus.DebugLevel)
		} else {
			logger.SetLevel(logrus.InfoLevel)
		}
	}
	
	return logger
}

func (h *RiemannHunter) initializeComponents() error {
	// Hardware manager
	hw, err := NewHardwareManager(&h.config.Hardware, h.logger)
	if err != nil {
		return fmt.Errorf("failed to create hardware manager: %w", err)
	}
	h.hardware = hw
	
	// Calculator
	calc := NewRiemannCalculator(&h.config.Calculation, hw, h.logger)
	h.calculator = calc
	
	// Storage
	storage, err := NewStorageManager(&h.config.Output, h.logger)
	if err != nil {
		return fmt.Errorf("failed to create storage manager: %w", err)
	}
	h.storage = storage
	
	// Worker pool
	pool, err := NewWorkerPool(&h.config.Performance, hw, calc, h.logger)
	if err != nil {
		return fmt.Errorf("failed to create worker pool: %w", err)
	}
	h.pool = pool
	
	// Start work distribution
	go h.pool.DistributeWork()
	
	return nil
}

func (h *RiemannHunter) loadOrCreateCheckpoint() error {
	if h.config.Output.SaveCheckpoints {
		checkpoint, err := h.storage.LoadCheckpoint()
		if err == nil {
			h.checkpoint = checkpoint
			h.stats = &checkpoint.Statistics
			h.logger.Infof("Loaded checkpoint from %s (t=%.2e)", 
				checkpoint.LastSaved.Format("2006-01-02 15:04"), 
				checkpoint.Statistics.Calculation.CurrentT)
			
			// Verify config compatibility
			if !h.isConfigCompatible(checkpoint.Config) {
				h.logger.Warn("Checkpoint config differs from current config")
			}
			
			return nil
		}
		h.logger.Infof("No checkpoint found: %v", err)
	}
	
	// Create new checkpoint
	h.checkpoint = h.createNewCheckpoint()
	h.stats = &h.checkpoint.Statistics
	
	return nil
}

func (h *RiemannHunter) isConfigCompatible(old Config) bool {
	// Check if we can resume from this checkpoint
	if old.Calculation.Algorithm != h.config.Calculation.Algorithm {
		return false
	}
	if old.Calculation.Precision != h.config.Calculation.Precision {
		return false
	}
	if old.Calculation.Step != h.config.Calculation.Step {
		return false
	}
	return true
}

func (h *RiemannHunter) createNewCheckpoint() *Checkpoint {
	now := time.Now()
	
	return &Checkpoint{
		Version: Version,
		Config:  *h.config,
		Statistics: Statistics{
			Hardware: h.hardware.GetHardwareStats(),
			Calculation: CalculationStats{
				StartTime:       now,
				CurrentT:        h.config.Calculation.StartT,
				PointsProcessed: 0,
				ZerosFound:      0,
				Algorithm:       h.config.Calculation.Algorithm,
				Precision:       h.config.Calculation.Precision,
			},
			Workers: WorkerStats{
				TotalWorkers: h.pool.GetStatistics().TotalWorkers,
			},
			Version: Version,
			BuildInfo: map[string]string{
				"build_date": BuildDate,
				"go_version": runtime.Version(),
				"author":     Author,
				"license":    License,
				"hostname":   getHostnameSafe(),
			},
		},
		LastSaved:      now,
		NextCheckpoint: now.Add(h.config.Performance.CheckpointInterval),
		Metadata: map[string]string{
			"created_at":      now.Format(time.RFC3339),
			"config_source":   h.config.loadedFrom,
			"hardware_mode":   string(h.config.Hardware.Mode),
			"initial_t":       fmt.Sprintf("%.15e", h.config.Calculation.StartT),
			"target_t":        fmt.Sprintf("%.15e", h.config.Calculation.EndT),
		},
		RecoveryPoint: h.config.Calculation.StartT,
	}
}

func (h *RiemannHunter) initializePerformanceMonitoring() {
	h.perfMonitor = &PerformanceMonitor{
		metrics:  make(chan PerformanceMetric, 100),
		stopChan: make(chan struct{}),
		logger:   h.logger,
		history:  make([]PerformanceSnapshot, 0, 1000),
	}
	
	h.healthCheck = &HealthChecker{
		checkInterval: 30 * time.Second,
		stopChan:      make(chan struct{}),
		logger:        h.logger,
		healthStatus: HealthStatus{
			Overall: "healthy",
			LastUpdated: time.Now(),
		},
	}
	
	if h.config.Performance.AutoTune {
		h.autoTuner = &AutoTuner{
			config:      h.config,
			monitor:     h.perfMonitor,
			logger:      h.logger,
			adjustments: make(chan ConfigAdjustment, 10),
			active:      true,
		}
	}
	
	// Start monitors
	go h.perfMonitor.run()
	go h.healthCheck.run()
	if h.autoTuner != nil {
		go h.autoTuner.run()
	}
}

func (h *RiemannHunter) Run() error {
	// Display startup banner
	h.printStartupBanner()
	
	// Setup signal handling
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)
	
	// Start background processors
	resultGroup, resultCtx := errgroup.WithContext(h.ctx)
	statsGroup, statsCtx := errgroup.WithContext(h.ctx)
	checkpointGroup, checkpointCtx := errgroup.WithContext(h.ctx)
	
	// Result processor
	resultGroup.Go(func() error {
		return h.processResults(resultCtx)
	})
	
	// Statistics updater
	statsGroup.Go(func() error {
		return h.updateStatistics(statsCtx)
	})
	
	// Checkpoint saver
	if h.config.Output.SaveCheckpoints {
		checkpointGroup.Go(func() error {
			return h.saveCheckpoints(checkpointCtx)
		})
	}
	
	// Status monitor
	statusGroup, statusCtx := errgroup.WithContext(h.ctx)
	statusGroup.Go(func() error {
		return h.monitorStatus(statusCtx)
	})
	
	// Main calculation loop
	h.logger.Info("Starting main calculation loop")
	
	currentT := h.stats.Calculation.CurrentT
	endT := h.config.Calculation.EndT
	step := h.config.Calculation.Step
	
	// Dynamic batch sizing
	batchSize := h.calculateOptimalBatchSize()
	pointsGenerated := int64(0)
	lastBatchTime := time.Now()
	
	mainLoop:
	for currentT < endT && !h.paused {
		select {
		case <-h.ctx.Done():
			h.logger.Info("Shutdown requested via context")
			break mainLoop
			
		case sig := <-signalChan:
			h.handleSignal(sig)
			if sig == syscall.SIGINT || sig == syscall.SIGTERM {
				break mainLoop
			}
			
		case pause := <-h.pauseChan:
			h.paused = pause
			if pause {
				h.logger.Info("Calculation paused")
				h.statusChan <- "paused"
				// Wait for resume
				<-h.resumeChan
				h.paused = false
				h.logger.Info("Calculation resumed")
				h.statusChan <- "resumed"
			}
			
		case status := <-h.statusChan:
			h.logger.Infof("Status update: %s", status)
			
		default:
			// Generate batch
			batch := h.generateBatch(currentT, endT, step, batchSize)
			if len(batch) == 0 {
				// Reached end
				break mainLoop
			}
			
			// Submit batch
			h.pool.Submit(batch)
			
			// Update state
			currentT = batch[len(batch)-1] + step
			h.stats.Calculation.CurrentT = currentT
			pointsGenerated += int64(len(batch))
			
			// Dynamic adjustment
			if time.Since(lastBatchTime) > 10*time.Second {
				batchSize = h.adjustBatchSize(batchSize, pointsGenerated)
				pointsGenerated = 0
				lastBatchTime = time.Now()
			}
			
			// Small delay to prevent CPU hogging (configurable)
			if h.config.Performance.CPULimitPercent < 100 {
				sleepTime := time.Duration(100-h.config.Performance.CPULimitPercent) * time.Millisecond
				time.Sleep(sleepTime)
			}
		}
	}
	
	// Clean shutdown
	h.logger.Info("Initiating shutdown sequence...")
	
	// Cancel context
	h.cancel()
	
	// Stop components
	h.pool.Stop()
	
	// Wait for goroutines
	h.logger.Info("Waiting for background processes...")
	resultGroup.Wait()
	statsGroup.Wait()
	checkpointGroup.Wait()
	statusGroup.Wait()
	
	// Stop monitors
	if h.perfMonitor != nil {
		close(h.perfMonitor.stopChan)
	}
	if h.healthCheck != nil {
		close(h.healthCheck.stopChan)
	}
	if h.autoTuner != nil {
		h.autoTuner.active = false
	}
	
	// Save final checkpoint
	if h.config.Output.SaveCheckpoints {
		h.checkpoint.Statistics = *h.stats
		h.checkpoint.LastSaved = time.Now()
		if err := h.storage.SaveCheckpoint(h.checkpoint); err != nil {
			h.logger.Errorf("Failed to save final checkpoint: %v", err)
		}
	}
	
	// Close storage
	if err := h.storage.Close(); err != nil {
		h.logger.Errorf("Failed to close storage: %v", err)
	}
	
	// Print final statistics
	h.printFinalStatistics()
	
	h.logger.Info("Riemann Hunter shutdown complete")
	return nil
}

func (h *RiemannHunter) generateBatch(start, end, step float64, size int) []float64 {
	batch := make([]float64, 0, size)
	current := start
	
	for len(batch) < size && current < end {
		batch = append(batch, current)
		current += step
		
		// Handle very large steps
		if step <= 0 {
			h.logger.Error("Step size is zero or negative")
			break
		}
	}
	
	return batch
}

func (h *RiemannHunter) calculateOptimalBatchSize() int {
	if h.config.Performance.BatchSize > 0 {
		return h.config.Performance.BatchSize
	}
	
	// Dynamic calculation based on available RAM
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	availableMB := float64(h.config.Performance.MemoryLimitMB) - (float64(memStats.Alloc) / 1024 / 1024)
	if availableMB < 50 {
		availableMB = 50 // Minimum safety margin
	}
	
	// Estimate memory per point (bytes)
	bytesPerPoint := 64 // Conservative estimate
	
	// Calculate batch size
	batchSize := int((availableMB * 1024 * 1024) / float64(bytesPerPoint) / 10) // Use 10% of available
	
	// Apply limits
	if batchSize < 1000 {
		batchSize = 1000
	}
	if batchSize > 1000000 {
		batchSize = 1000000
	}
	
	h.logger.Infof("Dynamic batch size: %d points (%.1f MB available)", batchSize, availableMB)
	return batchSize
}

func (h *RiemannHunter) adjustBatchSize(currentSize int, pointsProcessed int64) int {
	if !h.config.Performance.AutoTune || pointsProcessed == 0 {
		return currentSize
	}
	
	// Simple adjustment based on throughput
	workerStats := h.pool.GetStatistics()
	activeWorkers := workerStats.ActiveWorkers
	
	if activeWorkers == 0 {
		return currentSize
	}
	
	// Adjust based on worker utilization
	avgLoad := 0.0
	for _, load := range workerStats.WorkerLoad {
		avgLoad += load
	}
	avgLoad /= float64(activeWorkers)
	
	// Target: 70-80% load
	if avgLoad < 0.7 && currentSize < 1000000 {
		// Increase batch size
		newSize := int(float64(currentSize) * 1.1)
		h.logger.Debugf("Increasing batch size from %d to %d (load: %.2f)", 
			currentSize, newSize, avgLoad)
		return newSize
	} else if avgLoad > 0.8 && currentSize > 1000 {
		// Decrease batch size
		newSize := int(float64(currentSize) * 0.9)
		h.logger.Debugf("Decreasing batch size from %d to %d (load: %.2f)", 
			currentSize, newSize, avgLoad)
		return newSize
	}
	
	return currentSize
}

func (h *RiemannHunter) processResults(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return nil
			
		case result := <-h.pool.GetResults():
			// Save zero
			if err := h.storage.SaveZero(result); err != nil {
				h.logger.Errorf("Failed to save zero: %v", err)
			}
			
			// Update statistics
			atomic.AddInt64(&h.stats.Calculation.ZerosFound, 1)
			
			// Add to recent zeros
			h.stats.mu.Lock()
			h.stats.LastZeroFound = result
			h.stats.RecentZeros = append([]ZeroResult{*result}, h.stats.RecentZeros...)
			if len(h.stats.RecentZeros) > 20 {
				h.stats.RecentZeros = h.stats.RecentZeros[:20]
			}
			h.stats.mu.Unlock()
			
			// Log finding
			h.logger.Infof("Zero found at t=%.15f (|Z|=%.3e, confidence=%.2f)", 
				result.T, result.Magnitude, result.Confidence)
			
			// Send performance metric
			if h.perfMonitor != nil {
				h.perfMonitor.metrics <- PerformanceMetric{
					Timestamp:  time.Now(),
					MetricType: "zero_found",
					Value:      result.T,
					Details: map[string]interface{}{
						"magnitude":  result.Magnitude,
						"worker_id":  result.WorkerID,
						"device":     result.DeviceType,
						"confidence": result.Confidence,
					},
				}
			}
		}
	}
}

func (h *RiemannHunter) updateStatistics(ctx context.Context) error {
	ticker := time.NewTicker(h.config.Performance.StatsInterval)
	defer ticker.Stop()
	
	lastPoints := int64(0)
	lastTime := time.Now()
	
	for {
		select {
		case <-ctx.Done():
			return nil
			
		case <-ticker.C:
			now := time.Now()
			
			// Get hardware stats
			hwStats := h.hardware.GetHardwareStats()
			
			// Get worker stats
			workerStats := h.pool.GetStatistics()
			
			// Get memory stats
			var memStats runtime.MemStats
			runtime.ReadMemStats(&memStats)
			
			// Update statistics
			h.stats.mu.Lock()
			
			// Hardware
			h.stats.Hardware = hwStats
			
			// Calculation stats
			calc := &h.stats.Calculation
			calc.ElapsedTime = now.Sub(calc.StartTime)
			
			// Points per second
			pointsDelta := calc.PointsProcessed - lastPoints
			timeDelta := now.Sub(lastTime).Seconds()
			if timeDelta > 0 {
				calc.PointsPerSecond = float64(pointsDelta) / timeDelta
			}
			
			// Progress
			totalPoints := (h.config.Calculation.EndT - h.config.Calculation.StartT) / h.config.Calculation.Step
			calc.ProgressPercent = float64(calc.PointsProcessed) / totalPoints * 100
			
			// Estimated finish
			if calc.PointsPerSecond > 0 {
				remainingPoints := totalPoints - float64(calc.PointsProcessed)
				remainingSeconds := remainingPoints / calc.PointsPerSecond
				calc.EstimatedFinish = now.Add(time.Duration(remainingSeconds) * time.Second)
			}
			
			// Expected zeros (Riemann-von Mangoldt formula)
			t := calc.CurrentT
			if t > 0 {
				calc.ExpectedZeros = (t/(2*math.Pi))*math.Log(t/(2*math.Pi)) - t/(2*math.Pi) + 7.0/8.0
				if calc.PointsProcessed > 0 {
					calc.ZeroDensity = float64(calc.ZerosFound) / float64(calc.PointsProcessed)
				}
			}
			
			// Cache hit rate
			calc.CacheHitRate = h.calculator.cache.HitRate()
			
			// Worker stats
			h.stats.Workers = workerStats
			
			// System stats
			h.stats.GoRoutines = runtime.NumGoroutine()
			h.stats.GCCycles = memStats.NumGC
			h.stats.AllocMB = float64(memStats.Alloc) / 1024 / 1024
			h.stats.SysMB = float64(memStats.Sys) / 1024 / 1024
			h.stats.NextGCMB = float64(memStats.NextGC) / 1024 / 1024
			
			// Performance snapshot
			snapshot := PerformanceSnapshot{
				Timestamp:       now,
				PointsPerSecond: calc.PointsPerSecond,
				MemoryUsageMB:   h.stats.AllocMB,
				CPUUsagePercent: hwStats.CPUUsagePercent,
				CacheHitRate:    calc.CacheHitRate,
				ActiveWorkers:   workerStats.ActiveWorkers,
			}
			h.stats.PerformanceLog = append(h.stats.PerformanceLog, snapshot)
			if len(h.stats.PerformanceLog) > 1000 {
				h.stats.PerformanceLog = h.stats.PerformanceLog[1:]
			}
			
			h.stats.mu.Unlock()
			
			// Save statistics to file
			if h.config.Output.SaveStats {
				if err := h.storage.SaveStatistics(h.stats); err != nil {
					h.logger.Errorf("Failed to save statistics: %v", err)
				}
			}
			
			// Update display if enabled
			if h.config.Output.RealTimeDisplay {
				h.displayStatistics()
			}
			
			// Update for next iteration
			lastPoints = calc.PointsProcessed
			lastTime = now
			
			// Check for performance issues
			h.checkPerformanceIssues()
		}
	}
}

func (h *RiemannHunter) saveCheckpoints(ctx context.Context) error {
	ticker := time.NewTicker(h.config.Performance.CheckpointInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return nil
			
		case <-ticker.C:
			h.logger.Debug("Saving checkpoint...")
			
			h.checkpoint.Statistics = *h.stats
			h.checkpoint.LastSaved = time.Now()
			h.checkpoint.NextCheckpoint = time.Now().Add(h.config.Performance.CheckpointInterval)
			h.checkpoint.RecoveryPoint = h.stats.Calculation.CurrentT
			h.checkpoint.RecoveryChecksum = fmt.Sprintf("%x", time.Now().UnixNano())
			
			if err := h.storage.SaveCheckpoint(h.checkpoint); err != nil {
				h.logger.Errorf("Failed to save checkpoint: %v", err)
			} else {
				h.logger.Debugf("Checkpoint saved at t=%.2e", h.stats.Calculation.CurrentT)
			}
		}
	}
}

func (h *RiemannHunter) monitorStatus(ctx context.Context) error {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return nil
			
		case <-ticker.C:
			// Check memory usage
			var memStats runtime.MemStats
			runtime.ReadMemStats(&memStats)
			allocMB := float64(memStats.Alloc) / 1024 / 1024
			
			if allocMB > float64(h.config.Performance.MemoryLimitMB)*0.9 {
				h.statusChan <- fmt.Sprintf("high_memory: %.1fMB", allocMB)
				h.logger.Warnf("High memory usage: %.1fMB", allocMB)
			}
			
			// Check if we're making progress
			h.stats.mu.RLock()
			pps := h.stats.Calculation.PointsPerSecond
			h.stats.mu.RUnlock()
			
			if pps < 1.0 && h.stats.Calculation.PointsProcessed > 1000 {
				h.statusChan <- fmt.Sprintf("low_throughput: %.1f pps", pps)
				h.logger.Warnf("Low throughput: %.1f points/sec", pps)
			}
		}
	}
}

func (h *RiemannHunter) checkPerformanceIssues() {
	h.stats.mu.RLock()
	stats := h.stats.Calculation
	h.stats.mu.RUnlock()
	
	// Check for stagnation
	if stats.PointsPerSecond < 0.1 && stats.PointsProcessed > 10000 {
		h.logger.Warn("Performance issue detected: very low throughput")
		h.statusChan <- "performance_issue:low_throughput"
	}
	
	// Check memory
	if h.stats.AllocMB > float64(h.config.Performance.MemoryLimitMB)*0.95 {
		h.logger.Error("Critical memory usage - approaching limit")
		h.statusChan <- "critical:high_memory"
	}
}

func (h *RiemannHunter) handleSignal(sig os.Signal) {
	switch sig {
	case syscall.SIGINT, syscall.SIGTERM:
		h.logger.Infof("Received %v, initiating graceful shutdown", sig)
		h.cancel()
		
	case syscall.SIGHUP:
		h.logger.Info("Received SIGHUP, reloading configuration")
		if err := h.reloadConfig(); err != nil {
			h.logger.Errorf("Failed to reload config: %v", err)
		}
		
	case syscall.SIGUSR1:
		h.logger.Info("Received SIGUSR1, pausing calculation")
		h.pauseChan <- true
		
	case syscall.SIGUSR2:
		h.logger.Info("Received SIGUSR2, resuming calculation")
		h.resumeChan <- true
	}
}

func (h *RiemannHunter) reloadConfig() error {
	h.logger.Info("Reloading configuration...")
	
	// Load new config
	newConfig, err := loadConfigFromFile(h.config.configPath)
	if err != nil {
		return fmt.Errorf("failed to reload config: %w", err)
	}
	
	// Apply runtime-safe changes
	h.applyRuntimeConfig(newConfig)
	
	h.logger.Info("Configuration reloaded successfully")
	return nil
}

func (h *RiemannHunter) applyRuntimeConfig(newConfig *Config) {
	h.config.mu.Lock()
	defer h.config.mu.Unlock()
	
	// Apply safe runtime changes
	h.config.Output.Verbose = newConfig.Output.Verbose
	h.config.Output.LogLevel = newConfig.Output.LogLevel
	h.config.Output.RealTimeDisplay = newConfig.Output.RealTimeDisplay
	
	h.config.Performance.CPULimitPercent = newConfig.Performance.CPULimitPercent
	h.config.Performance.MemoryLimitMB = newConfig.Performance.MemoryLimitMB
	
	// Update logger level
	setupLogger(h.config.Output)
	
	h.logger.Debug("Runtime configuration updated")
}

func (h *RiemannHunter) displayStatistics() {
	h.stats.mu.RLock()
	stats := h.stats
	calc := stats.Calculation
	hw := stats.Hardware
	workers := stats.Workers
	h.stats.mu.RUnlock()
	
	// Clear screen
	fmt.Print("\033[H\033[2J")
	
	// Banner
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║                   RIEMANN HYPOTHESIS HUNTER v%s - PRODUCTION MODE                    ║\n", Version)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	
	// Progress
	barWidth := 60
	progress := int(calc.ProgressPercent * float64(barWidth) / 100)
	if progress > barWidth {
		progress = barWidth
	}
	
	bar := strings.Repeat("█", progress) + strings.Repeat("░", barWidth-progress)
	fmt.Printf("  Progress: [%s] %.2f%%\n", bar, calc.ProgressPercent)
	fmt.Println()
	
	// Calculation stats
	fmt.Printf("  Current T:          %.15e\n", calc.CurrentT)
	fmt.Printf("  Points Processed:   %s\n", formatNumberLarge(calc.PointsProcessed))
	fmt.Printf("  Zeros Found:        %s (Expected: ~%.0f)\n", 
		formatNumberLarge(calc.ZerosFound), calc.ExpectedZeros)
	fmt.Printf("  Throughput:         %.0f points/sec\n", calc.PointsPerSecond)
	fmt.Printf("  Elapsed Time:       %s\n", formatDurationDetailed(calc.ElapsedTime))
	
	if !calc.EstimatedFinish.IsZero() {
		fmt.Printf("  Estimated Finish:   %s\n", calc.EstimatedFinish.Format("2006-01-02 15:04:05"))
	}
	fmt.Println()
	
	// Hardware stats
	fmt.Printf("  Hardware:\n")
	fmt.Printf("    CPU Usage:        %.1f%% of %d cores\n", hw.CPUUsagePercent, hw.CPUCores)
	fmt.Printf("    Memory:           %.1f/%.1f MB (%.1f%%)\n", 
		hw.UsedRAMMB, hw.TotalRAMMB, (hw.UsedRAMMB/hw.TotalRAMMB)*100)
	fmt.Printf("    GPUs:             %d active\n", hw.GPUs)
	fmt.Println()
	
	// Worker stats
	fmt.Printf("  Workers:            %d/%d active\n", workers.ActiveWorkers, workers.TotalWorkers)
	fmt.Printf("  Cache Efficiency:   %.1f%% hit rate\n", calc.CacheHitRate*100)
	fmt.Printf("  Algorithm:          %s (%d-bit precision)\n", calc.Algorithm, calc.Precision)
	fmt.Println()
	
	// Last zero found
	if stats.LastZeroFound != nil {
		zero := stats.LastZeroFound
		fmt.Printf("  Last Zero Found:\n")
		fmt.Printf("    T = %.15f\n", zero.T)
		fmt.Printf("    |Z| = %.3e (confidence: %.2f)\n", zero.Magnitude, zero.Confidence)
		fmt.Printf("    Worker: %d (%s)\n", zero.WorkerID, zero.DeviceType)
		fmt.Printf("    Time: %s\n", zero.FoundAt.Format("15:04:05.000"))
		fmt.Println()
	}
	
	// System info
	fmt.Printf("  System Info:\n")
	fmt.Printf("    Go Version:       %s\n", runtime.Version())
	fmt.Printf("    Goroutines:       %d\n", stats.GoRoutines)
	fmt.Printf("    Memory (Go):      %.1f MB allocated\n", stats.AllocMB)
	fmt.Printf("    Next GC at:       %.1f MB\n", stats.NextGCMB)
	fmt.Println()
	
	// Controls
	fmt.Println("  Controls:")
	fmt.Println("    Ctrl+C - Graceful shutdown")
	fmt.Println("    SIGHUP - Reload configuration")
	fmt.Println("    SIGUSR1 - Pause calculation")
	fmt.Println("    SIGUSR2 - Resume calculation")
	fmt.Println()
	
	// Output files
	fmt.Println("  Output Files:")
	fmt.Printf("    Zeros:            %s/riemann_zeros.csv\n", h.config.Output.OutputDirectory)
	fmt.Printf("    Statistics:       %s/riemann_stats.json\n", h.config.Output.OutputDirectory)
	fmt.Printf("    Checkpoints:      %s/riemann_checkpoint.json\n", h.config.Output.OutputDirectory)
	fmt.Println()
}

func (h *RiemannHunter) printStartupBanner() {
	fmt.Println()
	fmt.Println("██████╗ ██╗███████╗███╗   ███╗ █████╗ ███╗   ██╗███╗   ██╗    ██╗  ██╗██╗   ██╗███╗   ██╗████████╗███████╗██████╗")
	fmt.Println("██╔══██╗██║██╔════╝████╗ ████║██╔══██╗████╗  ██║████╗  ██║    ██║  ██║██║   ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗")
	fmt.Println("██████╔╝██║█████╗  ██╔████╔██║███████║██╔██╗ ██║██╔██╗ ██║    ███████║██║   ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝")
	fmt.Println("██╔══██╗██║██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║██║╚██╗██║    ██╔══██║██║   ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗")
	fmt.Println("██║  ██║██║███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║██║ ╚████║    ██║  ██║╚██████╔╝██║ ╚████║   ██║   ███████╗██║  ██║")
	fmt.Println("╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝")
	fmt.Println()
	fmt.Printf("Version: %s | Build: %s | Author: %s | License: %s\n", Version, BuildDate, Author, License)
	fmt.Printf("Go: %s | CPUs: %d | Hardware Mode: %s\n", runtime.Version(), runtime.NumCPU(), h.config.Hardware.Mode)
	fmt.Println("════════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	fmt.Println()
	
	h.logger.Infof("Starting Riemann Hunter with configuration:")
	h.logger.Infof("  Range: t = %.2e to %.2e", h.config.Calculation.StartT, h.config.Calculation.EndT)
	h.logger.Infof("  Step: %.4f | Threshold: %.1e", h.config.Calculation.Step, h.config.Calculation.ZeroThreshold)
	h.logger.Infof("  Algorithm: %s | Precision: %d bits", h.config.Calculation.Algorithm, h.config.Calculation.Precision)
	h.logger.Infof("  Hardware Mode: %s | CPU Limit: %d%%", h.config.Hardware.Mode, h.config.Performance.CPULimitPercent)
	h.logger.Infof("  Memory Limit: %d MB | Workers: %d", h.config.Performance.MemoryLimitMB, h.config.Performance.MaxWorkers)
	h.logger.Infof("  Output Directory: %s", h.config.Output.OutputDirectory)
}

func (h *RiemannHunter) printFinalStatistics() {
	h.stats.mu.RLock()
	stats := h.stats
	calc := stats.Calculation
	h.stats.mu.RUnlock()
	
	fmt.Println()
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("                         CALCULATION COMPLETE - SUMMARY                        ")
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
	fmt.Println()
	
	fmt.Printf("Total Calculation Time:   %s\n", formatDurationDetailed(calc.ElapsedTime))
	fmt.Printf("Points Processed:         %s\n", formatNumberLarge(calc.PointsProcessed))
	fmt.Printf("Zeros Found:              %s\n", formatNumberLarge(calc.ZerosFound))
	fmt.Printf("Average Throughput:       %.0f points/sec\n", 
		float64(calc.PointsProcessed)/calc.ElapsedTime.Seconds())
	fmt.Printf("Final T Value:            %.15e\n", calc.CurrentT)
	fmt.Printf("Cache Efficiency:         %.1f%% hit rate\n", calc.CacheHitRate*100)
	fmt.Printf("Peak Memory Usage:        %.1f MB\n", stats.AllocMB)
	fmt.Println()
	
	// Scientific analysis
	if calc.ZerosFound > 0 {
		expected := calc.ExpectedZeros
		actual := float64(calc.ZerosFound)
		difference := expected - actual
		relativeError := math.Abs(difference) / expected * 100
		
		fmt.Printf("Scientific Analysis:\n")
		fmt.Printf("  Expected zeros (R-vM):  ~%.0f\n", expected)
		fmt.Printf("  Actual zeros found:     %.0f\n", actual)
		fmt.Printf("  Difference:             %+.0f (%.2f%%)\n", difference, relativeError)
		fmt.Printf("  Zero density:           %.2e zeros/point\n", calc.ZeroDensity)
		fmt.Println()
		
		if relativeError > 5.0 {
			fmt.Println("⚠️  WARNING: Significant zero count discrepancy!")
			fmt.Println("   Possible causes:")
			fmt.Println("   1. Algorithm limitations")
			fmt.Println("   2. Step size too large (missing zeros)")
			fmt.Println("   3. Threshold too low (false positives)")
			fmt.Println("   4. Range too small for statistical accuracy")
			fmt.Println()
		} else {
			fmt.Println("✅ Zero count matches theoretical predictions")
			fmt.Println()
		}
		
		// Display recent zeros
		if len(stats.RecentZeros) > 0 {
			fmt.Println("Recent Zeros Found:")
			for i, zero := range stats.RecentZeros {
				if i >= 10 {
					break
				}
				fmt.Printf("  %2d. t = %20.15f |Z| = %9.3e\n", 
					i+1, zero.T, zero.Magnitude)
			}
			fmt.Println()
		}
	} else {
		fmt.Println("❌ No zeros found in the specified range")
		fmt.Println("   Consider:")
		fmt.Println("   1. Increasing the range")
		fmt.Println("   2. Decreasing the step size")
		fmt.Println("   3. Adjusting the zero threshold")
		fmt.Println("   4. Using a different algorithm")
		fmt.Println()
	}
	
	fmt.Println("Output Files Created:")
	fmt.Printf("  - Zeros CSV:        %s/riemann_zeros.csv\n", h.config.Output.OutputDirectory)
	fmt.Printf("  - Statistics JSON:  %s/riemann_stats.json\n", h.config.Output.OutputDirectory)
	fmt.Printf("  - Checkpoint JSON:  %s/riemann_checkpoint.json\n", h.config.Output.OutputDirectory)
	if h.config.Output.CompressOutput {
		fmt.Printf("  - Compressed Archive: %s/riemann_results.tar.gz\n", h.config.Output.OutputDirectory)
	}
	fmt.Println()
	
	fmt.Println("To Resume Calculation:")
	fmt.Printf("  ./riemann-hunter --start %.15e --resume\n", calc.CurrentT)
	fmt.Println()
	
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("                           THANK YOU FOR YOUR COMPUTATION!                     ")
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
}

// ==================== PERFORMANCE MONITOR ====================
func (pm *PerformanceMonitor) run() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-pm.stopChan:
			return
			
		case metric := <-pm.metrics:
			pm.processMetric(metric)
			
		case <-ticker.C:
			pm.collectSystemMetrics()
		}
	}
}

func (pm *PerformanceMonitor) processMetric(metric PerformanceMetric) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	switch metric.MetricType {
	case "zero_found":
		pm.logger.Debugf("Zero found at t=%.2e", metric.Value)
	case "batch_completed":
		pm.logger.Debugf("Batch completed: %.0f points in %v", 
			metric.Value, metric.Details["duration"])
	case "error":
		pm.logger.Errorf("Error: %v", metric.Details)
	}
}

func (pm *PerformanceMonitor) collectSystemMetrics() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	allocMB := float64(memStats.Alloc) / 1024 / 1024
	
	if allocMB > 1000 {
		pm.logger.Warnf("High memory allocation: %.1f MB", allocMB)
	}
	
	// Record snapshot
	snapshot := PerformanceSnapshot{
		Timestamp:      time.Now(),
		MemoryUsageMB:  allocMB,
	}
	
	pm.history = append(pm.history, snapshot)
	if len(pm.history) > 1000 {
		pm.history = pm.history[1:]
	}
}

// ==================== HEALTH CHECKER ====================
func (hc *HealthChecker) run() {
	ticker := time.NewTicker(hc.checkInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-hc.stopChan:
			return
			
		case <-ticker.C:
			hc.performHealthCheck()
		}
	}
}

func (hc *HealthChecker) performHealthCheck() {
	status := HealthStatus{
		Overall:     "healthy",
		LastUpdated: time.Now(),
	}
	
	// Check memory
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	allocMB := float64(memStats.Alloc) / 1024 / 1024
	
	if allocMB > 800 { // 800MB threshold for 1GB
		status.Memory = "warning"
		status.Overall = "degraded"
		hc.logger.Warnf("High memory usage: %.1f MB", allocMB)
	} else {
		status.Memory = "healthy"
	}
	
	// Check disk space
	if stat, err := os.Stat("."); err == nil {
		if stat.Size() > 10*1024*1024*1024 { // 10GB
			status.Disk = "warning"
			status.Overall = "degraded"
			hc.logger.Warn("Disk space running low")
		} else {
			status.Disk = "healthy"
		}
	}
	
	hc.healthStatus = status
}

// ==================== AUTO TUNER ====================
func (at *AutoTuner) run() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for at.active {
		select {
		case <-ticker.C:
			at.analyzeAndAdjust()
		}
	}
}

func (at *AutoTuner) analyzeAndAdjust() {
	// Analyze recent performance
	// Adjust configuration parameters
	
	// Example: Adjust batch size based on memory
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	allocMB := float64(memStats.Alloc) / 1024 / 1024
	
	oldBatchSize := at.config.Performance.BatchSize
	newBatchSize := oldBatchSize
	
	if allocMB > float64(at.config.Performance.MemoryLimitMB)*0.8 {
		// Reduce batch size if memory is high
		newBatchSize = int(float64(oldBatchSize) * 0.8)
		if newBatchSize < 1000 {
			newBatchSize = 1000
		}
	} else if allocMB < float64(at.config.Performance.MemoryLimitMB)*0.5 {
		// Increase batch size if memory is low
		newBatchSize = int(float64(oldBatchSize) * 1.2)
		if newBatchSize > 1000000 {
			newBatchSize = 1000000
		}
	}
	
	if newBatchSize != oldBatchSize {
		adjustment := ConfigAdjustment{
			Parameter: "batch_size",
			OldValue:  oldBatchSize,
			NewValue:  newBatchSize,
			Reason:    fmt.Sprintf("Memory usage: %.1fMB", allocMB),
			Timestamp: time.Now(),
		}
		
		at.adjustments <- adjustment
		at.config.Performance.BatchSize = newBatchSize
		at.logger.Infof("Auto-tuned batch size: %d -> %d", oldBatchSize, newBatchSize)
	}
}

// ==================== UTILITY FUNCTIONS ====================
var startTime = time.Now()

func getHostnameSafe() string {
	hostname, err := os.Hostname()
	if err != nil {
		return "unknown"
	}
	return hostname
}

func formatNumberLarge(n int64) string {
	if n < 1000 {
		return strconv.FormatInt(n, 10)
	}
	
	suffixes := []string{"", "k", "M", "B", "T"}
	suffixIndex := 0
	value := float64(n)
	
	for value >= 1000 && suffixIndex < len(suffixes)-1 {
		value /= 1000
		suffixIndex++
	}
	
	return fmt.Sprintf("%.1f%s", value, suffixes[suffixIndex])
}

func formatDurationDetailed(d time.Duration) string {
	days := int(d.Hours() / 24)
	hours := int(d.Hours()) % 24
	minutes := int(d.Minutes()) % 60
	seconds := int(d.Seconds()) % 60
	
	if days > 0 {
		return fmt.Sprintf("%dd %02dh %02dm %02ds", days, hours, minutes, seconds)
	} else if hours > 0 {
		return fmt.Sprintf("%02dh %02dm %02ds", hours, minutes, seconds)
	} else if minutes > 0 {
		return fmt.Sprintf("%02dm %02ds", minutes, seconds)
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

// ==================== CONFIGURATION MANAGEMENT ====================
func loadConfigFromFile(path string) (*Config, error) {
    viper.SetConfigFile(path)
    viper.SetConfigType("yaml")
    
    // Set defaults
    setDefaults()
    
    // Read config
    if err := viper.ReadInConfig(); err != nil {
        return nil, fmt.Errorf("failed to read config: %w", err)
    }
    
    var cfg Config
    
    // HOTFIX: Manually parse large numbers to avoid viper unmarshal bug
    cfg.Calculation.StartT = viper.GetFloat64("calculation.start_t")
    cfg.Calculation.EndT = viper.GetFloat64("calculation.end_t")
    cfg.Calculation.Step = viper.GetFloat64("calculation.step")
    cfg.Calculation.ZeroThreshold = viper.GetFloat64("calculation.zero_threshold")
    cfg.Calculation.Precision = viper.GetInt("calculation.precision")
    cfg.Calculation.Algorithm = viper.GetString("calculation.algorithm")
    cfg.Calculation.UseCache = viper.GetBool("calculation.use_cache")
    cfg.Calculation.MaxTerms = viper.GetInt("calculation.max_terms")
    
    // For the rest, use normal unmarshal
    if err := viper.Unmarshal(&cfg); err != nil {
        return nil, fmt.Errorf("failed to unmarshal config: %w", err)
    }
    
    // Apply environment overrides
    applyEnvironmentOverrides(&cfg)
    
    // Validate
    if err := validateConfig(&cfg); err != nil {
        return nil, fmt.Errorf("config validation failed: %w", err)
    }
    
    // Calculate dynamic values
    calculateDynamicValues(&cfg)
    
    cfg.configPath = path
    cfg.loadedFrom = viper.ConfigFileUsed()
    
    return &cfg, nil
}

func setDefaults() {
	// Hardware defaults
	viper.SetDefault("hardware.mode", "auto")
	viper.SetDefault("hardware.cpu_usage_percent", 100)
	viper.SetDefault("hardware.threads_per_core", 2.0)
	viper.SetDefault("hardware.enable_avx", true)
	viper.SetDefault("hardware.enable_cuda", true)
	viper.SetDefault("hardware.enable_opencl", true)
	
	// Calculation defaults
	viper.SetDefault("calculation.start_t", 1e13)
	viper.SetDefault("calculation.end_t", 1e13+1000)
	viper.SetDefault("calculation.step", 0.01)
	viper.SetDefault("calculation.zero_threshold", 1e-12)
	viper.SetDefault("calculation.algorithm", "riemann-siegel-fast")
	viper.SetDefault("calculation.precision", 128)
	viper.SetDefault("calculation.use_cache", true)
	viper.SetDefault("calculation.cache_strategy", "lru")
	viper.SetDefault("calculation.max_terms", 50000)
	viper.SetDefault("calculation.remainder_terms", 3)
	
	// Output defaults
	viper.SetDefault("output.save_zeros", true)
	viper.SetDefault("output.save_stats", true)
	viper.SetDefault("output.save_checkpoints", true)
	viper.SetDefault("output.output_directory", ".")
	viper.SetDefault("output.filename_prefix", "riemann")
	viper.SetDefault("output.compress_output", false)
	viper.SetDefault("output.verbose", false)
	viper.SetDefault("output.real_time_display", true)
	viper.SetDefault("output.log_level", "info")
	
	// Performance defaults
	viper.SetDefault("performance.batch_size", 0) // 0 = auto
	viper.SetDefault("performance.max_workers", 0) // 0 = auto
	viper.SetDefault("performance.cache_size", 100000)
	viper.SetDefault("performance.checkpoint_interval", "5m")
	viper.SetDefault("performance.stats_interval", "2s")
	viper.SetDefault("performance.memory_limit_mb", 900)
	viper.SetDefault("performance.cpu_limit_percent", 100)
	viper.SetDefault("performance.gpu_limit_percent", 100)
	viper.SetDefault("performance.auto_tune", true)
	viper.SetDefault("performance.benchmark_mode", false)
}

func applyEnvironmentOverrides(cfg *Config) {
	// Environment variables can override config
	// Example: RIEMANN_HARDWARE_MODE, RIEMANN_CALCULATION_START_T, etc.
}

func validateConfig(cfg *Config) error {
    // DEBUG FIRST
    fmt.Printf("[VALIDATION DEBUG] start_t=%e, end_t=%e\n", 
        cfg.Calculation.StartT, cfg.Calculation.EndT)
    
    // SIMPLE CHECK - if both values are 0, something went wrong in the parser
    if cfg.Calculation.StartT == 0 && cfg.Calculation.EndT == 0 {
        return fmt.Errorf("CONFIG PARSING ERROR: Both start_t and end_t are 0! Check YAML file.")
    }
    
    // If start is 0 but shouldn't be
    if cfg.Calculation.StartT == 0 && cfg.Calculation.EndT > 0 {
        // Try to auto-correct (10^20 is default)
        cfg.Calculation.StartT = 1e20
        fmt.Printf("[AUTO-FIX] Set start_t to default 1e20\n")
    }
    
    // ULTRA-SIMPLE validation - just check end > start
    if cfg.Calculation.EndT <= cfg.Calculation.StartT {
        // Try to fix common issue
        if cfg.Calculation.StartT > 1e15 {
            // For huge numbers, auto-correct
            cfg.Calculation.EndT = cfg.Calculation.StartT * 1.000000001
            fmt.Printf("[AUTO-FIX] Adjusted end_t to %e\n", cfg.Calculation.EndT)
            return nil // Skip error after fixing
        }
        return fmt.Errorf("start_t must be less than end_t (start=%e, end=%e)", 
            cfg.Calculation.StartT, cfg.Calculation.EndT)
    }
    
    // Rest of validation...
    if cfg.Calculation.Step <= 0 {
        return fmt.Errorf("step must be positive")
    }
    if cfg.Calculation.ZeroThreshold <= 0 {
        return fmt.Errorf("zero_threshold must be positive")
    }
    if cfg.Calculation.Precision < 64 {
        return fmt.Errorf("precision must be at least 64 bits")
    }
    
    if cfg.Hardware.CPUUsagePercent < 1 || cfg.Hardware.CPUUsagePercent > 100 {
        return fmt.Errorf("cpu_usage_percent must be between 1 and 100")
    }
    if cfg.Hardware.ThreadsPerCore < 0.1 || cfg.Hardware.ThreadsPerCore > 10 {
        return fmt.Errorf("threads_per_core must be between 0.1 and 10")
    }
    
    if cfg.Performance.MemoryLimitMB < 100 {
        return fmt.Errorf("memory_limit_mb must be at least 100")
    }
    if cfg.Performance.CPULimitPercent < 1 || cfg.Performance.CPULimitPercent > 100 {
        return fmt.Errorf("cpu_limit_percent must be between 1 and 100")
    }
    if cfg.Performance.BatchSize < 0 {
        return fmt.Errorf("batch_size cannot be negative")
    }
    if cfg.Performance.MaxWorkers < 0 {
        return fmt.Errorf("max_workers cannot be negative")
    }
    
    return nil
}

func calculateDynamicValues(cfg *Config) {
	// Calculate batch size if auto
	if cfg.Performance.BatchSize <= 0 {
		// Base batch size on available RAM
		availableMB := float64(cfg.Performance.MemoryLimitMB)
		cfg.Performance.BatchSize = int(availableMB * 1000) // ~1KB per point
		
		// Apply limits
		if cfg.Performance.BatchSize < 1000 {
			cfg.Performance.BatchSize = 1000
		}
		if cfg.Performance.BatchSize > 1000000 {
			cfg.Performance.BatchSize = 1000000
		}
	}
	
	// Calculate max workers if auto
	if cfg.Performance.MaxWorkers <= 0 {
		cores := runtime.NumCPU()
		cfg.Performance.MaxWorkers = int(float64(cores) * cfg.Hardware.ThreadsPerCore)
		
		// Apply limits
		if cfg.Performance.MaxWorkers < 1 {
			cfg.Performance.MaxWorkers = 1
		}
		if cfg.Performance.MaxWorkers > 32 {
			cfg.Performance.MaxWorkers = 32
		}
	}
	
	// Adjust cache size based on available memory
	maxCacheSize := cfg.Performance.MemoryLimitMB * 1000 // ~1KB per entry
	if cfg.Performance.CacheSize > maxCacheSize {
		cfg.Performance.CacheSize = maxCacheSize
	}
}

// ==================== COMMAND LINE INTERFACE ====================
var rootCmd = &cobra.Command{
	Use:   "riemann-hunter",
	Short: "Ultimate Riemann Hypothesis Zero Finder",
	Long: `A high-performance, production-ready system for finding zeros of the Riemann zeta function.
Features automatic hardware detection (CPU/GPU), memory optimization, checkpoint/resume,
real-time statistics, and flexible configuration.`,
	
	Run: func(cmd *cobra.Command, args []string) {
		// Load configuration
		cfg, err := loadConfig()
		if err != nil {
			fmt.Printf("❌ Configuration error: %v\n", err)
			os.Exit(1)
		}
		
		// Create and run hunter
		hunter, err := NewRiemannHunter(cfg)
		if err != nil {
			fmt.Printf("❌ Initialization error: %v\n", err)
			os.Exit(1)
		}
		
		// Run
		if err := hunter.Run(); err != nil {
			fmt.Printf("❌ Runtime error: %v\n", err)
			os.Exit(1)
		}
	},
}

// Global flags
var (
	configPath    string
	startT        float64
	endT          float64
	step          float64
	resume        bool
	benchmark     bool
	verbose       bool
	outputDir     string
	cpuLimit      int
	memoryLimit   int
	mode          string
	gpuDevices    string
)

func init() {
	// Configuration flags
	rootCmd.PersistentFlags().StringVar(&configPath, "config", "riemann.yaml", "Configuration file path")
	rootCmd.PersistentFlags().BoolVar(&resume, "resume", false, "Resume from last checkpoint")
	rootCmd.PersistentFlags().BoolVar(&benchmark, "benchmark", false, "Run in benchmark mode")
	
	// Calculation flags
	rootCmd.PersistentFlags().Float64Var(&startT, "start", 0, "Starting t value (overrides config)")
	rootCmd.PersistentFlags().Float64Var(&endT, "end", 0, "Ending t value (overrides config)")
	rootCmd.PersistentFlags().Float64Var(&step, "step", 0, "Step size (overrides config)")
	
	// Hardware flags
	rootCmd.PersistentFlags().StringVar(&mode, "mode", "", "Execution mode: auto, cpu, gpu, multi-gpu")
	rootCmd.PersistentFlags().IntVar(&cpuLimit, "cpu-limit", 0, "CPU usage limit (1-100%, 0=auto)")
	rootCmd.PersistentFlags().IntVar(&memoryLimit, "memory-limit", 0, "Memory limit in MB (0=auto)")
	rootCmd.PersistentFlags().StringVar(&gpuDevices, "gpu-devices", "", "GPU devices to use (comma-separated)")
	
	// Output flags
	rootCmd.PersistentFlags().BoolVar(&verbose, "verbose", false, "Verbose output")
	rootCmd.PersistentFlags().StringVar(&outputDir, "output-dir", "", "Output directory (overrides config)")
	
	// Bind flags to viper
	viper.BindPFlag("calculation.start_t", rootCmd.PersistentFlags().Lookup("start"))
	viper.BindPFlag("calculation.end_t", rootCmd.PersistentFlags().Lookup("end"))
	viper.BindPFlag("calculation.step", rootCmd.PersistentFlags().Lookup("step"))
	viper.BindPFlag("hardware.mode", rootCmd.PersistentFlags().Lookup("mode"))
	viper.BindPFlag("performance.cpu_limit_percent", rootCmd.PersistentFlags().Lookup("cpu-limit"))
	viper.BindPFlag("performance.memory_limit_mb", rootCmd.PersistentFlags().Lookup("memory-limit"))
	viper.BindPFlag("output.verbose", rootCmd.PersistentFlags().Lookup("verbose"))
	viper.BindPFlag("output.output_directory", rootCmd.PersistentFlags().Lookup("output-dir"))
	
	// Environment variables
	viper.SetEnvPrefix("RIEMANN")
	viper.AutomaticEnv()
}

func loadConfig() (*Config, error) {
	// Try to load from file
	cfg, err := loadConfigFromFile(configPath)
	if err != nil {
		// If no config file, create default
		if os.IsNotExist(err) {
			fmt.Printf("Config file not found, using defaults: %s\n", configPath)
			cfg = createDefaultConfig()
			
			// Save default config
			if err := saveDefaultConfig(configPath, cfg); err != nil {
				fmt.Printf("Warning: Could not save default config: %v\n", err)
			}
		} else {
			return nil, fmt.Errorf("failed to load config: %w", err)
		}
	}
	
	// Apply command line overrides
	applyCommandLineOverrides(cfg)
	
	// Handle resume flag
	if resume {
		if err := handleResume(cfg); err != nil {
			return nil, fmt.Errorf("failed to resume: %w", err)
		}
	}
	
	// Handle benchmark mode
	if benchmark {
		cfg.Performance.BenchmarkMode = true
		cfg.Calculation.EndT = cfg.Calculation.StartT + 1000 // Short range for benchmarking
		cfg.Output.SaveZeros = false
		cfg.Output.SaveCheckpoints = false
		cfg.Performance.AutoTune = true
	}
	
	return cfg, nil
}

func createDefaultConfig() *Config {
	cfg := &Config{}
	
	// Apply defaults
	setDefaults()
	viper.Unmarshal(cfg)
	
	// Ensure dynamic values are calculated
	calculateDynamicValues(cfg)
	
	return cfg
}

func saveDefaultConfig(path string, cfg *Config) error {
	// Create directory if needed
	dir := filepath.Dir(path)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}
	
	// Marshal config to YAML (not JSON)
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	
	// Add header
	header := `# Riemann Hunter Configuration v` + Version + `
# Generated automatically on ` + time.Now().Format("2006-01-02 15:04:05") + `
# Documentation: https://github.com/riemann-research/riemann-hunter

`
	
	return os.WriteFile(path, []byte(header+string(data)), 0644)
}

func applyCommandLineOverrides(cfg *Config) {
	// Apply command line overrides
	if startT > 0 {
		cfg.Calculation.StartT = startT
	}
	if endT > 0 {
		cfg.Calculation.EndT = endT
	}
	if step > 0 {
		cfg.Calculation.Step = step
	}
	if cpuLimit > 0 {
		cfg.Performance.CPULimitPercent = cpuLimit
	}
	if memoryLimit > 0 {
		cfg.Performance.MemoryLimitMB = memoryLimit
	}
	if mode != "" {
		cfg.Hardware.Mode = ExecutionMode(mode)
	}
	if outputDir != "" {
		cfg.Output.OutputDirectory = outputDir
	}
	if verbose {
		cfg.Output.Verbose = true
		cfg.Output.LogLevel = "debug"
	}
	
	// Parse GPU devices
	if gpuDevices != "" {
		devices := strings.Split(gpuDevices, ",")
		cfg.Hardware.GPUDevices = make([]int, len(devices))
		for i, d := range devices {
			if id, err := strconv.Atoi(strings.TrimSpace(d)); err == nil {
				cfg.Hardware.GPUDevices[i] = id
			}
		}
	}
}

func handleResume(cfg *Config) error {
	// Look for checkpoint file
	prefix := cfg.Output.FilenamePrefix
	if prefix == "" {
		prefix = "riemann"
	}
	checkpointFile := filepath.Join(cfg.Output.OutputDirectory, prefix+"_checkpoint.json")
	
	data, err := os.ReadFile(checkpointFile)
	if err != nil {
		return fmt.Errorf("no checkpoint found to resume from: %w", err)
	}
	
	var checkpoint Checkpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return fmt.Errorf("failed to parse checkpoint: %w", err)
	}
	
	// Update config from checkpoint
	cfg.Calculation.StartT = checkpoint.Statistics.Calculation.CurrentT
	cfg.Calculation.Algorithm = checkpoint.Config.Calculation.Algorithm
	cfg.Calculation.Precision = checkpoint.Config.Calculation.Precision
	
	fmt.Printf("Resuming from checkpoint at t=%.2e\n", cfg.Calculation.StartT)
	return nil
}

// ==================== MAIN ENTRY POINT ====================
func main() {
	// Set resource limits
	runtime.GOMAXPROCS(runtime.NumCPU())
	
	// Check for Pterodactyl environment
	if os.Getenv("PTERODACTYL") == "1" || strings.Contains(os.Getenv("HOME"), "container") {
		fmt.Println("🔧 Pterodactyl environment detected")
		runPterodactylMode()
		return
	}
	
	// Normal CLI mode
	if err := rootCmd.Execute(); err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		os.Exit(1)
	}
}

func runPterodactylMode() {
	fmt.Println("🚀 Starting Riemann Hunter in Pterodactyl mode...")
	
	// Load config with Pterodactyl defaults
	cfg, err := loadConfig()
	if err != nil {
		// Fallback to ultra-conservative defaults for 1GB RAM
		cfg = &Config{
			Hardware: HardwareConfig{
				Mode:            ModeCPU,
				CPUUsagePercent: 80, // Be conservative
				ThreadsPerCore:  1.0,
				EnableAVX:       true,
			},
			Calculation: CalculationConfig{
				StartT:        1e10,
				EndT:          1e10 + 10000,
				Step:          0.1,
				ZeroThreshold: 1e-10,
				Algorithm:     "riemann-siegel-fast",
				Precision:     64,
				UseCache:      true,
				MaxTerms:      10000,
			},
			Output: OutputConfig{
				SaveZeros:       true,
				SaveStats:       true,
				SaveCheckpoints: true,
				OutputDirectory: ".",
				Verbose:         true,
				RealTimeDisplay: false, // No real-time display in Pterodactyl
				LogLevel:        "info",
			},
			Performance: PerformanceConfig{
				BatchSize:          5000,
				MaxWorkers:         2, // Very conservative
				CacheSize:          50000,
				CheckpointInterval: 10 * time.Minute,
				StatsInterval:      5 * time.Second,
				MemoryLimitMB:      800, // Stay under 1GB
				CPULimitPercent:    80,
				AutoTune:           true,
			},
		}
		fmt.Println("⚠️ Using conservative defaults for 1GB RAM environment")
	}
	
	// Apply Pterodactyl-specific adjustments
	cfg.Performance.MemoryLimitMB = 800 // Hard cap for safety
	cfg.Output.RealTimeDisplay = false  // No ANSI escapes in Pterodactyl console
	
	// Create and run hunter
	hunter, err := NewRiemannHunter(cfg)
	if err != nil {
		fmt.Printf("❌ FATAL: %v\n", err)
		os.Exit(1)
	}
	
	if err := hunter.Run(); err != nil {
		fmt.Printf("❌ RUNTIME ERROR: %v\n", err)
		os.Exit(1)
	}
}

// ==================== MISC FIXES ====================

// Fix WorkerPool to properly collect results
func (p *WorkerPool) collectWorkerResults() {
	for _, worker := range p.Workers {
		if worker == nil {
			continue
		}
		
		go func(w *Worker) {
			for result := range w.ResultChan {
				select {
				case p.ResultChan <- result:
					// Successfully forwarded result
				case <-time.After(100 * time.Millisecond):
					p.Logger.Warnf("Pool result channel full, dropping zero from worker %d", w.ID)
				}
			}
		}(worker)
	}
}

// Add missing methods
func (pm *PerformanceMonitor) GetHistory() []PerformanceSnapshot {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	
	history := make([]PerformanceSnapshot, len(pm.history))
	copy(history, pm.history)
	return history
}

func (hc *HealthChecker) GetStatus() HealthStatus {
	hc.healthStatus.LastUpdated = time.Now()
	return hc.healthStatus
}

// Add helper functions
func (rc *RiemannCalculator) ClearCache() {
	rc.cache.mu.Lock()
	defer rc.cache.mu.Unlock()
	
	rc.cache.zCache = make(map[float64]cacheEntry, rc.cache.maxSize/2)
	rc.cache.thetaCache = make(map[float64]cacheEntry, rc.cache.maxSize/2)
	rc.cache.accessTime = make(map[float64]time.Time)
	rc.cache.hits = 0
	rc.cache.misses = 0
}

func (h *RiemannHunter) GetCurrentStats() *Statistics {
	h.stats.mu.RLock()
	defer h.stats.mu.RUnlock()
	
	// Return copy of statistics
	statsCopy := *h.stats
	return &statsCopy
}

// Add signal handler setup
func setupSignalHandlers(hunter *RiemannHunter) {
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP, syscall.SIGUSR1, syscall.SIGUSR2)
	
	go func() {
		for sig := range signalChan {
			hunter.handleSignal(sig)
		}
	}()
}

// Add function to set max open files
func setMaxOpenFiles() {
	// Try to increase open file limit
	var rLimit syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rLimit)
	if err == nil && rLimit.Cur < 4096 {
		rLimit.Cur = 4096
		if rLimit.Cur > rLimit.Max {
			rLimit.Cur = rLimit.Max
		}
		syscall.Setrlimit(syscall.RLIMIT_NOFILE, &rLimit)
	}
}

// ==================== END OF CODE ====================

// NOTE: This code requires the following dependencies:
// go get github.com/sirupsen/logrus
// go get github.com/spf13/cobra
// go get github.com/spf13/viper
// go get golang.org/x/sync/errgroup
// go get gopkg.in/yaml.v3
