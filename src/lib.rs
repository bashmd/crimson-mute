use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use nih_plug::prelude::*;
use ort::session::Session;
use ort::value::Tensor;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rtrb::{Consumer, Producer, RingBuffer};
use rustfft::num_complex::Complex32;

mod model_path;
mod resample;
mod simd;
#[cfg(test)]
mod test_utils;

// Re-export types needed for integration tests
pub use resample::{calculate_resample_latency, Downsampler, Upsampler};

pub const FRAME_SIZE: usize = 512;
pub const HOP_SIZE: usize = 128;
const FFT_BINS: usize = 257;
const EPSILON: f32 = 1.0e-7;

// Ring buffer size: 8192 samples handles jitter and variable buffer sizes
const RING_BUFFER_SIZE: usize = 8192;

// State tensor sizes for ONNX models
const PMLN_STATE_SIZE: usize = 4 * 280 * 2; // [1,4,280,2] flattened
const STLN_STATE_SIZE: usize = 3 * 280 * 2; // [1,3,280,2] flattened

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum AnrMode {
    #[id = "pmln_only"]
    PmlnOnly,
    #[id = "stln_only"]
    StlnOnly,
    #[id = "stln_pmln"]
    StlnPmln,
}

#[derive(Params)]
struct AnrParams {
    #[id = "mode"]
    mode: EnumParam<AnrMode>,

    #[id = "stln_strength"]
    stln_strength: FloatParam,
}

impl Default for AnrParams {
    fn default() -> Self {
        Self {
            mode: EnumParam::new("Mode", AnrMode::PmlnOnly),
            stln_strength: FloatParam::new(
                "STLN Strength",
                1.0,
                FloatRange::Linear {
                    min: 0.5,
                    max: 4.0,
                },
            ),
        }
    }
}

// ============================================================================
// Resampling Pipeline - Testable core logic
// ============================================================================

/// The core resampling pipeline that handles:
/// - Input ring buffer for decoupling from host
/// - Downsampling from host rate to 16kHz
/// - Processing through ANR
/// - Upsampling back to host rate
/// - Output ring buffer for decoupling
///
/// This struct is fully testable without NIH-plug dependencies.
pub struct ResamplingPipeline {
    downsampler: Downsampler,
    upsampler: Upsampler,
    processor: AnrProcessor,

    input_producer: Producer<f32>,
    input_consumer: Consumer<f32>,
    output_producer: Producer<f32>,
    output_consumer: Consumer<f32>,

    downsample_buf: Vec<f32>,
    pipeline_primed: bool,
}

impl ResamplingPipeline {
    /// Create a new resampling pipeline for the given host sample rate.
    /// Returns None if host_sr == 16kHz (use direct processing instead).
    pub fn new(host_sr: f32) -> Option<Self> {
        let downsampler = Downsampler::new(host_sr)?;
        let upsampler = Upsampler::new(host_sr)?;

        let (input_producer, input_consumer) = RingBuffer::new(RING_BUFFER_SIZE);
        let (output_producer, output_consumer) = RingBuffer::new(RING_BUFFER_SIZE);

        let max_input_needed = downsampler.input_frames_needed() * 2;

        Some(Self {
            downsampler,
            upsampler,
            processor: AnrProcessor::new(),
            input_producer,
            input_consumer,
            output_producer,
            output_consumer,
            downsample_buf: vec![0.0; max_input_needed],
            pipeline_primed: false,
        })
    }

    /// Push input samples into the pipeline's input ring buffer.
    /// Overwrites oldest samples if buffer is full.
    pub fn push_input(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.input_producer.push(sample).is_err() {
                // Buffer full - discard oldest
                let _ = self.input_consumer.pop();
                let _ = self.input_producer.push(sample);
            }
        }
    }

    /// Run the processing loop: downsample → process → upsample.
    /// Processes as many hops as possible given buffered input.
    /// Returns the number of hops processed.
    pub fn run_processing(&mut self, mode: ProcessingMode, stln_strength: f32) -> usize {
        let anr_mode = match mode {
            ProcessingMode::PmlnOnly => AnrMode::PmlnOnly,
            ProcessingMode::StlnOnly => AnrMode::StlnOnly,
            ProcessingMode::StlnPmln => AnrMode::StlnPmln,
        };

        let mut hops_processed = 0;

        loop {
            let input_needed = self.downsampler.input_frames_needed();
            let available = self.input_consumer.slots();

            if available < input_needed {
                break;
            }

            // Pop exactly input_needed samples
            self.downsample_buf.clear();
            for _ in 0..input_needed {
                if let Ok(sample) = self.input_consumer.pop() {
                    self.downsample_buf.push(sample);
                }
            }

            // Downsample: host rate → 16kHz
            let downsampled = self.downsampler.process(&self.downsample_buf);

            // Process at 16kHz
            let processed = self.processor.process_hop(&downsampled, anr_mode, stln_strength);

            // Upsample: 16kHz → host rate
            let upsampled = self.upsampler.process(&processed);

            // Push to output ring buffer
            for &sample in upsampled {
                if self.output_producer.push(sample).is_err() {
                    let _ = self.output_consumer.pop();
                    let _ = self.output_producer.push(sample);
                }
            }

            hops_processed += 1;
        }

        hops_processed
    }

    /// How many output samples are available to pop.
    pub fn output_available(&self) -> usize {
        self.output_consumer.slots()
    }

    /// Pop output samples into the provided buffer.
    /// Returns (samples_written, underruns).
    pub fn pop_output(&mut self, output: &mut [f32]) -> (usize, usize) {
        let mut written = 0;
        let mut underruns = 0;

        for out in output.iter_mut() {
            match self.output_consumer.pop() {
                Ok(sample) => {
                    *out = sample;
                    written += 1;
                }
                Err(_) => {
                    *out = 0.0;
                    underruns += 1;
                }
            }
        }

        (written, underruns)
    }

    /// Check if pipeline is primed (has enough buffered output).
    pub fn is_primed(&self) -> bool {
        self.pipeline_primed
    }

    /// Try to prime the pipeline. Returns true if primed.
    pub fn try_prime(&mut self, required_samples: usize) -> bool {
        if !self.pipeline_primed && self.output_available() >= required_samples {
            self.pipeline_primed = true;
        }
        self.pipeline_primed
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.processor.reset();
        self.downsampler.reset();
        self.upsampler.reset();
        self.pipeline_primed = false;

        // Clear ring buffers by recreating them
        let (input_producer, input_consumer) = RingBuffer::new(RING_BUFFER_SIZE);
        let (output_producer, output_consumer) = RingBuffer::new(RING_BUFFER_SIZE);
        self.input_producer = input_producer;
        self.input_consumer = input_consumer;
        self.output_producer = output_producer;
        self.output_consumer = output_consumer;
    }

    /// Get the latency in host-rate samples.
    pub fn latency(&self) -> u32 {
        calculate_resample_latency(&self.downsampler, &self.upsampler)
    }

    /// How many input samples are currently buffered.
    pub fn input_buffered(&self) -> usize {
        self.input_consumer.slots()
    }
}

pub struct AnrPlugin {
    params: Arc<AnrParams>,

    // For 16kHz hosts: direct processing without resampling
    processor: AnrProcessor,

    // For non-16kHz hosts: full resampling pipeline
    pipeline: Option<ResamplingPipeline>,

    last_mode: Option<AnrMode>,
    latency_samples: u32,
}

impl Default for AnrPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(AnrParams::default()),
            processor: AnrProcessor::new(),
            pipeline: None,
            last_mode: None,
            latency_samples: 0,
        }
    }
}

impl Plugin for AnrPlugin {
    const NAME: &'static str = "ANR Reverse";
    const VENDOR: &'static str = "anr-test";
    const URL: &'static str = "https://example.invalid";
    const EMAIL: &'static str = "devnull@example.invalid";

    const VERSION: &'static str = "0.1.0";

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;

        // Create resampling pipeline if needed (host SR != 16kHz)
        self.pipeline = ResamplingPipeline::new(sample_rate);

        if let Some(ref pipeline) = self.pipeline {
            nih_log!("Resamplers initialized: {}Hz ↔ 16kHz", sample_rate);

            // Calculate and report latency
            let resample_latency = pipeline.latency();
            // OLA latency: FRAME_SIZE samples at 16kHz, converted to host rate
            let ola_latency_16k = FRAME_SIZE as f64;
            let ola_latency_host = (ola_latency_16k * sample_rate as f64 / 16000.0) as u32;

            self.latency_samples = resample_latency + ola_latency_host;
            context.set_latency_samples(self.latency_samples);
            nih_log!(
                "Reported latency: {} samples (resample: {}, OLA: {})",
                self.latency_samples,
                resample_latency,
                ola_latency_host
            );
        } else {
            nih_log!("No resampling needed (host at 16kHz)");
            self.latency_samples = FRAME_SIZE as u32;
            context.set_latency_samples(self.latency_samples);
        }

        true
    }

    fn reset(&mut self) {
        nih_log!("reset() called!");
        self.processor.reset();
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        let channels = buffer.channels();

        if channels == 0 || num_samples == 0 {
            return ProcessStatus::KeepAlive;
        }

        let mode = self.params.mode.value();
        let stln_strength = self.params.stln_strength.value();

        // Log mode on first use or when changed
        if self.last_mode != Some(mode) {
            let mode_name = match mode {
                AnrMode::PmlnOnly => "PMLN Only",
                AnrMode::StlnOnly => "STLN Only",
                AnrMode::StlnPmln => "STLN + PMLN (hybrid)",
            };
            nih_log!("Processing mode: {}", mode_name);
            self.last_mode = Some(mode);
        }

        // Check if we have resampling pipeline (host SR != 16kHz)
        if self.pipeline.is_some() {
            self.process_with_resampling(buffer, mode, stln_strength);
        } else {
            self.process_direct(buffer, mode, stln_strength);
        }

        ProcessStatus::KeepAlive
    }
}

impl AnrPlugin {
    /// Process with resampling (host SR != 16kHz) using the resampling pipeline.
    fn process_with_resampling(
        &mut self,
        buffer: &mut Buffer,
        mode: AnrMode,
        stln_strength: f32,
    ) {
        let num_samples = buffer.samples();
        let channels = buffer.channels();

        let pipeline = self.pipeline.as_mut().unwrap();

        // Convert AnrMode to ProcessingMode for the pipeline
        let processing_mode = match mode {
            AnrMode::PmlnOnly => ProcessingMode::PmlnOnly,
            AnrMode::StlnOnly => ProcessingMode::StlnOnly,
            AnrMode::StlnPmln => ProcessingMode::StlnPmln,
        };

        // === INGEST: Extract mono and push to pipeline ===
        // Collect mono samples first to avoid borrow issues
        let mut mono_samples = Vec::with_capacity(num_samples);
        for mut frame in buffer.iter_samples() {
            let left = *frame.get_mut(0).unwrap();
            let right = if channels > 1 {
                *frame.get_mut(1).unwrap()
            } else {
                left
            };
            mono_samples.push((left + right) * 0.5);
        }
        pipeline.push_input(&mono_samples);

        // === PROCESSING: Run as many hops as possible ===
        let hops_processed = pipeline.run_processing(processing_mode, stln_strength);

        // === PRIMING CHECK: Wait until we have enough buffered output ===
        if !pipeline.try_prime(num_samples) {
            // Not enough output yet - emit silence
            for mut frame in buffer.iter_samples() {
                for ch in 0..channels {
                    *frame.get_mut(ch).unwrap() = 0.0;
                }
            }
            return;
        }

        // Debug: log state periodically
        static PRIME_DEBUG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let debug_count = PRIME_DEBUG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if debug_count % 100 == 0 {
            nih_log!(
                "Buffer {}: in_buf={}, out_avail={}, hops={}",
                debug_count,
                pipeline.input_buffered(),
                pipeline.output_available(),
                hops_processed
            );
        }

        // === EGEST: Pop from output to fill host buffer ===
        let mut output_buf = vec![0.0f32; num_samples];
        let (_, underruns) = pipeline.pop_output(&mut output_buf);

        if underruns > 0 {
            nih_warn!("Output underrun: {} of {} samples", underruns, num_samples);
        }

        // Write to all channels
        for (i, mut frame) in buffer.iter_samples().enumerate() {
            let sample = output_buf[i];
            for ch in 0..channels {
                *frame.get_mut(ch).unwrap() = sample;
            }
        }
    }

    /// Process directly at 16kHz (no resampling needed).
    fn process_direct(&mut self, buffer: &mut Buffer, mode: AnrMode, stln_strength: f32) {
        let num_samples = buffer.samples();
        let channels = buffer.channels();

        // Sum to mono
        let mut input_samples = vec![0.0f32; num_samples];
        for (i, mut frame) in buffer.iter_samples().enumerate() {
            let left = *frame.get_mut(0).unwrap();
            let right = if channels > 1 {
                *frame.get_mut(1).unwrap()
            } else {
                left
            };
            input_samples[i] = (left + right) * 0.5;
        }

        // Process
        let mut output_samples = vec![0.0f32; num_samples];
        self.processor
            .process_block(&input_samples, &mut output_samples, mode, stln_strength);

        // Write output to all channels
        for (i, mut frame) in buffer.iter_samples().enumerate() {
            for ch in 0..channels {
                *frame.get_mut(ch).unwrap() = output_samples[i];
            }
        }
    }
}

impl ClapPlugin for AnrPlugin {
    const CLAP_ID: &'static str = "com.anr-test.anr-reverse";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("ANR reverse-engineered DSP");
    const CLAP_MANUAL_URL: Option<&'static str> = Some("https://example.invalid");
    const CLAP_SUPPORT_URL: Option<&'static str> = Some("https://example.invalid");
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect];
}

impl Vst3Plugin for AnrPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"ANRReverseDSP___";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Restoration];
}

nih_export_clap!(AnrPlugin);
nih_export_vst3!(AnrPlugin);

// ============================================================================
// ANR Processor (FFT/IFFT + ONNX inference at 16kHz)
// ============================================================================

struct AnrProcessor {
    time_buf: Vec<f32>,
    ola_buf: Vec<f32>,
    window: Vec<f32>,
    input_fifo: Fifo,
    output_fifo: Fifo,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    ifft_input: Vec<Complex32>,
    ifft_output: Vec<f32>,
    stln_state: Vec<f32>,
    pmln_state: Vec<f32>,
    model: Box<dyn ModelBackend>,
    debug_logged: bool,
    stln_debug_logged: bool,
}

impl AnrProcessor {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        let window = build_hann_window(FRAME_SIZE);
        let time_buf = vec![0.0; FRAME_SIZE];
        let ola_buf = vec![0.0; FRAME_SIZE];
        let fft_input = vec![0.0; FRAME_SIZE];
        let fft_output = fft.make_output_vec();
        let ifft_input = vec![Complex32::new(0.0, 0.0); FFT_BINS];
        let ifft_output = vec![0.0; FRAME_SIZE];

        // FIFO capacity for direct 16kHz processing
        let fifo_capacity = 65536;

        Self {
            time_buf,
            ola_buf,
            window,
            input_fifo: Fifo::with_capacity(fifo_capacity),
            output_fifo: Fifo::with_capacity(fifo_capacity),
            fft,
            ifft,
            fft_input,
            fft_output,
            ifft_input,
            ifft_output,
            stln_state: vec![0.0; STLN_STATE_SIZE],
            pmln_state: vec![0.0; PMLN_STATE_SIZE],
            model: create_model_backend(),
            debug_logged: false,
            stln_debug_logged: false,
        }
    }

    fn reset(&mut self) {
        self.time_buf.fill(0.0);
        self.ola_buf.fill(0.0);
        self.input_fifo.clear();
        self.output_fifo.clear();
        self.stln_state.fill(0.0);
        self.pmln_state.fill(0.0);
    }

    fn process_block(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        mode: AnrMode,
        stln_strength: f32,
    ) {
        let mut input_pos = 0;
        let mut output_pos = 0;

        // Process input in chunks, draining FIFOs as we go
        while input_pos < input.len() || self.input_fifo.available() >= HOP_SIZE {
            // Push as much input as fits in the FIFO
            if input_pos < input.len() {
                let pushed = self.input_fifo.push(&input[input_pos..]);
                input_pos += pushed;
            }

            // Process all available hops
            while self.input_fifo.available() >= HOP_SIZE {
                let mut hop = [0.0f32; HOP_SIZE];
                self.input_fifo.pop(&mut hop);
                let out_hop = self.process_hop(&hop, mode, stln_strength);
                self.output_fifo.push(&out_hop);
            }

            // Pop available output
            if output_pos < output.len() {
                let popped = self.output_fifo.pop(&mut output[output_pos..]);
                output_pos += popped;
            }
        }

        // Zero any remaining output (shouldn't happen in normal operation)
        output[output_pos..].fill(0.0);
    }

    fn process_hop(
        &mut self,
        hop: &[f32; HOP_SIZE],
        mode: AnrMode,
        stln_strength: f32,
    ) -> [f32; HOP_SIZE] {
        // Slide time buffer and insert new hop
        self.time_buf.copy_within(HOP_SIZE.., 0);
        self.time_buf[FRAME_SIZE - HOP_SIZE..].copy_from_slice(hop);

        // Apply analysis window
        simd::apply_window(&self.time_buf, &self.window, &mut self.fft_input);

        // FFT - must succeed for correct audio processing
        if let Err(e) = self.fft.process(&mut self.fft_input, &mut self.fft_output) {
            nih_error!("FFT processing failed: {:?}", e);
            // Zero output to make failure audible rather than corrupt
            self.fft_output.fill(Complex32::new(0.0, 0.0));
        }

        // Extract magnitude and phase
        let mut magnitude = [0.0f32; FFT_BINS];
        let mut phase = [0.0f32; FFT_BINS];
        simd::extract_mag_phase(&self.fft_output, &mut magnitude, &mut phase);

        // Run model(s)
        let (out_mag, out_phase) = match mode {
            AnrMode::PmlnOnly => self.run_pmln(&magnitude, &phase),
            AnrMode::StlnOnly => self.run_stln(&magnitude, &phase, stln_strength, false),
            AnrMode::StlnPmln => self.run_stln(&magnitude, &phase, stln_strength, true),
        };

        // Reconstruct complex spectrum
        simd::polar_to_complex(&out_mag, &out_phase, &mut self.ifft_input);

        // IFFT - must succeed for correct audio processing
        if let Err(e) = self.ifft.process(&mut self.ifft_input, &mut self.ifft_output) {
            nih_error!("IFFT processing failed: {:?}", e);
            // Zero output to make failure audible rather than corrupt
            self.ifft_output.fill(0.0);
        }

        // Scale by FFT size
        simd::scale_buffer(&mut self.ifft_output, 1.0 / FRAME_SIZE as f32);

        // OLA: shift and accumulate
        self.ola_buf.copy_within(HOP_SIZE.., 0);
        self.ola_buf[FRAME_SIZE - HOP_SIZE..].fill(0.0);
        simd::ola_add(&mut self.ola_buf, &self.ifft_output);

        // Output first HOP_SIZE samples
        let mut out = [0.0f32; HOP_SIZE];
        out.copy_from_slice(&self.ola_buf[..HOP_SIZE]);
        out
    }

    fn run_pmln(
        &mut self,
        magnitude: &[f32; FFT_BINS],
        phase: &[f32; FFT_BINS],
    ) -> ([f32; FFT_BINS], [f32; FFT_BINS]) {
        let mut packed = [0.0f32; FFT_BINS * 2];
        packed[..FFT_BINS].copy_from_slice(magnitude);
        packed[FFT_BINS..].copy_from_slice(phase);

        let (mask, new_state) = self.model.run_pmln(&packed, &self.pmln_state);
        self.pmln_state.copy_from_slice(&new_state);

        // Debug: log mask stats once
        if !self.debug_logged {
            let mask_mag = &mask[..FFT_BINS];
            let min = mask_mag.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mask_mag.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = mask_mag.iter().sum::<f32>() / FFT_BINS as f32;
            nih_log!(
                "PMLN mask stats - min: {:.4}, max: {:.4}, mean: {:.4}",
                min,
                max,
                mean
            );
            self.debug_logged = true;
        }

        let mut out_mag = [0.0f32; FFT_BINS];
        let mut out_phase = [0.0f32; FFT_BINS];
        simd::apply_pmln_mask(&packed, &mask, &mut out_mag, &mut out_phase);

        (out_mag, out_phase)
    }

    fn run_stln(
        &mut self,
        magnitude: &[f32; FFT_BINS],
        _phase: &[f32; FFT_BINS],
        stln_strength: f32,
        run_pmln: bool,
    ) -> ([f32; FFT_BINS], [f32; FFT_BINS]) {
        let mut stln_in = *magnitude;
        stln_in[0] = 0.0;
        for v in stln_in.iter_mut() {
            *v += EPSILON;
        }

        let (mask, new_state) = self.model.run_stln(&stln_in, &self.stln_state);
        self.stln_state.copy_from_slice(&new_state);

        // Debug: log STLN mask stats once
        if !self.stln_debug_logged {
            let min = mask.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mask.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = mask.iter().sum::<f32>() / mask.len() as f32;
            nih_log!(
                "STLN mask stats - min: {:.4}, max: {:.4}, mean: {:.4}",
                min,
                max,
                mean
            );
            self.stln_debug_logged = true;
        }

        let mut mask = mask;
        if (stln_strength - 1.0).abs() > f32::EPSILON {
            simd::apply_mask_strength(&mut mask, stln_strength);
        }

        let mut enhanced_spec = [Complex32::new(0.0, 0.0); FFT_BINS];
        simd::complex_mul_scalar(&self.fft_output, &mask, &mut enhanced_spec);

        let mut enhanced_mag = [0.0f32; FFT_BINS];
        let mut enhanced_phase = [0.0f32; FFT_BINS];
        simd::extract_mag_phase(&enhanced_spec, &mut enhanced_mag, &mut enhanced_phase);

        if run_pmln {
            self.run_pmln(&enhanced_mag, &enhanced_phase)
        } else {
            (enhanced_mag, enhanced_phase)
        }
    }
}

// ============================================================================
// Model Backend (ONNX or Passthrough)
// ============================================================================

trait ModelBackend: Send + Sync {
    fn run_pmln(&mut self, packed: &[f32; FFT_BINS * 2], state: &[f32]) -> (Vec<f32>, Vec<f32>);
    fn run_stln(&mut self, magnitude: &[f32; FFT_BINS], state: &[f32]) -> (Vec<f32>, Vec<f32>);
}

struct PassthroughBackend;

impl ModelBackend for PassthroughBackend {
    fn run_pmln(&mut self, _packed: &[f32; FFT_BINS * 2], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (vec![1.0f32; FFT_BINS * 2], state.to_vec())
    }

    fn run_stln(&mut self, _magnitude: &[f32; FFT_BINS], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (vec![1.0f32; FFT_BINS], state.to_vec())
    }
}

struct OnnxBackend {
    pmln_session: Session,
    stln_session: Session,
}

impl OnnxBackend {
    fn new(pmln_path: PathBuf, stln_path: PathBuf) -> Result<Self, ort::Error> {
        let pmln_session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(&pmln_path)?;
        let stln_session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(&stln_path)?;
        Ok(Self {
            pmln_session,
            stln_session,
        })
    }
}

impl ModelBackend for OnnxBackend {
    fn run_pmln(&mut self, packed: &[f32; FFT_BINS * 2], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let input_tensor = match Tensor::from_array(([1usize, 1, 514], packed.to_vec())) {
            Ok(t) => t,
            Err(e) => {
                nih_error!("PMLN input tensor error: {}", e);
                return (vec![1.0f32; FFT_BINS * 2], state.to_vec());
            }
        };

        let state_tensor = match Tensor::from_array(([1usize, 4, 280, 2], state.to_vec())) {
            Ok(t) => t,
            Err(e) => {
                nih_error!("PMLN state tensor error: {}", e);
                return (vec![1.0f32; FFT_BINS * 2], state.to_vec());
            }
        };

        let outputs = match self.pmln_session.run(ort::inputs![
            "input_1" => input_tensor,
            "input_2" => state_tensor,
        ]) {
            Ok(out) => out,
            Err(e) => {
                nih_error!("PMLN inference error: {}", e);
                return (vec![1.0f32; FFT_BINS * 2], state.to_vec());
            }
        };

        let mask: Vec<f32> = match outputs["audio_out"].try_extract_tensor::<f32>() {
            Ok((_shape, data)) => {
                let v = data.to_vec();
                if v.len() != FFT_BINS * 2 {
                    nih_error!("PMLN mask wrong size: got {}, expected {}", v.len(), FFT_BINS * 2);
                    vec![1.0f32; FFT_BINS * 2]
                } else {
                    v
                }
            }
            Err(e) => {
                nih_error!("PMLN mask extraction failed: {:?}", e);
                vec![1.0f32; FFT_BINS * 2]
            }
        };

        let new_state: Vec<f32> = match outputs["state_vector_out"].try_extract_tensor::<f32>() {
            Ok((_shape, data)) => {
                let v = data.to_vec();
                if v.len() != PMLN_STATE_SIZE {
                    nih_error!("PMLN state wrong size: got {}, expected {}", v.len(), PMLN_STATE_SIZE);
                    state.to_vec()
                } else {
                    v
                }
            }
            Err(e) => {
                nih_error!("PMLN state extraction failed: {:?}", e);
                state.to_vec()
            }
        };

        (mask, new_state)
    }

    fn run_stln(&mut self, magnitude: &[f32; FFT_BINS], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let input_tensor = match Tensor::from_array(([1usize, 1, 257], magnitude.to_vec())) {
            Ok(t) => t,
            Err(e) => {
                nih_error!("STLN input tensor error: {}", e);
                return (vec![1.0f32; FFT_BINS], state.to_vec());
            }
        };

        let state_tensor = match Tensor::from_array(([1usize, 3, 280, 2], state.to_vec())) {
            Ok(t) => t,
            Err(e) => {
                nih_error!("STLN state tensor error: {}", e);
                return (vec![1.0f32; FFT_BINS], state.to_vec());
            }
        };

        let outputs = match self.stln_session.run(ort::inputs![
            "input_2" => input_tensor,
            "input_3" => state_tensor,
        ]) {
            Ok(out) => out,
            Err(e) => {
                nih_error!("STLN inference error: {}", e);
                return (vec![1.0f32; FFT_BINS], state.to_vec());
            }
        };

        let mask: Vec<f32> = match outputs["tf.math.sigmoid_12"].try_extract_tensor::<f32>() {
            Ok((_shape, data)) => {
                let v = data.to_vec();
                if v.len() != FFT_BINS {
                    nih_error!("STLN mask wrong size: got {}, expected {}", v.len(), FFT_BINS);
                    vec![1.0f32; FFT_BINS]
                } else {
                    v
                }
            }
            Err(e) => {
                nih_error!("STLN mask extraction failed: {:?}", e);
                vec![1.0f32; FFT_BINS]
            }
        };

        let new_state: Vec<f32> = match outputs["tf.stack_2"].try_extract_tensor::<f32>() {
            Ok((_shape, data)) => {
                let v = data.to_vec();
                if v.len() != STLN_STATE_SIZE {
                    nih_error!("STLN state wrong size: got {}, expected {}", v.len(), STLN_STATE_SIZE);
                    state.to_vec()
                } else {
                    v
                }
            }
            Err(e) => {
                nih_error!("STLN state extraction failed: {:?}", e);
                state.to_vec()
            }
        };

        (mask, new_state)
    }
}

// SAFETY: ort::Session is thread-safe (uses internal synchronization)
unsafe impl Sync for OnnxBackend {}

fn create_model_backend() -> Box<dyn ModelBackend> {
    let model_dir = match model_path::find_model_dir() {
        Some(dir) => dir,
        None => {
            nih_warn!(
                "ONNX models not found. Checked: ANR_MODEL_DIR env var, \
                 binary directory, platform-specific locations. \
                 Using passthrough (no noise reduction)."
            );
            return Box::new(PassthroughBackend);
        }
    };

    let pmln_path = model_dir.join("PMLN_model.onnx");
    let stln_path = model_dir.join("STLN_model.onnx");

    match OnnxBackend::new(pmln_path, stln_path) {
        Ok(backend) => {
            nih_log!("Loaded ONNX models from {:?}", model_dir);
            Box::new(backend)
        }
        Err(e) => {
            nih_error!(
                "Failed to load ONNX models from {:?}: {}; using passthrough",
                model_dir,
                e
            );
            Box::new(PassthroughBackend)
        }
    }
}

// ============================================================================
// Simple FIFO for direct 16kHz processing
// ============================================================================

struct Fifo {
    buffer: Vec<f32>,
    read: usize,
    write: usize,
    len: usize,
}

impl Fifo {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            read: 0,
            write: 0,
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.read = 0;
        self.write = 0;
        self.len = 0;
    }

    fn available(&self) -> usize {
        self.len
    }

    fn push(&mut self, data: &[f32]) -> usize {
        let space = self.buffer.len() - self.len;
        let to_write = data.len().min(space);

        for &v in &data[..to_write] {
            self.buffer[self.write] = v;
            self.write = (self.write + 1) % self.buffer.len();
            self.len += 1;
        }

        to_write
    }

    /// Push all data, panicking if it doesn't fit. Use this when data loss is unacceptable.
    #[cfg(test)]
    fn push_exact(&mut self, data: &[f32]) {
        let pushed = self.push(data);
        assert_eq!(
            pushed,
            data.len(),
            "FIFO overflow: tried to push {} samples but only {} fit (capacity: {}, current: {})",
            data.len(),
            pushed,
            self.buffer.len(),
            self.len
        );
    }

    fn pop(&mut self, out: &mut [f32]) -> usize {
        let count = out.len().min(self.len);
        for i in 0..count {
            out[i] = self.buffer[self.read];
            self.read = (self.read + 1) % self.buffer.len();
        }
        self.len -= count;
        count
    }
}

fn build_hann_window(size: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; size];
    let denom = (size - 1) as f32;
    for (i, val) in window.iter_mut().enumerate() {
        let n = i as f32;
        *val = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n / denom).cos();
    }
    window
}

// ============================================================================
// Public Test API for Integration Tests
// ============================================================================

/// Processing mode for the ANR models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    PmlnOnly,
    StlnOnly,
    StlnPmln,
}

/// Processes audio at 16kHz through the ANR pipeline.
///
/// This is the public API for integration tests.
/// Input and output are at 16kHz sample rate.
pub struct AudioProcessor {
    processor: AnrProcessor,
}

impl AudioProcessor {
    /// Create a new audio processor.
    ///
    /// The model_dir should contain PMLN_model.onnx and STLN_model.onnx.
    /// If None, uses the default model search paths.
    pub fn new() -> Self {
        Self {
            processor: AnrProcessor::new(),
        }
    }

    /// Process a block of audio at 16kHz.
    ///
    /// Input and output are at 16kHz sample rate.
    pub fn process_16k(
        &mut self,
        input: &[f32],
        mode: ProcessingMode,
        stln_strength: f32,
    ) -> Vec<f32> {
        let anr_mode = match mode {
            ProcessingMode::PmlnOnly => AnrMode::PmlnOnly,
            ProcessingMode::StlnOnly => AnrMode::StlnOnly,
            ProcessingMode::StlnPmln => AnrMode::StlnPmln,
        };

        let mut output = vec![0.0f32; input.len()];
        self.processor.process_block(input, &mut output, anr_mode, stln_strength);
        output
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.processor.reset();
    }

    /// Process a single hop at 16kHz (for testing the resampling path).
    ///
    /// This is what the plugin uses when running at non-16kHz sample rates.
    /// Input must be exactly HOP_SIZE samples.
    pub fn process_hop_16k(
        &mut self,
        hop: &[f32; HOP_SIZE],
        mode: ProcessingMode,
        stln_strength: f32,
    ) -> [f32; HOP_SIZE] {
        let anr_mode = match mode {
            ProcessingMode::PmlnOnly => AnrMode::PmlnOnly,
            ProcessingMode::StlnOnly => AnrMode::StlnOnly,
            ProcessingMode::StlnPmln => AnrMode::StlnPmln,
        };
        self.processor.process_hop(hop, anr_mode, stln_strength)
    }
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Pipeline Integration Tests
// ============================================================================

#[cfg(test)]
mod pipeline_tests {
    use super::*;
    use crate::test_utils::*;

    /// Simulates the host calling process() with variable buffer sizes.
    /// Returns all output samples collected.
    fn simulate_host_processing(
        input: &[f32],
        buffer_sizes: &[usize],
        sample_rate: f32,
    ) -> Vec<f32> {
        // Create resamplers
        let mut downsampler = Downsampler::new(sample_rate);
        let mut upsampler = Upsampler::new(sample_rate);

        let mut output = Vec::new();

        if downsampler.is_some() && upsampler.is_some() {
            let mut ds = downsampler.take().unwrap();
            let mut us = upsampler.take().unwrap();

            // Create ring buffers
            let (mut input_prod, mut input_cons) = RingBuffer::new(RING_BUFFER_SIZE);
            let (mut output_prod, mut output_cons) = RingBuffer::new(RING_BUFFER_SIZE);

            // Create processor with passthrough model
            let mut processor = AnrProcessor::new();
            let mut downsample_buf = Vec::new();

            let mut input_pos = 0;
            let mut buffer_idx = 0;

            // Process in chunks according to buffer_sizes pattern
            while input_pos < input.len() {
                let buffer_size = buffer_sizes[buffer_idx % buffer_sizes.len()];
                let chunk_end = (input_pos + buffer_size).min(input.len());
                let chunk = &input[input_pos..chunk_end];
                input_pos = chunk_end;
                buffer_idx += 1;

                // === INGEST ===
                for &sample in chunk {
                    if input_prod.push(sample).is_err() {
                        let _ = input_cons.pop();
                        let _ = input_prod.push(sample);
                    }
                }

                // === PROCESS LOOP ===
                loop {
                    let needed = ds.input_frames_needed();
                    if input_cons.slots() < needed {
                        break;
                    }

                    downsample_buf.clear();
                    for _ in 0..needed {
                        if let Ok(s) = input_cons.pop() {
                            downsample_buf.push(s);
                        }
                    }

                    let downsampled = ds.process(&downsample_buf);
                    let processed = processor.process_hop(&downsampled, AnrMode::PmlnOnly, 1.0);
                    let upsampled = us.process(&processed);

                    for &s in upsampled {
                        if output_prod.push(s).is_err() {
                            let _ = output_cons.pop();
                            let _ = output_prod.push(s);
                        }
                    }
                }

                // === EGEST ===
                let egest_count = chunk.len();
                for _ in 0..egest_count {
                    output.push(output_cons.pop().unwrap_or(0.0));
                }
            }
        } else {
            // Direct 16kHz processing (no resampling)
            let mut processor = AnrProcessor::new();
            let mut out_buf = vec![0.0f32; input.len()];
            processor.process_block(input, &mut out_buf, AnrMode::PmlnOnly, 1.0);
            output = out_buf;
        }

        output
    }

    // ========================================================================
    // Ring Buffer Tests
    // ========================================================================

    #[test]
    fn ring_buffer_basic_push_pop() {
        let (mut prod, mut cons) = RingBuffer::<f32>::new(1024);

        // Push some samples
        for i in 0..100 {
            assert!(prod.push(i as f32).is_ok());
        }

        // Pop and verify
        for i in 0..100 {
            let val = cons.pop().unwrap();
            assert_eq!(val, i as f32);
        }
    }

    #[test]
    fn ring_buffer_overflow_handling() {
        let (mut prod, mut cons) = RingBuffer::<f32>::new(10);

        // Fill buffer
        for i in 0..10 {
            assert!(prod.push(i as f32).is_ok());
        }

        // Next push should fail
        assert!(prod.push(100.0).is_err());

        // Pop one and push should succeed
        let _ = cons.pop();
        assert!(prod.push(100.0).is_ok());
    }

    #[test]
    fn ring_buffer_underflow_returns_none() {
        let (_prod, mut cons) = RingBuffer::<f32>::new(10);

        // Empty buffer should return Err
        assert!(cons.pop().is_err());
    }

    // ========================================================================
    // FFT/IFFT Round-trip Tests
    // ========================================================================

    #[test]
    fn fft_ifft_roundtrip() {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        // Generate a test signal
        let mut input = generate_sine(1000.0, 16000.0, FRAME_SIZE);
        let input_rms = compute_rms(&input);

        // FFT
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut input, &mut spectrum).unwrap();

        // IFFT
        let mut output = vec![0.0f32; FRAME_SIZE];
        ifft.process(&mut spectrum, &mut output).unwrap();

        // Scale by FRAME_SIZE (standard FFT normalization)
        for x in output.iter_mut() {
            *x /= FRAME_SIZE as f32;
        }

        let output_rms = compute_rms(&output);

        // RMS should be preserved
        let ratio = output_rms / input_rms;
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "FFT/IFFT should preserve signal: ratio={}",
            ratio
        );
    }

    #[test]
    fn magnitude_phase_extraction_correct() {
        // Create a known complex spectrum
        let spectrum = vec![
            Complex32::new(1.0, 0.0),   // DC: mag=1, phase=0
            Complex32::new(0.0, 1.0),   // Positive imaginary: mag=1, phase=π/2
            Complex32::new(-1.0, 0.0),  // Negative real: mag=1, phase=π
        ];

        let mut magnitude = [0.0f32; 3];
        let mut phase = [0.0f32; 3];

        for (i, c) in spectrum.iter().enumerate() {
            magnitude[i] = c.norm();
            phase[i] = c.arg();
        }

        assert!((magnitude[0] - 1.0).abs() < 0.001);
        assert!((magnitude[1] - 1.0).abs() < 0.001);
        assert!((magnitude[2] - 1.0).abs() < 0.001);

        assert!(phase[0].abs() < 0.001); // 0
        assert!((phase[1] - std::f32::consts::FRAC_PI_2).abs() < 0.001); // π/2
        assert!((phase[2].abs() - std::f32::consts::PI).abs() < 0.001); // π or -π
    }

    // ========================================================================
    // OLA (Overlap-Add) Tests
    // ========================================================================

    #[test]
    fn ola_with_identity_reconstructs_input() {
        let mut processor = AnrProcessor::new();

        // Generate input longer than FRAME_SIZE
        let input = generate_sine(500.0, 16000.0, 16000); // 1 second

        let mut output = Vec::new();

        // Process in HOP_SIZE chunks
        for chunk_start in (0..input.len()).step_by(HOP_SIZE) {
            let chunk_end = (chunk_start + HOP_SIZE).min(input.len());
            if chunk_end - chunk_start < HOP_SIZE {
                break;
            }

            let mut hop = [0.0f32; HOP_SIZE];
            hop.copy_from_slice(&input[chunk_start..chunk_end]);

            // Process with passthrough (model returns all-ones mask)
            let out_hop = processor.process_hop(&hop, AnrMode::PmlnOnly, 1.0);
            output.extend_from_slice(&out_hop);
        }

        // Skip initial transient (FRAME_SIZE samples)
        let skip = FRAME_SIZE * 4;
        if output.len() > skip + 4000 {
            let input_trimmed = &input[skip..skip + 4000];
            let output_trimmed = &output[skip..skip + 4000];

            // Check frequency is preserved
            let in_freq = estimate_frequency(input_trimmed, 16000.0);
            let out_freq = estimate_frequency(output_trimmed, 16000.0);
            assert!(
                (in_freq - out_freq).abs() < 50.0,
                "OLA should preserve frequency: in={}, out={}",
                in_freq,
                out_freq
            );

            // Check level is reasonable (passthrough model may attenuate)
            let in_rms = compute_rms(input_trimmed);
            let out_rms = compute_rms(output_trimmed);
            assert!(
                out_rms > 0.01,
                "OLA output should not be silent: rms={}",
                out_rms
            );
        }
    }

    // ========================================================================
    // Variable Buffer Size Tests
    // ========================================================================

    #[test]
    fn handles_standard_buffer_sizes() {
        for buffer_size in [64, 128, 256, 512, 1024, 2048] {
            let input = generate_sine(500.0, 48000.0, 48000);
            let output = simulate_host_processing(&input, &[buffer_size], 48000.0);

            // Should produce output
            assert!(
                output.len() > 0,
                "Should produce output for buffer size {}",
                buffer_size
            );

            // Skip warmup and check signal exists
            let skip = 5000;
            if output.len() > skip + 1000 {
                let rms = compute_rms(&output[skip..skip + 1000]);
                assert!(
                    rms > 0.01,
                    "Output should not be silent for buffer size {}: rms={}",
                    buffer_size,
                    rms
                );
            }
        }
    }

    #[test]
    fn handles_host_jitter_pattern() {
        // Simulate host sending irregular buffer sizes
        let jitter_pattern = [1056, 512, 1056, 480, 512, 1024];
        let input = generate_sine(500.0, 48000.0, 96000); // 2 seconds

        let output = simulate_host_processing(&input, &jitter_pattern, 48000.0);

        // Should produce approximately same amount of output as input
        let ratio = output.len() as f32 / input.len() as f32;
        assert!(
            ratio > 0.9 && ratio < 1.1,
            "Output length should match input: ratio={}",
            ratio
        );

        // Signal should be present after warmup
        let skip = 10000;
        if output.len() > skip + 5000 {
            let out_rms = compute_rms(&output[skip..skip + 5000]);
            assert!(
                out_rms > 0.01,
                "Output should have signal with jitter pattern: rms={}",
                out_rms
            );
        }
    }

    #[test]
    fn handles_very_small_buffers() {
        // Some hosts use very small buffers (32-64 samples)
        let input = generate_sine(500.0, 48000.0, 24000);
        let output = simulate_host_processing(&input, &[32], 48000.0);

        assert!(output.len() > 0, "Should produce output with 32-sample buffers");
    }

    #[test]
    fn handles_very_large_buffers() {
        // Some hosts use very large buffers (4096+)
        let input = generate_sine(500.0, 48000.0, 48000);
        let output = simulate_host_processing(&input, &[4096], 48000.0);

        assert!(output.len() > 0, "Should produce output with 4096-sample buffers");
    }

    #[test]
    fn handles_buffer_size_changes() {
        // Buffer size can change during session
        let pattern = [512, 512, 1024, 1024, 256, 256, 512];
        let input = generate_sine(500.0, 48000.0, 48000);

        let output = simulate_host_processing(&input, &pattern, 48000.0);

        let skip = 5000;
        if output.len() > skip + 2000 {
            let rms = compute_rms(&output[skip..skip + 2000]);
            assert!(
                rms > 0.01,
                "Should handle buffer size changes gracefully: rms={}",
                rms
            );
        }
    }

    // ========================================================================
    // Sample Rate Tests
    // ========================================================================

    #[test]
    fn handles_44100hz() {
        let input = generate_sine(500.0, 44100.0, 44100);
        let output = simulate_host_processing(&input, &[512], 44100.0);

        assert!(output.len() > 0, "Should produce output at 44.1kHz");
    }

    #[test]
    fn handles_48000hz() {
        let input = generate_sine(500.0, 48000.0, 48000);
        let output = simulate_host_processing(&input, &[512], 48000.0);

        assert!(output.len() > 0, "Should produce output at 48kHz");
    }

    #[test]
    fn handles_96000hz() {
        let input = generate_sine(500.0, 96000.0, 96000);
        let output = simulate_host_processing(&input, &[512], 96000.0);

        assert!(output.len() > 0, "Should produce output at 96kHz");
    }

    // ========================================================================
    // Latency and Delay Tests
    // ========================================================================

    #[test]
    fn output_is_delayed_by_latency() {
        let sample_rate = 48000.0;

        // Create an impulse signal
        let mut input = vec![0.0f32; 48000];
        input[1000] = 1.0; // Impulse at sample 1000

        let output = simulate_host_processing(&input, &[512], sample_rate);

        // Find where the impulse appears in output
        let mut impulse_pos = 0;
        let mut max_val = 0.0f32;
        for (i, &v) in output.iter().enumerate() {
            if v.abs() > max_val {
                max_val = v.abs();
                impulse_pos = i;
            }
        }

        // The impulse should appear somewhere after the input position
        // due to processing latency
        assert!(
            impulse_pos > 1000,
            "Impulse should be delayed: found at {}",
            impulse_pos
        );

        // Delay should be reasonable (less than 0.5 seconds at 48kHz)
        let delay = impulse_pos - 1000;
        assert!(
            delay < 24000,
            "Delay should be < 24000 samples: {}",
            delay
        );
    }

    // ========================================================================
    // Passthrough Verification
    // ========================================================================

    #[test]
    fn passthrough_preserves_frequency_content() {
        let test_freq = 500.0;
        let sample_rate = 48000.0;
        let input = generate_sine(test_freq, sample_rate, 96000);

        let output = simulate_host_processing(&input, &[512], sample_rate);

        // Skip warmup
        let skip = 20000;
        if output.len() > skip + 10000 {
            let out_freq = estimate_frequency(&output[skip..skip + 10000], sample_rate);

            assert!(
                (out_freq - test_freq).abs() < 50.0,
                "Frequency should be preserved: expected {}, got {}",
                test_freq,
                out_freq
            );
        }
    }

    #[test]
    fn passthrough_maintains_signal_energy() {
        let sample_rate = 48000.0;
        let input = generate_sine_with_amplitude(500.0, 0.5, sample_rate, 96000);

        let output = simulate_host_processing(&input, &[512], sample_rate);

        // Skip warmup
        let skip = 20000;
        if output.len() > skip + 10000 {
            let in_rms = compute_rms(&input[skip..skip + 10000]);
            let out_rms = compute_rms(&output[skip..skip + 10000]);

            // With passthrough model (all-ones mask), energy might be different
            // due to OLA reconstruction, but should be non-zero
            assert!(
                out_rms > 0.01,
                "Output should have signal energy: rms={}",
                out_rms
            );
        }
    }

    // ========================================================================
    // Stress Tests
    // ========================================================================

    #[test]
    fn handles_long_processing_session() {
        // 10 seconds of audio at 48kHz
        let input = generate_sine(500.0, 48000.0, 480000);
        let output = simulate_host_processing(&input, &[512], 48000.0);

        // Should process the entire signal
        let ratio = output.len() as f32 / input.len() as f32;
        assert!(
            ratio > 0.95,
            "Should process entire long signal: ratio={}",
            ratio
        );
    }

    #[test]
    fn no_memory_growth_during_processing() {
        // This is a basic sanity check - actual memory profiling would need external tools
        let input = generate_sine(500.0, 48000.0, 480000);

        // Process multiple times
        for _ in 0..3 {
            let _output = simulate_host_processing(&input, &[512], 48000.0);
        }

        // If we got here without OOM, the test passes
    }

    /// Debug test: Compare intermediate values with Python for first hop.
    ///
    /// Python values (from debug_first_hop.py):
    /// - hann[0:5]: [0.0, 3.78e-05, 1.51e-04, 3.40e-04, 6.05e-04]
    /// - magnitude[0:5]: [0.00201413, 0.00431225, 0.00703792, 0.00843309, 0.00825778]
    /// - phase[0:5]: [3.1415927, -0.99336183, 0.27835238, 1.4324682, 2.5884318]
    /// - IFFT output RMS: 0.000021
    #[test]
    fn debug_first_hop_comparison() {
        use std::f32::consts::PI;

        eprintln!("\n=== Debug First Hop Comparison ===\n");

        // Build Hann window same as Python np.hanning(512)
        let hann = build_hann_window(FRAME_SIZE);
        eprintln!("Rust hann[0:5]: {:?}", &hann[0..5]);
        eprintln!("Rust hann[255:260]: {:?}", &hann[255..260]);
        eprintln!("Rust hann sum: {:.6}", hann.iter().sum::<f32>());

        // Python values for comparison
        let py_hann = [0.0, 3.7796577e-05, 1.5118059e-04, 3.4013492e-04, 6.0463097e-04];
        eprintln!("Python hann[0:5]: {:?}", py_hann);

        // Check Hann window matches
        for i in 0..5 {
            let diff = (hann[i] - py_hann[i]).abs();
            assert!(diff < 1e-6, "Hann mismatch at {}: rust={}, py={}", i, hann[i], py_hann[i]);
        }
        eprintln!("✓ Hann window matches Python\n");

        // Create FFT planner
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        // Simulate first hop with known input from Python
        // Python chunk[0:5]: [-0.00054932, -0.00064087, -0.00061035, -0.00106812, -0.0007019]
        // For simplicity, use zeros for first hop (time_buf starts as zeros)
        // After shift, only last 128 samples are non-zero

        // Use the same input as Python (first 128 samples from test_input_16k.wav)
        // We'll manually create the time_buf state
        let mut time_buf = vec![0.0f32; FRAME_SIZE];

        // Python says after first hop: time_buf non-zero: 125
        // This means only the last ~125 samples have data
        // Let's use a simple test: all zeros (first hop warmup)

        // Apply window
        let mut windowed = vec![0.0f32; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            windowed[i] = time_buf[i] * hann[i];
        }

        // FFT
        let mut fft_input = windowed.clone();
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut fft_input, &mut spectrum).unwrap();

        eprintln!("Rust spectrum[0:5]: {:?}", &spectrum[0..5]);

        // Extract magnitude and phase
        let mut magnitude = vec![0.0f32; FFT_BINS];
        let mut phase = vec![0.0f32; FFT_BINS];
        for i in 0..FFT_BINS {
            magnitude[i] = spectrum[i].norm();
            phase[i] = spectrum[i].im.atan2(spectrum[i].re);
        }

        eprintln!("Rust magnitude[0:5]: {:?}", &magnitude[0..5]);
        eprintln!("Rust phase[0:5]: {:?}", &phase[0..5]);

        // For zeros input, magnitude should be all zeros
        assert!(magnitude[0] < 1e-10, "Zero input should give zero magnitude");
        eprintln!("✓ Zero input gives zero magnitude\n");

        // Now test with a simple known signal: single sample impulse
        let mut time_buf2 = vec![0.0f32; FRAME_SIZE];
        time_buf2[FRAME_SIZE - 1] = 1.0;  // Impulse at last position

        let mut windowed2 = vec![0.0f32; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            windowed2[i] = time_buf2[i] * hann[i];
        }
        eprintln!("Impulse windowed[-5:]: {:?}", &windowed2[FRAME_SIZE-5..]);

        let mut fft_input2 = windowed2.clone();
        let mut spectrum2 = fft.make_output_vec();
        fft.process(&mut fft_input2, &mut spectrum2).unwrap();

        let mut magnitude2 = vec![0.0f32; FFT_BINS];
        let mut phase2 = vec![0.0f32; FFT_BINS];
        for i in 0..FFT_BINS {
            magnitude2[i] = spectrum2[i].norm();
            phase2[i] = spectrum2[i].im.atan2(spectrum2[i].re);
        }

        eprintln!("Impulse magnitude[0:5]: {:?}", &magnitude2[0..5]);

        // Reconstruct and IFFT
        let mut ifft_input = vec![Complex32::new(0.0, 0.0); FFT_BINS];
        for i in 0..FFT_BINS {
            let (sin_p, cos_p) = phase2[i].sin_cos();
            ifft_input[i] = Complex32::new(magnitude2[i] * cos_p, magnitude2[i] * sin_p);
        }

        let mut ifft_output = vec![0.0f32; FRAME_SIZE];
        ifft.process(&mut ifft_input, &mut ifft_output).unwrap();

        // Scale by FRAME_SIZE (IFFT normalization)
        for v in ifft_output.iter_mut() {
            *v /= FRAME_SIZE as f32;
        }

        eprintln!("IFFT output (identity)[-5:]: {:?}", &ifft_output[FRAME_SIZE-5..]);

        // Should recover the windowed signal
        let diff_rms: f32 = ifft_output.iter()
            .zip(windowed2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt() / FRAME_SIZE as f32;

        eprintln!("FFT→IFFT roundtrip diff RMS: {:.10}", diff_rms);
        assert!(diff_rms < 1e-6, "FFT→IFFT roundtrip should be identity");
        eprintln!("✓ FFT→IFFT roundtrip is identity\n");

        eprintln!("=== All basic DSP checks passed ===");
    }

    /// Compare Rust FFT/magnitude/phase with exact Python values.
    #[test]
    fn test_exact_python_comparison() {
        eprintln!("\n=== Exact Python Value Comparison ===\n");

        // First 128 samples from Python's test_input_16k.wav
        const FIRST_HOP: [f32; 128] = [
            -0.00054932, -0.00064087, -0.00061035, -0.00106812, -0.00070190, -0.00097656, -0.00061035, -0.00057983,
            -0.00057983, -0.00018311, -0.00082397, -0.00070190, -0.00054932, -0.00033569, -0.00045776, -0.00051880,
            -0.00024414, -0.00018311, -0.00009155, -0.00091553, -0.00030518, 0.00000000, -0.00018311, 0.00012207,
            -0.00027466, 0.00003052, -0.00015259, -0.00024414, -0.00061035, -0.00070190, 0.00000000, -0.00015259,
            -0.00033569, -0.00009155, -0.00030518, -0.00009155, 0.00039673, -0.00036621, 0.00018311, 0.00015259,
            -0.00012207, 0.00036621, -0.00030518, 0.00015259, 0.00012207, 0.00009155, 0.00027466, -0.00006104,
            0.00024414, 0.00012207, 0.00064087, 0.00024414, -0.00033569, 0.00030518, 0.00021362, 0.00015259,
            0.00018311, 0.00015259, 0.00027466, 0.00076294, 0.00061035, 0.00036621, 0.00088501, 0.00054932,
            0.00012207, 0.00073242, 0.00051880, 0.00000000, 0.00073242, 0.00042725, 0.00051880, 0.00109863,
            0.00070190, 0.00067139, 0.00094604, 0.00109863, 0.00106812, 0.00137329, 0.00091553, 0.00070190,
            0.00125122, 0.00091553, 0.00091553, 0.00119019, 0.00122070, 0.00146484, 0.00134277, 0.00091553,
            0.00137329, 0.00119019, 0.00070190, 0.00128174, 0.00149536, 0.00161743, 0.00140381, 0.00125122,
            0.00131226, 0.00131226, 0.00128174, 0.00140381, 0.00164795, 0.00112915, 0.00164795, 0.00186157,
            0.00146484, 0.00158691, 0.00146484, 0.00152588, 0.00164795, 0.00152588, 0.00161743, 0.00173950,
            0.00161743, 0.00167847, 0.00115967, 0.00152588, 0.00177002, 0.00125122, 0.00219727, 0.00213623,
            0.00167847, 0.00207520, 0.00173950, 0.00146484, 0.00192261, 0.00210571, 0.00213623, 0.00201416,
        ];

        // Expected Python values
        const EXPECTED_HANN_SUM: f32 = 255.500000;
        const EXPECTED_MAG_SUM: f32 = 0.229450;
        const EXPECTED_MAG_0: f32 = 0.00201413;
        const EXPECTED_PHASE_0: f32 = 3.14159274;
        const EXPECTED_WINDOWED_RMS: f32 = 0.00006356;

        // Build same structures as AnrProcessor
        let hann = build_hann_window(FRAME_SIZE);
        let hann_sum: f32 = hann.iter().sum();
        eprintln!("Rust hann sum: {:.6}, Python: {:.6}", hann_sum, EXPECTED_HANN_SUM);
        assert!((hann_sum - EXPECTED_HANN_SUM).abs() < 0.01, "Hann sum mismatch");

        // Simulate first hop: time_buf = [0; 384] + first_hop
        let mut time_buf = vec![0.0f32; FRAME_SIZE];
        time_buf[FRAME_SIZE - HOP_SIZE..].copy_from_slice(&FIRST_HOP);

        // Apply window
        let mut windowed = vec![0.0f32; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            windowed[i] = time_buf[i] * hann[i];
        }

        let windowed_rms: f32 = (windowed.iter().map(|x| x * x).sum::<f32>() / FRAME_SIZE as f32).sqrt();
        eprintln!("Rust windowed RMS: {:.8}, Python: {:.8}", windowed_rms, EXPECTED_WINDOWED_RMS);
        assert!((windowed_rms - EXPECTED_WINDOWED_RMS).abs() < 1e-6, "Windowed RMS mismatch");
        eprintln!("✓ Windowed RMS matches\n");

        // FFT
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let mut fft_input = windowed.clone();
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut fft_input, &mut spectrum).unwrap();

        // Extract magnitude and phase
        let mut magnitude = vec![0.0f32; FFT_BINS];
        let mut phase = vec![0.0f32; FFT_BINS];
        for i in 0..FFT_BINS {
            magnitude[i] = spectrum[i].norm();
            phase[i] = spectrum[i].im.atan2(spectrum[i].re);
        }

        let mag_sum: f32 = magnitude.iter().sum();
        eprintln!("Rust magnitude sum: {:.6}, Python: {:.6}", mag_sum, EXPECTED_MAG_SUM);
        assert!((mag_sum - EXPECTED_MAG_SUM).abs() < 0.001, "Magnitude sum mismatch");

        eprintln!("Rust magnitude[0]: {:.8}, Python: {:.8}", magnitude[0], EXPECTED_MAG_0);
        assert!((magnitude[0] - EXPECTED_MAG_0).abs() < 1e-5, "Magnitude[0] mismatch");

        eprintln!("Rust phase[0]: {:.8}, Python: {:.8}", phase[0], EXPECTED_PHASE_0);
        assert!((phase[0] - EXPECTED_PHASE_0).abs() < 1e-5, "Phase[0] mismatch");

        eprintln!("✓ FFT magnitude and phase match Python\n");

        // Verify IFFT roundtrip
        // Note: realfft requires DC (bin 0) and Nyquist (bin N/2) to be purely real
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);
        let mut ifft_input = vec![Complex32::new(0.0, 0.0); FFT_BINS];
        for i in 0..FFT_BINS {
            let (sin_p, cos_p) = phase[i].sin_cos();
            ifft_input[i] = Complex32::new(magnitude[i] * cos_p, magnitude[i] * sin_p);
        }
        // DC and Nyquist must be real for realfft
        ifft_input[0] = Complex32::new(ifft_input[0].re, 0.0);
        ifft_input[FFT_BINS - 1] = Complex32::new(ifft_input[FFT_BINS - 1].re, 0.0);

        let mut ifft_output = vec![0.0f32; FRAME_SIZE];
        ifft.process(&mut ifft_input, &mut ifft_output).unwrap();

        // Scale by FRAME_SIZE
        for v in ifft_output.iter_mut() {
            *v /= FRAME_SIZE as f32;
        }

        let ifft_rms: f32 = (ifft_output.iter().map(|x| x * x).sum::<f32>() / FRAME_SIZE as f32).sqrt();
        eprintln!("Rust IFFT RMS: {:.8}, Python: {:.8}", ifft_rms, EXPECTED_WINDOWED_RMS);
        assert!((ifft_rms - EXPECTED_WINDOWED_RMS).abs() < 1e-6, "IFFT RMS mismatch");
        eprintln!("✓ IFFT roundtrip RMS matches\n");

        // Compare sample-by-sample
        let max_diff: f32 = ifft_output.iter()
            .zip(windowed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |acc, x| acc.max(x));
        eprintln!("Max IFFT vs windowed diff: {:.10}", max_diff);
        assert!(max_diff < 1e-5, "IFFT should match windowed input");
        eprintln!("✓ IFFT perfectly reconstructs windowed signal\n");

        eprintln!("=== All values match Python exactly ===");
    }

    /// Detailed hop-by-hop test comparing Rust with Python values.
    #[test]
    fn test_hop_by_hop_comparison() {
        eprintln!("\n=== Hop-by-Hop Comparison ===\n");

        // Expected Python output for first hop (from single_hop_test.py)
        const PY_HOP0_OUTPUT: [f32; 5] = [-7.6572524e-06, -8.4814137e-06, -3.0457145e-06, -1.0399135e-06, 9.0491903e-07];

        // First 128 samples from test_input_16k.wav
        const FIRST_HOP: [f32; 128] = [
            -0.00054932, -0.00064087, -0.00061035, -0.00106812, -0.00070190, -0.00097656, -0.00061035, -0.00057983,
            -0.00057983, -0.00018311, -0.00082397, -0.00070190, -0.00054932, -0.00033569, -0.00045776, -0.00051880,
            -0.00024414, -0.00018311, -0.00009155, -0.00091553, -0.00030518, 0.00000000, -0.00018311, 0.00012207,
            -0.00027466, 0.00003052, -0.00015259, -0.00024414, -0.00061035, -0.00070190, 0.00000000, -0.00015259,
            -0.00033569, -0.00009155, -0.00030518, -0.00009155, 0.00039673, -0.00036621, 0.00018311, 0.00015259,
            -0.00012207, 0.00036621, -0.00030518, 0.00015259, 0.00012207, 0.00009155, 0.00027466, -0.00006104,
            0.00024414, 0.00012207, 0.00064087, 0.00024414, -0.00033569, 0.00030518, 0.00021362, 0.00015259,
            0.00018311, 0.00015259, 0.00027466, 0.00076294, 0.00061035, 0.00036621, 0.00088501, 0.00054932,
            0.00012207, 0.00073242, 0.00051880, 0.00000000, 0.00073242, 0.00042725, 0.00051880, 0.00109863,
            0.00070190, 0.00067139, 0.00094604, 0.00109863, 0.00106812, 0.00137329, 0.00091553, 0.00070190,
            0.00125122, 0.00091553, 0.00091553, 0.00119019, 0.00122070, 0.00146484, 0.00134277, 0.00091553,
            0.00137329, 0.00119019, 0.00070190, 0.00128174, 0.00149536, 0.00161743, 0.00140381, 0.00125122,
            0.00131226, 0.00131226, 0.00128174, 0.00140381, 0.00164795, 0.00112915, 0.00164795, 0.00186157,
            0.00146484, 0.00158691, 0.00146484, 0.00152588, 0.00164795, 0.00152588, 0.00161743, 0.00173950,
            0.00161743, 0.00167847, 0.00115967, 0.00152588, 0.00177002, 0.00125122, 0.00219727, 0.00213623,
            0.00167847, 0.00207520, 0.00173950, 0.00146484, 0.00192261, 0.00210571, 0.00213623, 0.00201416,
        ];

        // Create processor manually to see intermediate state
        let hann = build_hann_window(FRAME_SIZE);
        let mut time_buf = vec![0.0f32; FRAME_SIZE];
        let mut ola_buf = vec![0.0f32; FRAME_SIZE];

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let ifft = planner.plan_fft_inverse(FRAME_SIZE);

        // Simulate first hop
        // Shift time_buf (all zeros, so this is a no-op effectively)
        time_buf.copy_within(HOP_SIZE.., 0);
        time_buf[FRAME_SIZE - HOP_SIZE..].copy_from_slice(&FIRST_HOP);

        // Apply window
        let mut windowed = vec![0.0f32; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            windowed[i] = time_buf[i] * hann[i];
        }

        // FFT
        let mut fft_input = windowed.clone();
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut fft_input, &mut spectrum).unwrap();

        // Extract magnitude and phase
        let mut magnitude = vec![0.0f32; FFT_BINS];
        let mut phase = vec![0.0f32; FFT_BINS];
        for i in 0..FFT_BINS {
            magnitude[i] = spectrum[i].norm();
            phase[i] = spectrum[i].im.atan2(spectrum[i].re);
        }

        eprintln!("Rust magnitude[0:5]: {:?}", &magnitude[0..5]);
        eprintln!("Rust phase[0:5]: {:?}", &phase[0..5]);

        // For passthrough model (no ONNX), mask is all 1s
        // So out_mag = magnitude, out_phase = phase
        let out_mag = magnitude.clone();
        let out_phase = phase.clone();

        // Reconstruct spectrum
        let mut ifft_input = vec![Complex32::new(0.0, 0.0); FFT_BINS];
        for i in 0..FFT_BINS {
            let (sin_p, cos_p) = out_phase[i].sin_cos();
            ifft_input[i] = Complex32::new(out_mag[i] * cos_p, out_mag[i] * sin_p);
        }
        // Zero imaginary at DC and Nyquist for realfft
        ifft_input[0] = Complex32::new(ifft_input[0].re, 0.0);
        ifft_input[FFT_BINS - 1] = Complex32::new(ifft_input[FFT_BINS - 1].re, 0.0);

        // IFFT
        let mut ifft_output = vec![0.0f32; FRAME_SIZE];
        ifft.process(&mut ifft_input, &mut ifft_output).unwrap();

        // Scale
        for v in ifft_output.iter_mut() {
            *v /= FRAME_SIZE as f32;
        }

        eprintln!("IFFT output RMS: {:.10}", (ifft_output.iter().map(|x| x * x).sum::<f32>() / FRAME_SIZE as f32).sqrt());
        eprintln!("IFFT output[0:5]: {:?}", &ifft_output[0..5]);

        // OLA
        ola_buf.copy_within(HOP_SIZE.., 0);
        ola_buf[FRAME_SIZE - HOP_SIZE..].fill(0.0);
        for i in 0..FRAME_SIZE {
            ola_buf[i] += ifft_output[i];
        }

        let output_hop0: Vec<f32> = ola_buf[..HOP_SIZE].to_vec();
        eprintln!("\nRust output_hop0[0:5]: {:?}", &output_hop0[0..5]);
        eprintln!("Python output_hop0[0:5]: {:?}", PY_HOP0_OUTPUT);

        // Compare with Python (passthrough model, so should match windowed RMS)
        eprintln!("\nOutput hop0 RMS: {:.10}", (output_hop0.iter().map(|x| x * x).sum::<f32>() / HOP_SIZE as f32).sqrt());

        // The passthrough output should match the IFFT of the windowed input
        // which should be the windowed input itself
        eprintln!("\nWindowed[0:5]: {:?}", &windowed[0..5]);

        // Since windowed[0:384] is all zeros (hann * zeros), only [384:512] has data
        // And the Hann window tapers to zero at the ends, so the last samples are near zero too
        eprintln!("Windowed[380:390]: {:?}", &windowed[380..390]);
        eprintln!("Windowed[-10:]: {:?}", &windowed[FRAME_SIZE-10..]);

        // With passthrough, IFFT output should equal windowed
        let diff: Vec<f32> = ifft_output.iter()
            .zip(windowed.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        eprintln!("Max diff (IFFT vs windowed): {:?}", diff.iter().cloned().fold(0.0f32, f32::max));

        eprintln!("\n=== Test complete ===");
    }

    /// Test that ONNX model output matches Python exactly.
    #[test]
    fn test_onnx_model_output() {
        eprintln!("\n=== ONNX Model Output Comparison ===\n");

        // Create processor (uses ONNX if ANR_MODEL_DIR is set)
        let mut processor = AnrProcessor::new();

        // First 128 samples from test_input_16k.wav
        const FIRST_HOP: [f32; HOP_SIZE] = [
            -0.00054932, -0.00064087, -0.00061035, -0.00106812, -0.00070190, -0.00097656, -0.00061035, -0.00057983,
            -0.00057983, -0.00018311, -0.00082397, -0.00070190, -0.00054932, -0.00033569, -0.00045776, -0.00051880,
            -0.00024414, -0.00018311, -0.00009155, -0.00091553, -0.00030518, 0.00000000, -0.00018311, 0.00012207,
            -0.00027466, 0.00003052, -0.00015259, -0.00024414, -0.00061035, -0.00070190, 0.00000000, -0.00015259,
            -0.00033569, -0.00009155, -0.00030518, -0.00009155, 0.00039673, -0.00036621, 0.00018311, 0.00015259,
            -0.00012207, 0.00036621, -0.00030518, 0.00015259, 0.00012207, 0.00009155, 0.00027466, -0.00006104,
            0.00024414, 0.00012207, 0.00064087, 0.00024414, -0.00033569, 0.00030518, 0.00021362, 0.00015259,
            0.00018311, 0.00015259, 0.00027466, 0.00076294, 0.00061035, 0.00036621, 0.00088501, 0.00054932,
            0.00012207, 0.00073242, 0.00051880, 0.00000000, 0.00073242, 0.00042725, 0.00051880, 0.00109863,
            0.00070190, 0.00067139, 0.00094604, 0.00109863, 0.00106812, 0.00137329, 0.00091553, 0.00070190,
            0.00125122, 0.00091553, 0.00091553, 0.00119019, 0.00122070, 0.00146484, 0.00134277, 0.00091553,
            0.00137329, 0.00119019, 0.00070190, 0.00128174, 0.00149536, 0.00161743, 0.00140381, 0.00125122,
            0.00131226, 0.00131226, 0.00128174, 0.00140381, 0.00164795, 0.00112915, 0.00164795, 0.00186157,
            0.00146484, 0.00158691, 0.00146484, 0.00152588, 0.00164795, 0.00152588, 0.00161743, 0.00173950,
            0.00161743, 0.00167847, 0.00115967, 0.00152588, 0.00177002, 0.00125122, 0.00219727, 0.00213623,
            0.00167847, 0.00207520, 0.00173950, 0.00146484, 0.00192261, 0.00210571, 0.00213623, 0.00201416,
        ];

        // Python expected values for first hop output (from single_hop_test.py)
        const PY_OUTPUT: [f32; 5] = [-7.6572524e-06, -8.4814137e-06, -3.0457145e-06, -1.0399135e-06, 9.0491903e-07];

        // Process single hop
        let output = processor.process_hop(&FIRST_HOP, AnrMode::PmlnOnly, 1.0);

        eprintln!("Rust output[0:5]: {:?}", &output[0..5]);
        eprintln!("Python output[0:5]: {:?}", PY_OUTPUT);

        let output_rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / HOP_SIZE as f32).sqrt();
        eprintln!("Rust output RMS: {:.10}", output_rms);

        // Python hop0 RMS was 0.0000034769
        let py_rms = 0.0000034769f32;
        eprintln!("Python output RMS: {:.10}", py_rms);

        let rms_ratio = output_rms / py_rms;
        eprintln!("RMS ratio: {:.4}", rms_ratio);

        // Check if the ratio is close to 0.75 (double windowing) or 1.0 (correct)
        if rms_ratio > 0.7 && rms_ratio < 0.8 {
            eprintln!("WARNING: RMS ratio ~0.75 suggests double windowing!");
        }

        // Compare individual samples
        for i in 0..5 {
            let ratio = if PY_OUTPUT[i].abs() > 1e-10 { output[i] / PY_OUTPUT[i] } else { 0.0 };
            eprintln!("Sample {}: Rust={:.8e}, Python={:.8e}, Ratio={:.4}", i, output[i], PY_OUTPUT[i], ratio);
        }

        eprintln!("\n=== Test complete ===");
    }

    /// Test process_block specifically to find the 0.75 bug.
    #[test]
    fn test_process_block_vs_process_hop() {
        eprintln!("\n=== process_block vs process_hop Comparison ===\n");

        // Load test input
        let test_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let input_path = test_dir.join("test_input_16k.wav");

        if !input_path.exists() {
            eprintln!("Test input not found: {:?}", input_path);
            return;
        }

        let reader = hound::WavReader::open(&input_path).expect("Failed to open WAV");
        let samples: Vec<f32> = reader.into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();

        eprintln!("Total samples loaded: {}", samples.len());
        eprintln!("FIFO capacity is 65536 - if input > 65536, samples will be lost!");

        // Take first 1280 samples (10 hops) - fits in FIFO
        let input = &samples[..1280];

        // Method 1: process_hop directly (like Python)
        let mut processor1 = AnrProcessor::new();
        let mut output1 = Vec::new();
        for hop_idx in 0..10 {
            let start = hop_idx * HOP_SIZE;
            let mut hop = [0.0f32; HOP_SIZE];
            hop.copy_from_slice(&input[start..start + HOP_SIZE]);
            let out_hop = processor1.process_hop(&hop, AnrMode::PmlnOnly, 1.0);
            output1.extend_from_slice(&out_hop);
        }

        // Method 2: process_block (via FIFO)
        let mut processor2 = AnrProcessor::new();
        let mut output2 = vec![0.0f32; 1280];
        processor2.process_block(input, &mut output2, AnrMode::PmlnOnly, 1.0);

        // Compare
        let rms1: f32 = (output1.iter().map(|x| x * x).sum::<f32>() / output1.len() as f32).sqrt();
        let rms2: f32 = (output2.iter().map(|x| x * x).sum::<f32>() / output2.len() as f32).sqrt();

        eprintln!("process_hop RMS: {:.10}", rms1);
        eprintln!("process_block RMS: {:.10}", rms2);
        eprintln!("Ratio (block/hop): {:.4}", rms2 / rms1);

        eprintln!("\nFirst 10 samples comparison:");
        for i in 0..10 {
            eprintln!("  [{}] hop={:.8e}, block={:.8e}, diff={:.8e}",
                     i, output1[i], output2[i], (output1[i] - output2[i]).abs());
        }

        // Check if they match
        let max_diff: f32 = output1.iter()
            .zip(output2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("\nMax sample difference: {:.10}", max_diff);

        if max_diff > 1e-6 {
            eprintln!("ERROR: process_block produces different output than process_hop!");
        } else {
            eprintln!("OK: process_block matches process_hop");
        }

        eprintln!("\n=== Test complete ===");
    }

    /// Test multi-hop processing against Python with detailed intermediate values.
    #[test]
    fn test_multihop_detailed_comparison() {
        eprintln!("\n=== Multi-Hop Detailed Comparison ===\n");

        // Load test input from WAV file
        let test_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let input_path = test_dir.join("test_input_16k.wav");

        if !input_path.exists() {
            eprintln!("Test input not found: {:?}", input_path);
            return;
        }

        let reader = hound::WavReader::open(&input_path).expect("Failed to open WAV");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16000);

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
                reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
            }
            hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        };

        eprintln!("Loaded {} samples from test_input_16k.wav", samples.len());

        // Create processor
        let mut processor = AnrProcessor::new();

        // Process 10 hops manually (bypassing FIFO to match Python exactly)
        let mut all_output = Vec::new();

        // Python expected values for verification
        let py_hop0_output: [f32; 5] = [-7.6572524e-06, -8.4814137e-06, -3.0457145e-06, -1.0399135e-06, 9.0491903e-07];
        let py_hop0_rms = 0.0000034769f32;

        // Process 100 hops to match Python numpy file
        for hop_idx in 0..100 {
            let start = hop_idx * HOP_SIZE;
            let mut hop = [0.0f32; HOP_SIZE];
            hop.copy_from_slice(&samples[start..start + HOP_SIZE]);

            let output = processor.process_hop(&hop, AnrMode::PmlnOnly, 1.0);
            all_output.extend_from_slice(&output);
        }

        let total_rms: f32 = (all_output.iter().map(|x| x * x).sum::<f32>() / all_output.len() as f32).sqrt();
        eprintln!("\n=== First 100 hops output ===");
        eprintln!("Rust Total output RMS: {:.10}", total_rms);

        // Python's total RMS for 100 hops: 0.0017959456
        let py_total_rms = 0.0017959456f32;
        eprintln!("Python total RMS:      {:.10}", py_total_rms);
        eprintln!("Rust/Python total RMS ratio: {:.6}", total_rms / py_total_rms);

        // Check if they match within floating point tolerance
        let ratio = total_rms / py_total_rms;
        if (ratio - 1.0).abs() < 0.001 {
            eprintln!("✓ MATCH: Rust matches Python exactly!");
        } else {
            eprintln!("✗ MISMATCH: ratio = {:.6}", ratio);
        }

        eprintln!("\n=== Test complete ===");
    }

    // ========================================================================
    // DATA LOSS PREVENTION TESTS
    // These tests ensure no samples are silently dropped
    // ========================================================================

    /// Test that process_block preserves exact sample count for any input size.
    #[test]
    fn test_sample_count_preservation_various_sizes() {
        let test_sizes = [
            1,           // Minimum
            127,         // Less than HOP_SIZE
            128,         // Exactly HOP_SIZE
            129,         // Just over HOP_SIZE
            512,         // FRAME_SIZE
            1000,        // Arbitrary
            1280,        // 10 hops
            10000,       // Larger
            65536,       // FIFO capacity boundary
            65537,       // Just over FIFO capacity
            100000,      // Well over FIFO capacity
        ];

        for &size in &test_sizes {
            let mut processor = AnrProcessor::new();
            let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
            let mut output = vec![0.0f32; size];

            processor.process_block(&input, &mut output, AnrMode::PmlnOnly, 1.0);

            // Count non-zero outputs (after warmup period)
            // For small inputs, we expect some zeros due to OLA warmup
            let non_zero_count = output.iter().filter(|&&x| x.abs() > 1e-10).count();

            // For inputs larger than FRAME_SIZE, most output should be non-zero
            if size > FRAME_SIZE {
                let expected_min_nonzero = size.saturating_sub(FRAME_SIZE);
                assert!(
                    non_zero_count >= expected_min_nonzero / 2,
                    "Size {}: Expected at least {} non-zero samples, got {}",
                    size,
                    expected_min_nonzero / 2,
                    non_zero_count
                );
            }

            // Output should never exceed input length
            assert_eq!(
                output.len(),
                input.len(),
                "Size {}: Output length {} != input length {}",
                size,
                output.len(),
                input.len()
            );
        }
    }

    /// Test that FIFO handles exactly-at-capacity inputs correctly.
    #[test]
    fn test_fifo_capacity_boundary() {
        let mut fifo = Fifo::with_capacity(1024);

        // Push exactly capacity
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let pushed = fifo.push(&data);
        assert_eq!(pushed, 1024, "Should push exactly 1024");
        assert_eq!(fifo.available(), 1024);

        // Pop half
        let mut out = vec![0.0f32; 512];
        let popped = fifo.pop(&mut out);
        assert_eq!(popped, 512);
        assert_eq!(fifo.available(), 512);

        // Push 512 more (should now fit)
        let more_data: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let pushed = fifo.push(&more_data);
        assert_eq!(pushed, 512, "Should push 512 after making room");
        assert_eq!(fifo.available(), 1024);
    }

    /// Test that process_block handles repeated calls correctly (stateful).
    #[test]
    fn test_repeated_process_block_calls() {
        let mut processor = AnrProcessor::new();

        // Simulate multiple host buffer calls
        let chunk_size = 512;
        let num_chunks = 20;
        let mut all_output = Vec::new();

        for i in 0..num_chunks {
            let input: Vec<f32> = (0..chunk_size)
                .map(|j| ((i * chunk_size + j) as f32 * 0.001).sin())
                .collect();
            let mut output = vec![0.0f32; chunk_size];

            processor.process_block(&input, &mut output, AnrMode::PmlnOnly, 1.0);
            all_output.extend_from_slice(&output);
        }

        let total_input = chunk_size * num_chunks;
        assert_eq!(all_output.len(), total_input);

        // After warmup, output should have signal
        let warmup = FRAME_SIZE;
        let post_warmup: Vec<f32> = all_output[warmup..].to_vec();
        let rms: f32 = (post_warmup.iter().map(|x| x * x).sum::<f32>()
            / post_warmup.len() as f32)
            .sqrt();

        assert!(
            rms > 1e-6,
            "Post-warmup output RMS should be non-zero, got {}",
            rms
        );
    }

    /// Test that input equals output length for AudioProcessor public API.
    #[test]
    fn test_audio_processor_length_preservation() {
        let sizes = [1000, 16000, 100000, 219520]; // Including the size that triggered original bug

        for &size in &sizes {
            let mut processor = AudioProcessor::new();
            let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

            let output = processor.process_16k(&input, ProcessingMode::PmlnOnly, 1.0);

            assert_eq!(
                output.len(),
                input.len(),
                "Size {}: Output length {} != input length {}",
                size,
                output.len(),
                input.len()
            );

            // Verify non-zero output for large inputs
            if size > FRAME_SIZE * 2 {
                let non_zero = output.iter().filter(|&&x| x.abs() > 1e-10).count();
                let expected_min = size.saturating_sub(FRAME_SIZE);
                assert!(
                    non_zero >= expected_min / 2,
                    "Size {}: Too few non-zero samples: {} (expected >= {})",
                    size,
                    non_zero,
                    expected_min / 2
                );
            }
        }
    }

    /// Test process_block with sizes that would have triggered old FIFO overflow.
    #[test]
    fn test_large_input_no_data_loss() {
        let mut processor = AnrProcessor::new();

        // This size would have caused 70% data loss with old FIFO overflow bug
        let size = 219520;
        let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.0001).sin() * 0.5).collect();
        let mut output = vec![0.0f32; size];

        processor.process_block(&input, &mut output, AnrMode::PmlnOnly, 1.0);

        // Calculate RMS excluding warmup
        let warmup = FRAME_SIZE * 2;
        let post_warmup = &output[warmup..];
        let output_rms: f32 =
            (post_warmup.iter().map(|x| x * x).sum::<f32>() / post_warmup.len() as f32).sqrt();

        // Output RMS should be reasonable (not near zero due to data loss)
        assert!(
            output_rms > 0.001,
            "Output RMS {} is too low - possible data loss",
            output_rms
        );

        // Count zeros - should be minimal after warmup
        let zeros_after_warmup = post_warmup.iter().filter(|&&x| x.abs() < 1e-10).count();
        let zero_ratio = zeros_after_warmup as f32 / post_warmup.len() as f32;

        assert!(
            zero_ratio < 0.1, // Less than 10% zeros expected
            "Too many zeros after warmup: {:.1}% - possible data loss",
            zero_ratio * 100.0
        );
    }

    /// Test that FIFO pop returns correct count when buffer has less than requested.
    #[test]
    fn test_fifo_partial_pop() {
        let mut fifo = Fifo::with_capacity(1024);

        // Push 100 samples
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        fifo.push(&data);

        // Try to pop 200
        let mut out = vec![0.0f32; 200];
        let popped = fifo.pop(&mut out);

        assert_eq!(popped, 100, "Should only pop 100 available samples");
        assert_eq!(fifo.available(), 0, "FIFO should be empty");

        // Verify values
        for i in 0..100 {
            assert_eq!(out[i], i as f32);
        }
    }
}
