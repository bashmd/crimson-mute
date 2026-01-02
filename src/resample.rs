//! Audio resampling for host SR ↔ 16kHz conversion.
//!
//! Uses rubato's sinc resamplers with the decoupled pattern:
//! - SincFixedOut for downsampling: variable input → fixed HOP_SIZE output
//! - SincFixedIn for upsampling: fixed HOP_SIZE input → variable output

use rubato::{
    Resampler as RubatoResampler, SincFixedIn, SincFixedOut, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};

const TARGET_SR: usize = 16_000;
const HOP_SIZE: usize = 128;

fn sinc_params() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.925,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    }
}

/// Downsampler: host SR → 16kHz, produces exactly HOP_SIZE samples per call.
/// Uses SincFixedOut (variable input → fixed output).
pub struct Downsampler {
    resampler: SincFixedOut<f32>,
    host_sr: usize,
    /// Temp buffers for rubato (single channel)
    input_buf: Vec<Vec<f32>>,
    output_buf: Vec<Vec<f32>>,
}

impl Downsampler {
    /// Create a new downsampler for the given host sample rate.
    /// Returns None if host_sr == 16kHz (no resampling needed).
    pub fn new(host_sr: f32) -> Option<Self> {
        let host_sr_int = host_sr as usize;

        if (host_sr - TARGET_SR as f32).abs() < 1.0 {
            return None;
        }

        // Ratio is output_rate / input_rate = 16000 / host_sr
        let ratio = TARGET_SR as f64 / host_sr_int as f64;
        let resampler = SincFixedOut::<f32>::new(
            ratio,
            1.0, // max relative ratio (no dynamic resampling)
            sinc_params(),
            HOP_SIZE,
            1, // mono
        )
        .ok()?;

        let input_frames_max = resampler.input_frames_max();

        Some(Self {
            resampler,
            host_sr: host_sr_int,
            input_buf: vec![vec![0.0; input_frames_max]],
            output_buf: vec![vec![0.0; HOP_SIZE]],
        })
    }

    /// Returns how many host-rate samples are needed to produce the next HOP_SIZE frame.
    pub fn input_frames_needed(&self) -> usize {
        self.resampler.input_frames_next()
    }

    /// Process exactly `input_frames_needed()` samples, returns exactly HOP_SIZE samples at 16kHz.
    /// Panics if input length doesn't match `input_frames_needed()`.
    pub fn process(&mut self, input: &[f32]) -> [f32; HOP_SIZE] {
        let needed = self.resampler.input_frames_next();
        assert_eq!(
            input.len(),
            needed,
            "Downsampler: expected {} samples, got {}",
            needed,
            input.len()
        );

        // Copy input to rubato buffer
        self.input_buf[0].clear();
        self.input_buf[0].extend_from_slice(input);

        // Ensure output buffer is sized
        self.output_buf[0].clear();
        self.output_buf[0].resize(HOP_SIZE, 0.0);

        // Process
        match self.resampler.process_into_buffer(&self.input_buf, &mut self.output_buf, None) {
            Ok((_in_used, out_len)) => {
                debug_assert_eq!(out_len, HOP_SIZE, "SincFixedOut should produce exactly HOP_SIZE");
            }
            Err(e) => {
                nih_plug::prelude::nih_error!("Downsampler error: {:?}", e);
            }
        }

        let mut result = [0.0f32; HOP_SIZE];
        result.copy_from_slice(&self.output_buf[0][..HOP_SIZE]);
        result
    }

    /// Get the output delay in 16kHz samples.
    pub fn output_delay(&self) -> usize {
        self.resampler.output_delay()
    }

    /// Get host sample rate.
    pub fn host_sr(&self) -> usize {
        self.host_sr
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.resampler.reset();
    }
}

/// Upsampler: 16kHz → host SR, consumes exactly HOP_SIZE samples per call.
/// Uses SincFixedIn (fixed input → variable output).
pub struct Upsampler {
    resampler: SincFixedIn<f32>,
    host_sr: usize,
    /// Temp buffers for rubato (single channel)
    input_buf: Vec<Vec<f32>>,
    output_buf: Vec<Vec<f32>>,
    /// Maximum output frames per call
    output_max: usize,
}

impl Upsampler {
    /// Create a new upsampler for the given host sample rate.
    /// Returns None if host_sr == 16kHz (no resampling needed).
    pub fn new(host_sr: f32) -> Option<Self> {
        let host_sr_int = host_sr as usize;

        if (host_sr - TARGET_SR as f32).abs() < 1.0 {
            return None;
        }

        // Ratio is output_rate / input_rate = host_sr / 16000
        let ratio = host_sr_int as f64 / TARGET_SR as f64;
        let resampler = SincFixedIn::<f32>::new(
            ratio,
            1.0, // max relative ratio
            sinc_params(),
            HOP_SIZE,
            1, // mono
        )
        .ok()?;

        let output_max = resampler.output_frames_max();

        Some(Self {
            resampler,
            host_sr: host_sr_int,
            input_buf: vec![vec![0.0; HOP_SIZE]],
            output_buf: vec![vec![0.0; output_max]],
            output_max,
        })
    }

    /// Process exactly HOP_SIZE samples at 16kHz, returns variable number of host-rate samples.
    /// The returned slice is valid until the next call to process().
    pub fn process(&mut self, input: &[f32; HOP_SIZE]) -> &[f32] {
        // Copy input to rubato buffer
        self.input_buf[0].clear();
        self.input_buf[0].extend_from_slice(input);

        // Ensure output buffer is sized
        self.output_buf[0].clear();
        self.output_buf[0].resize(self.output_max, 0.0);

        // Process
        match self.resampler.process_into_buffer(&self.input_buf, &mut self.output_buf, None) {
            Ok((_in_used, out_len)) => {
                &self.output_buf[0][..out_len]
            }
            Err(e) => {
                nih_plug::prelude::nih_error!("Upsampler error: {:?}", e);
                &[]
            }
        }
    }

    /// Get the output delay in host-rate samples.
    pub fn output_delay(&self) -> usize {
        self.resampler.output_delay()
    }

    /// Get expected output frames per call (approximate).
    pub fn expected_output_frames(&self) -> usize {
        (HOP_SIZE as f64 * self.host_sr as f64 / TARGET_SR as f64) as usize
    }

    /// Get host sample rate.
    pub fn host_sr(&self) -> usize {
        self.host_sr
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.resampler.reset();
    }
}

/// Combined latency calculation for the resampling chain.
pub fn calculate_resample_latency(downsampler: &Downsampler, upsampler: &Upsampler) -> u32 {
    let host_sr = downsampler.host_sr() as f64;

    // Downsampler delay (in 16kHz samples) converted to host rate
    let down_delay_16k = downsampler.output_delay();
    let down_delay_host = (down_delay_16k as f64 * host_sr / TARGET_SR as f64) as u32;

    // Upsampler delay (already in host-rate samples)
    let up_delay = upsampler.output_delay() as u32;

    // Initial input buffering (need input_frames_needed before first output)
    let input_buffer_delay = downsampler.input_frames_needed() as u32;

    down_delay_host + up_delay + input_buffer_delay
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    // ========================================================================
    // Downsampler Tests
    // ========================================================================

    #[test]
    fn downsampler_creates_for_48khz() {
        let ds = Downsampler::new(48000.0);
        assert!(ds.is_some(), "Should create downsampler for 48kHz");
        let ds = ds.unwrap();
        assert_eq!(ds.host_sr(), 48000);
    }

    #[test]
    fn downsampler_creates_for_44100hz() {
        let ds = Downsampler::new(44100.0);
        assert!(ds.is_some(), "Should create downsampler for 44.1kHz");
        let ds = ds.unwrap();
        assert_eq!(ds.host_sr(), 44100);
    }

    #[test]
    fn downsampler_returns_none_for_16khz() {
        let ds = Downsampler::new(16000.0);
        assert!(ds.is_none(), "Should return None for 16kHz (no resampling needed)");
    }

    #[test]
    fn downsampler_produces_exact_hop_size() {
        let mut ds = Downsampler::new(48000.0).unwrap();

        // Generate input of exactly the required size
        let input_needed = ds.input_frames_needed();
        let input = generate_sine(1000.0, 48000.0, input_needed);

        let output = ds.process(&input);

        assert_eq!(output.len(), HOP_SIZE, "Output must be exactly HOP_SIZE ({})", HOP_SIZE);
    }

    #[test]
    fn downsampler_input_frames_needed_is_reasonable() {
        let ds = Downsampler::new(48000.0).unwrap();
        let needed = ds.input_frames_needed();

        // At 48kHz → 16kHz (3:1 ratio), we need ~3 * HOP_SIZE input for HOP_SIZE output
        // Plus some for the sinc filter
        assert!(needed >= HOP_SIZE * 3, "Should need at least 3x HOP_SIZE input");
        assert!(needed < HOP_SIZE * 6, "Should need less than 6x HOP_SIZE input");
    }

    #[test]
    fn downsampler_multiple_frames_consistent() {
        let mut ds = Downsampler::new(48000.0).unwrap();

        // Process multiple frames
        for i in 0..10 {
            let input_needed = ds.input_frames_needed();
            let input = generate_sine(1000.0, 48000.0, input_needed);
            let output = ds.process(&input);

            assert_eq!(
                output.len(),
                HOP_SIZE,
                "Frame {} should produce exactly HOP_SIZE samples",
                i
            );
        }
    }

    #[test]
    fn downsampler_preserves_signal_energy() {
        let mut ds = Downsampler::new(48000.0).unwrap();

        // Process enough frames to get past initial transient
        let warmup_frames = 10;
        for _ in 0..warmup_frames {
            let input_needed = ds.input_frames_needed();
            let input = generate_sine(1000.0, 48000.0, input_needed);
            ds.process(&input);
        }

        // Now measure energy
        let input_needed = ds.input_frames_needed();
        let input = generate_sine_with_amplitude(1000.0, 0.5, 48000.0, input_needed);
        let input_rms = compute_rms(&input);

        let output = ds.process(&input);
        let output_rms = compute_rms(&output);

        // Energy should be roughly preserved (within 6dB)
        let ratio = output_rms / input_rms;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Energy ratio {} should be close to 1.0",
            ratio
        );
    }

    #[test]
    fn downsampler_reset_clears_state() {
        let mut ds = Downsampler::new(48000.0).unwrap();

        // Process some frames
        for _ in 0..5 {
            let input_needed = ds.input_frames_needed();
            let input = generate_sine(1000.0, 48000.0, input_needed);
            ds.process(&input);
        }

        // Reset
        ds.reset();

        // After reset, input_frames_needed should be back to initial value
        // (we can't easily test internal state, but behavior should be consistent)
        let input_needed = ds.input_frames_needed();
        assert!(input_needed > 0, "Should need input after reset");
    }

    // ========================================================================
    // Upsampler Tests
    // ========================================================================

    #[test]
    fn upsampler_creates_for_48khz() {
        let us = Upsampler::new(48000.0);
        assert!(us.is_some(), "Should create upsampler for 48kHz");
    }

    #[test]
    fn upsampler_creates_for_44100hz() {
        let us = Upsampler::new(44100.0);
        assert!(us.is_some(), "Should create upsampler for 44.1kHz");
    }

    #[test]
    fn upsampler_returns_none_for_16khz() {
        let us = Upsampler::new(16000.0);
        assert!(us.is_none(), "Should return None for 16kHz (no resampling needed)");
    }

    #[test]
    fn upsampler_output_length_reasonable() {
        let mut us = Upsampler::new(48000.0).unwrap();

        // For 48kHz, expected output ≈ HOP_SIZE * 3 = 384
        let input: [f32; HOP_SIZE] = [0.0; HOP_SIZE];
        let output = us.process(&input);

        // First call may be shorter due to filter warmup
        // But should produce some output
        assert!(output.len() > 0, "Should produce some output");
    }

    #[test]
    fn upsampler_steady_state_output_correct() {
        let mut us = Upsampler::new(48000.0).unwrap();

        // Warm up the upsampler
        for _ in 0..10 {
            let input: [f32; HOP_SIZE] = [0.0; HOP_SIZE];
            us.process(&input);
        }

        // Now check steady-state output length
        let input = generate_sine(1000.0, 16000.0, HOP_SIZE);
        let mut input_arr = [0.0f32; HOP_SIZE];
        input_arr.copy_from_slice(&input);

        let output = us.process(&input_arr);

        // At 48kHz, expect ~384 samples (128 * 3), allow some variance
        let expected = (HOP_SIZE as f64 * 48000.0 / 16000.0) as usize;
        assert!(
            (output.len() as i32 - expected as i32).abs() <= 5,
            "Output length {} should be close to expected {}",
            output.len(),
            expected
        );
    }

    #[test]
    fn upsampler_preserves_signal_energy() {
        let mut us = Upsampler::new(48000.0).unwrap();

        // Warm up
        for _ in 0..10 {
            let input: [f32; HOP_SIZE] = [0.0; HOP_SIZE];
            us.process(&input);
        }

        // Measure energy
        let input = generate_sine_with_amplitude(1000.0, 0.5, 16000.0, HOP_SIZE);
        let mut input_arr = [0.0f32; HOP_SIZE];
        input_arr.copy_from_slice(&input);
        let input_rms = compute_rms(&input);

        let output = us.process(&input_arr);
        let output_rms = compute_rms(output);

        // Energy should be roughly preserved (within 6dB)
        let ratio = output_rms / input_rms;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Energy ratio {} should be close to 1.0",
            ratio
        );
    }

    #[test]
    fn upsampler_reset_clears_state() {
        let mut us = Upsampler::new(48000.0).unwrap();

        // Process some frames with signal
        for _ in 0..5 {
            let input = generate_sine(1000.0, 16000.0, HOP_SIZE);
            let mut input_arr = [0.0f32; HOP_SIZE];
            input_arr.copy_from_slice(&input);
            us.process(&input_arr);
        }

        // Reset
        us.reset();

        // Process silence - should get mostly silence out (after a brief transient)
        let input: [f32; HOP_SIZE] = [0.0; HOP_SIZE];
        for _ in 0..5 {
            us.process(&input);
        }

        let output = us.process(&input);
        let rms = compute_rms(output);

        assert!(rms < 0.01, "After reset and silence input, output should be near-silent");
    }

    // ========================================================================
    // Round-trip Tests (Downsample → Upsample)
    // ========================================================================

    #[test]
    fn roundtrip_preserves_signal_48khz() {
        let mut ds = Downsampler::new(48000.0).unwrap();
        let mut us = Upsampler::new(48000.0).unwrap();

        // Generate a longer test signal at 48kHz
        // Use a frequency that's well below Nyquist at 16kHz (8kHz)
        let test_freq = 500.0; // 500 Hz - well within 16kHz bandwidth
        let total_samples = 48000; // 1 second
        let input_signal = generate_sine(test_freq, 48000.0, total_samples);

        let mut output_signal = Vec::new();
        let mut input_pos = 0;

        // Process through downsample → upsample chain
        while input_pos < input_signal.len() {
            let needed = ds.input_frames_needed();
            if input_pos + needed > input_signal.len() {
                break;
            }

            // Downsample
            let downsampled = ds.process(&input_signal[input_pos..input_pos + needed]);
            input_pos += needed;

            // Upsample
            let upsampled = us.process(&downsampled);
            output_signal.extend_from_slice(upsampled);
        }

        // Skip initial transient (first ~1000 samples)
        let skip = 2000;
        if output_signal.len() > skip + 10000 {
            let input_trimmed = &input_signal[skip..skip + 10000];
            let output_trimmed = &output_signal[skip..skip + 10000];

            // Check frequency is preserved
            let input_freq = estimate_frequency(input_trimmed, 48000.0);
            let output_freq = estimate_frequency(output_trimmed, 48000.0);

            assert!(
                (input_freq - output_freq).abs() < 50.0,
                "Frequency should be preserved: input={}, output={}",
                input_freq,
                output_freq
            );

            // Check signal level is preserved (within 6dB)
            let input_rms = compute_rms(input_trimmed);
            let output_rms = compute_rms(output_trimmed);
            let ratio = output_rms / input_rms;

            assert!(
                ratio > 0.5 && ratio < 2.0,
                "Signal level should be preserved: ratio={}",
                ratio
            );
        }
    }

    #[test]
    fn roundtrip_rejects_frequencies_above_8khz() {
        let mut ds = Downsampler::new(48000.0).unwrap();
        let mut us = Upsampler::new(48000.0).unwrap();

        // Generate a signal at 10kHz - above 8kHz Nyquist for 16kHz
        let test_freq = 10000.0;
        let total_samples = 24000;
        let input_signal = generate_sine(test_freq, 48000.0, total_samples);
        let input_rms = compute_rms(&input_signal);

        let mut output_signal = Vec::new();
        let mut input_pos = 0;

        while input_pos < input_signal.len() {
            let needed = ds.input_frames_needed();
            if input_pos + needed > input_signal.len() {
                break;
            }

            let downsampled = ds.process(&input_signal[input_pos..input_pos + needed]);
            input_pos += needed;

            let upsampled = us.process(&downsampled);
            output_signal.extend_from_slice(upsampled);
        }

        // The 10kHz signal should be heavily attenuated
        let skip = 2000;
        if output_signal.len() > skip + 5000 {
            let output_trimmed = &output_signal[skip..skip + 5000];
            let output_rms = compute_rms(output_trimmed);

            // Should be attenuated by at least 20dB
            let attenuation_db = 20.0 * (input_rms / output_rms).log10();
            assert!(
                attenuation_db > 20.0,
                "10kHz signal should be attenuated >20dB, got {}dB",
                attenuation_db
            );
        }
    }

    #[test]
    fn roundtrip_multiple_sample_rates() {
        for host_sr in [44100.0, 48000.0, 96000.0] {
            let ds = Downsampler::new(host_sr);
            let us = Upsampler::new(host_sr);

            assert!(
                ds.is_some() && us.is_some(),
                "Should create resamplers for {}Hz",
                host_sr
            );

            let mut ds = ds.unwrap();
            let mut us = us.unwrap();

            // Quick sanity check - process a few frames
            for _ in 0..10 {
                let needed = ds.input_frames_needed();
                let input = generate_sine(500.0, host_sr, needed);
                let downsampled = ds.process(&input);
                assert_eq!(downsampled.len(), HOP_SIZE);

                let upsampled = us.process(&downsampled);
                assert!(upsampled.len() > 0);
            }
        }
    }

    // ========================================================================
    // Latency Tests
    // ========================================================================

    #[test]
    fn latency_calculation_is_reasonable() {
        let ds = Downsampler::new(48000.0).unwrap();
        let us = Upsampler::new(48000.0).unwrap();

        let latency = calculate_resample_latency(&ds, &us);

        // Latency should be positive and reasonable
        // At 48kHz, expect a few hundred to a couple thousand samples
        assert!(latency > 100, "Latency {} should be > 100 samples", latency);
        assert!(latency < 5000, "Latency {} should be < 5000 samples", latency);
    }
}
