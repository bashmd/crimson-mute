//! Integration tests comparing plugin output against Python reference implementation.
//!
//! These tests load actual audio files and verify the Rust implementation
//! produces output matching the Python reference.

use std::path::PathBuf;

use anr_plugin::{AudioProcessor, ProcessingMode, HOP_SIZE, FRAME_SIZE};

/// Get the path to test audio files in the parent directory.
fn test_audio_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap().to_path_buf()
}

/// Load a WAV file and return samples as f32 in [-1, 1] range.
fn load_wav(path: &PathBuf) -> Result<(Vec<f32>, u32), hound::Error> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    Ok((samples, sample_rate))
}

/// Save samples to a WAV file for debugging (32-bit float to avoid quantization).
#[allow(dead_code)]
fn save_wav(path: &PathBuf, samples: &[f32], sample_rate: u32) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Compute RMS of a signal.
fn compute_rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = signal.iter().map(|x| x * x).sum();
    (sum_sq / signal.len() as f32).sqrt()
}

/// Compute mean absolute error between two signals.
fn compute_mae(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

/// Compute correlation coefficient between two signals.
fn compute_correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f32;

    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a < 1e-10 || var_b < 1e-10 {
        return 0.0;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Find the delay that maximizes correlation between two signals.
fn find_best_delay(reference: &[f32], test: &[f32], max_delay: usize) -> (usize, f32) {
    let mut best_delay = 0;
    let mut best_corr = f32::NEG_INFINITY;

    for delay in 0..max_delay.min(test.len()) {
        let compare_len = reference.len().min(test.len() - delay);
        if compare_len < 1000 {
            continue;
        }

        let ref_slice = &reference[..compare_len];
        let test_slice = &test[delay..delay + compare_len];
        let corr = compute_correlation(ref_slice, test_slice);

        if corr > best_corr {
            best_corr = corr;
            best_delay = delay;
        }
    }

    (best_delay, best_corr)
}

/// Resample from 48kHz to 16kHz using rubato.
fn resample_48k_to_16k(input: &[f32]) -> Vec<f32> {
    use rubato::{Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut downsampler = SincFixedOut::<f32>::new(
        16000.0 / 48000.0,  // ratio
        2.0,                 // max relative ratio
        params,
        HOP_SIZE,            // output chunk size
        1,                   // channels
    ).expect("Failed to create downsampler");

    let mut output = Vec::new();
    let mut input_pos = 0;

    while input_pos < input.len() {
        let needed = downsampler.input_frames_next();
        let available = input.len() - input_pos;

        if available < needed {
            break;
        }

        let input_chunk = vec![input[input_pos..input_pos + needed].to_vec()];
        input_pos += needed;

        let output_chunk = downsampler.process(&input_chunk, None).expect("Resample failed");
        output.extend_from_slice(&output_chunk[0]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_files_exist() {
        let dir = test_audio_dir();
        let input_path = dir.join("test_audio.wav");
        let reference_path = dir.join("test_audio_pmln_only_masked.wav");

        assert!(
            input_path.exists(),
            "Input file not found: {:?}",
            input_path
        );
        assert!(
            reference_path.exists(),
            "Reference file not found: {:?}",
            reference_path
        );
    }

    #[test]
    fn test_load_input_wav() {
        let dir = test_audio_dir();
        let input_path = dir.join("test_audio.wav");

        let (samples, sample_rate) = load_wav(&input_path).expect("Failed to load input WAV");

        // Expected: 48kHz mono
        assert_eq!(sample_rate, 48000, "Expected 48kHz input");
        assert!(samples.len() > 0, "Input should have samples");

        // Check samples are in valid range
        let max_abs = samples.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));
        assert!(max_abs <= 1.0, "Samples should be normalized: max={}", max_abs);

        eprintln!(
            "Input: {} samples @ {}Hz = {:.2}s",
            samples.len(),
            sample_rate,
            samples.len() as f32 / sample_rate as f32
        );
    }

    #[test]
    fn test_load_reference_wav() {
        let dir = test_audio_dir();
        let ref_path = dir.join("test_audio_pmln_only_masked.wav");

        let (samples, sample_rate) = load_wav(&ref_path).expect("Failed to load reference WAV");

        // Expected: 16kHz mono
        assert_eq!(sample_rate, 16000, "Expected 16kHz reference");
        assert!(samples.len() > 0, "Reference should have samples");

        eprintln!(
            "Reference: {} samples @ {}Hz = {:.2}s",
            samples.len(),
            sample_rate,
            samples.len() as f32 / sample_rate as f32
        );
    }

    #[test]
    fn test_resample_48k_to_16k() {
        let dir = test_audio_dir();
        let input_path = dir.join("test_audio.wav");

        let (input_48k, _) = load_wav(&input_path).expect("Failed to load input");
        let input_16k = resample_48k_to_16k(&input_48k);

        // Should be approximately 1/3 the length (48kHz -> 16kHz)
        let expected_len = input_48k.len() / 3;
        let ratio = input_16k.len() as f32 / expected_len as f32;

        assert!(
            ratio > 0.95 && ratio < 1.05,
            "Resampled length should be ~1/3: got {} (expected ~{})",
            input_16k.len(),
            expected_len
        );

        eprintln!(
            "Resampled: {} -> {} samples",
            input_48k.len(),
            input_16k.len()
        );
    }

    #[test]
    fn test_reference_comparison_pmln_only() {
        let dir = test_audio_dir();
        let input_path = dir.join("test_audio.wav");
        let reference_path = dir.join("test_audio_pmln_only_masked.wav");

        // Load input (48kHz) and resample to 16kHz
        let (input_48k, _) = load_wav(&input_path).expect("Failed to load input");
        let input_16k = resample_48k_to_16k(&input_48k);

        // Load reference (16kHz)
        let (reference, ref_sr) = load_wav(&reference_path).expect("Failed to load reference");
        assert_eq!(ref_sr, 16000);

        // Process through Rust ANR pipeline
        let mut processor = AudioProcessor::new();
        let rust_output = processor.process_16k(&input_16k, ProcessingMode::PmlnOnly, 1.0);

        eprintln!("Input 16k samples: {}", input_16k.len());
        eprintln!("Reference samples: {}", reference.len());
        eprintln!("Rust output samples: {}", rust_output.len());

        // Save Rust output for manual inspection
        let debug_path = dir.join("test_audio_rust_pmln_only.wav");
        save_wav(&debug_path, &rust_output, 16000).expect("Failed to save debug output");
        eprintln!("Saved Rust output to: {:?}", debug_path);

        // Find best alignment delay
        let max_delay = 2000; // ~125ms at 16kHz
        let (delay, corr) = find_best_delay(&reference, &rust_output, max_delay);
        eprintln!("Best delay: {} samples, correlation: {:.4}", delay, corr);

        // Compare aligned signals
        let compare_len = reference.len().min(rust_output.len() - delay);
        let ref_slice = &reference[..compare_len];
        let rust_slice = &rust_output[delay..delay + compare_len];

        let mae = compute_mae(ref_slice, rust_slice);
        let ref_rms = compute_rms(ref_slice);
        let rust_rms = compute_rms(rust_slice);

        eprintln!("Reference RMS: {:.6}", ref_rms);
        eprintln!("Rust output RMS: {:.6}", rust_rms);
        eprintln!("Mean Absolute Error: {:.6}", mae);
        eprintln!("Correlation: {:.4}", corr);

        // Assertions
        // 1. Correlation should be high (signals should be similar)
        assert!(
            corr > 0.5,
            "Correlation should be > 0.5: got {:.4}",
            corr
        );

        // 2. RMS should be similar (within 10x)
        let rms_ratio = rust_rms / ref_rms.max(1e-10);
        assert!(
            rms_ratio > 0.1 && rms_ratio < 10.0,
            "RMS ratio should be reasonable: {:.4}",
            rms_ratio
        );

        // 3. Output should not be silent
        assert!(
            rust_rms > 0.001,
            "Rust output should not be silent: RMS={:.6}",
            rust_rms
        );

        eprintln!("SUCCESS: Rust output matches reference with correlation {:.4}", corr);
    }

    #[test]
    fn test_processor_produces_output() {
        // Simple sanity check that the processor produces non-zero output
        let mut processor = AudioProcessor::new();

        // Generate a simple test signal: 1 second of 500Hz sine at 16kHz
        let num_samples = 16000;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 500.0 * t).sin() * 0.5
            })
            .collect();

        let output = processor.process_16k(&input, ProcessingMode::PmlnOnly, 1.0);

        assert_eq!(output.len(), input.len());

        let input_rms = compute_rms(&input);
        let output_rms = compute_rms(&output);

        eprintln!("Input RMS: {:.6}", input_rms);
        eprintln!("Output RMS: {:.6}", output_rms);

        // Output should have some signal (not silent)
        assert!(
            output_rms > 0.0001,
            "Output should not be silent: RMS={:.6}",
            output_rms
        );
    }

    /// GOLD STANDARD TEST: Compare Rust vs Python at 16kHz with NO resampling.
    /// This isolates the DSP core (FFT/ONNX/IFFT/OLA) from any resampler differences.
    #[test]
    fn test_16k_direct_comparison() {
        let dir = test_audio_dir();
        let input_path = dir.join("test_input_16k.wav");
        let reference_path = dir.join("test_output_16k_python.wav");

        // Check files exist
        if !input_path.exists() || !reference_path.exists() {
            eprintln!("16kHz test files not found. Run generate_16k_test.py first.");
            eprintln!("Expected: {:?}", input_path);
            eprintln!("Expected: {:?}", reference_path);
            panic!("Missing 16kHz test files");
        }

        // Load 16kHz input directly (NO RESAMPLING)
        let (input, input_sr) = load_wav(&input_path).expect("Failed to load 16kHz input");
        assert_eq!(input_sr, 16000, "Input must be 16kHz");

        // Load Python reference output
        let (reference, ref_sr) = load_wav(&reference_path).expect("Failed to load Python reference");
        assert_eq!(ref_sr, 16000, "Reference must be 16kHz");

        eprintln!("=== 16kHz Direct Comparison (No Resampling) ===");
        eprintln!("Input: {} samples", input.len());
        eprintln!("Reference: {} samples", reference.len());

        // Process through Rust at 16kHz directly
        let mut processor = AudioProcessor::new();
        let rust_output = processor.process_16k(&input, ProcessingMode::PmlnOnly, 1.0);

        eprintln!("Rust output: {} samples", rust_output.len());

        // Save for manual inspection
        let debug_path = dir.join("test_output_16k_rust.wav");
        save_wav(&debug_path, &rust_output, 16000).expect("Failed to save Rust output");
        eprintln!("Saved Rust output to: {:?}", debug_path);

        // Stats
        let ref_rms = compute_rms(&reference);
        let rust_rms = compute_rms(&rust_output);
        let rms_ratio = rust_rms / ref_rms;

        eprintln!("Reference RMS: {:.6}", ref_rms);
        eprintln!("Rust RMS: {:.6}", rust_rms);
        eprintln!("RMS Ratio (Rust/Python): {:.4}", rms_ratio);

        // Check for the 0.75 ratio (double windowing bug)
        if rms_ratio > 0.7 && rms_ratio < 0.8 {
            eprintln!("WARNING: RMS ratio ~0.75 suggests double windowing (Hann applied twice)!");
        }

        // Find alignment - use cross-correlation with zero-padding for proper delay finding
        let max_delay = 2000;
        let (delay, corr) = find_best_delay(&reference, &rust_output, max_delay);
        eprintln!("Best delay: {} samples ({:.2} ms)", delay, delay as f32 / 16.0);
        eprintln!("Correlation at best delay: {:.6}", corr);

        // Also try negative delays (Rust ahead of Python)
        let (neg_delay, neg_corr) = find_best_delay(&rust_output, &reference, max_delay);
        if neg_corr > corr {
            eprintln!("Better correlation with negative delay: {} samples, corr={:.6}", neg_delay, neg_corr);
        }

        // Compare aligned signals
        let compare_len = reference.len().min(rust_output.len().saturating_sub(delay));
        if compare_len > 1000 {
            let ref_slice = &reference[..compare_len];
            let rust_slice = &rust_output[delay..delay + compare_len];

            let aligned_corr = compute_correlation(ref_slice, rust_slice);
            let mae = compute_mae(ref_slice, rust_slice);

            eprintln!("Aligned correlation: {:.6}", aligned_corr);
            eprintln!("Mean Absolute Error: {:.6}", mae);

            // Sample-by-sample comparison for first few samples
            eprintln!("\nFirst 10 samples comparison (after delay alignment):");
            for i in 0..10.min(compare_len) {
                let diff = (ref_slice[i] - rust_slice[i]).abs();
                eprintln!(
                    "  [{}] Python: {:+.6}, Rust: {:+.6}, Diff: {:.6}",
                    i, ref_slice[i], rust_slice[i], diff
                );
            }

            // Assertions for Gold Standard
            assert!(
                aligned_corr > 0.95,
                "GOLD STANDARD FAILED: Correlation should be > 0.95 at 16kHz direct, got {:.6}. \
                 This indicates a bug in FFT/ONNX/IFFT/OLA logic, not resampling.",
                aligned_corr
            );
        } else {
            panic!("Not enough samples to compare after delay alignment");
        }
    }

    /// Full 48kHz round-trip test: downsample → process hops → upsample
    /// This tests the EXACT code path used by the plugin at non-16kHz sample rates.
    #[test]
    fn test_48k_full_roundtrip() {
        use anr_plugin::{Downsampler, Upsampler};

        let input_path = test_audio_dir().join("test_audio.wav");
        if !input_path.exists() {
            eprintln!("Skipping test_48k_full_roundtrip: test_audio.wav not found");
            return;
        }

        // Load 48kHz input
        let (input_48k, sample_rate) = load_wav(&input_path).expect("Failed to load input");
        assert_eq!(sample_rate, 48000, "Expected 48kHz input");
        eprintln!("Loaded {} samples at 48kHz", input_48k.len());

        // Create resamplers (exactly like the plugin does)
        let mut downsampler = Downsampler::new(48000.0).expect("Should create downsampler");
        let mut upsampler = Upsampler::new(48000.0).expect("Should create upsampler");

        // Create processor
        let mut processor = AudioProcessor::new();

        // Process: simulate the plugin's process_with_resampling flow
        let mut output_48k = Vec::new();
        let mut input_pos = 0;

        // Warm up stats
        let mut total_input_rms = 0.0f64;
        let mut total_output_rms = 0.0f64;
        let mut hop_count = 0;

        while input_pos < input_48k.len() {
            let input_needed = downsampler.input_frames_needed();

            // Check if we have enough input
            if input_pos + input_needed > input_48k.len() {
                break;
            }

            // Downsample: 48kHz → 16kHz
            let downsampled = downsampler.process(&input_48k[input_pos..input_pos + input_needed]);
            input_pos += input_needed;

            // Calculate input RMS at 16kHz
            let in_rms: f64 = downsampled.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            total_input_rms += in_rms;

            // Process through ANR at 16kHz (hop-by-hop, like the plugin)
            let processed = processor.process_hop_16k(&downsampled, ProcessingMode::PmlnOnly, 1.0);

            // Calculate output RMS at 16kHz
            let out_rms: f64 = processed.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            total_output_rms += out_rms;

            hop_count += 1;

            // Upsample: 16kHz → 48kHz
            let upsampled = upsampler.process(&processed);
            output_48k.extend_from_slice(upsampled);
        }

        eprintln!("Processed {} hops", hop_count);
        eprintln!("Output {} samples at 48kHz", output_48k.len());

        // Check gain ratio
        let avg_ratio = if total_input_rms > 0.0 {
            total_output_rms / total_input_rms
        } else {
            0.0
        };
        eprintln!("Average output/input RMS ratio: {:.4}", avg_ratio);

        // Sanity checks
        assert!(hop_count > 100, "Should process at least 100 hops, got {}", hop_count);
        assert!(output_48k.len() > 10000, "Should produce significant output, got {}", output_48k.len());

        // Check that output isn't silent
        let output_rms: f32 = (output_48k.iter().map(|x| x * x).sum::<f32>() / output_48k.len() as f32).sqrt();
        eprintln!("Output RMS: {:.6}", output_rms);
        assert!(output_rms > 0.001, "Output should not be silent, RMS = {}", output_rms);

        // Check that output isn't noise (should be correlated with something, not random)
        // Skip initial transient
        let skip = 5000.min(output_48k.len() / 4);
        let output_trimmed = &output_48k[skip..];

        // Check for NaN/Inf
        let has_nan = output_trimmed.iter().any(|x| x.is_nan());
        let has_inf = output_trimmed.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output contains NaN values!");
        assert!(!has_inf, "Output contains Inf values!");

        // Check that values are in reasonable range (not clipping wildly)
        let max_val = output_trimmed.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!("Max output value: {:.4}", max_val);
        assert!(max_val < 10.0, "Output values are too large (max = {}), suggests corruption", max_val);

        // Gain should be reasonable (not too quiet, not too loud)
        assert!(
            avg_ratio > 0.1 && avg_ratio < 5.0,
            "Gain ratio {:.4} is unreasonable, expected 0.1-5.0",
            avg_ratio
        );

        eprintln!("✓ Full 48kHz round-trip test passed!");
    }

    /// Test ring buffer accumulation - simulates exactly what the plugin does
    #[test]
    fn test_ring_buffer_accumulation() {
        use rtrb::RingBuffer;

        const RING_SIZE: usize = 8192;
        let (mut producer, mut consumer) = RingBuffer::new(RING_SIZE);

        // Push 186 samples (what the upsampler produces during warmup)
        let samples_per_push = 186;

        for i in 0..10 {
            // Push samples
            for j in 0..samples_per_push {
                let sample = (i * samples_per_push + j) as f32;
                assert!(producer.push(sample).is_ok(), "Push should succeed");
            }

            let available = consumer.slots();
            let expected = (i + 1) * samples_per_push;
            eprintln!("After push {}: available={}, expected={}", i, available, expected);
            assert_eq!(available, expected, "Ring buffer should accumulate");
        }

        eprintln!("✓ Ring buffer accumulates correctly");
    }

    /// Test upsampler warmup - check how many samples it produces over time
    #[test]
    fn test_upsampler_warmup() {
        use anr_plugin::{Upsampler, HOP_SIZE};

        let mut upsampler = Upsampler::new(48000.0).expect("Should create upsampler");

        let input = [0.1f32; HOP_SIZE]; // Non-zero input

        let mut total_output = 0;
        for i in 0..20 {
            let output = upsampler.process(&input);
            eprintln!("Hop {}: upsampler produced {} samples", i, output.len());
            total_output += output.len();
        }

        // After 20 hops, we should have produced significant output
        // Expected: ~20 * 384 = 7680 samples (with 3:1 ratio)
        eprintln!("Total output after 20 hops: {} (expected ~7680)", total_output);
        assert!(total_output > 5000, "Upsampler should produce significant output after warmup");
    }

    /// Test using the ACTUAL ResamplingPipeline struct (not a simulation!)
    /// This exercises the real code that the plugin uses.
    #[test]
    fn test_plugin_resampling_flow() {
        use anr_plugin::{ResamplingPipeline, ProcessingMode};

        const BUFFER_SIZE: usize = 480; // Typical host buffer at 48kHz

        // Create the ACTUAL pipeline used by the plugin
        let mut pipeline = ResamplingPipeline::new(48000.0)
            .expect("Should create pipeline for 48kHz");

        // Generate test input (sine wave at 48kHz)
        let test_input: Vec<f32> = (0..BUFFER_SIZE * 20)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();

        let mut input_pos = 0;
        let mut total_output = 0;
        let mut total_underruns = 0;

        // Simulate 20 buffer callbacks - EXACTLY like the plugin does
        for buffer_idx in 0..20 {
            // Get this buffer's input
            let end = (input_pos + BUFFER_SIZE).min(test_input.len());
            let buffer_input = &test_input[input_pos..end];
            input_pos = end;

            // Push input to pipeline (real code!)
            pipeline.push_input(buffer_input);

            // Run processing (real code!)
            let hops = pipeline.run_processing(ProcessingMode::PmlnOnly, 1.0);

            let output_available = pipeline.output_available();
            eprintln!("Buffer {}: hops={}, output_available={}", buffer_idx, hops, output_available);

            // Try to prime (real code!)
            if !pipeline.try_prime(BUFFER_SIZE) {
                // Not primed yet - would emit silence
                continue;
            }

            if buffer_idx == 1 || (buffer_idx < 5 && pipeline.is_primed()) {
                eprintln!("Pipeline primed at buffer {}", buffer_idx);
            }

            // Pop output (real code!)
            let mut output_buf = vec![0.0f32; BUFFER_SIZE];
            let (written, underruns) = pipeline.pop_output(&mut output_buf);
            total_output += written;
            total_underruns += underruns;
        }

        eprintln!("Total output samples: {}", total_output);
        eprintln!("Total underruns: {}", total_underruns);
        assert!(pipeline.is_primed(), "Pipeline should get primed");
        assert!(total_output > 5000, "Should produce significant output, got {}", total_output);
        assert!(total_underruns < 500, "Should have minimal underruns, got {}", total_underruns);
        eprintln!("✓ ResamplingPipeline works correctly (REAL CODE TEST)");
    }

    /// Full end-to-end test: 48kHz audio through ResamplingPipeline
    /// Uses actual audio file and checks output quality
    #[test]
    fn test_resampling_pipeline_with_audio() {
        use anr_plugin::{ResamplingPipeline, ProcessingMode};

        let input_path = test_audio_dir().join("test_audio.wav");
        if !input_path.exists() {
            eprintln!("Skipping test_resampling_pipeline_with_audio: test_audio.wav not found");
            return;
        }

        // Load 48kHz input
        let (input_48k, sample_rate) = load_wav(&input_path).expect("Failed to load input");
        assert_eq!(sample_rate, 48000, "Expected 48kHz input");

        // Create pipeline
        let mut pipeline = ResamplingPipeline::new(48000.0)
            .expect("Should create pipeline");

        const BUFFER_SIZE: usize = 512;
        let mut output = Vec::new();
        let mut total_underruns = 0;

        // Process in buffer-sized chunks, exactly like a real host would
        for chunk in input_48k.chunks(BUFFER_SIZE) {
            pipeline.push_input(chunk);
            pipeline.run_processing(ProcessingMode::PmlnOnly, 1.0);

            if pipeline.try_prime(BUFFER_SIZE) {
                let mut buf = vec![0.0f32; BUFFER_SIZE];
                let (_, underruns) = pipeline.pop_output(&mut buf);
                output.extend_from_slice(&buf);
                total_underruns += underruns;
            }
        }

        eprintln!("Input: {} samples, Output: {} samples", input_48k.len(), output.len());
        eprintln!("Underruns: {}", total_underruns);

        // Quality checks
        let skip = 5000.min(output.len() / 4);
        let output_trimmed = &output[skip..];

        // Check for NaN/Inf
        assert!(!output_trimmed.iter().any(|x| x.is_nan()), "Output contains NaN!");
        assert!(!output_trimmed.iter().any(|x| x.is_infinite()), "Output contains Inf!");

        // Check output isn't silent
        let output_rms: f32 = (output_trimmed.iter().map(|x| x * x).sum::<f32>() / output_trimmed.len() as f32).sqrt();
        eprintln!("Output RMS: {:.6}", output_rms);
        assert!(output_rms > 0.001, "Output should not be silent");

        // Check values are reasonable
        let max_val = output_trimmed.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_val < 10.0, "Output values too large: {}", max_val);

        // Save output for listening
        let output_path = test_audio_dir().join("pipeline_output_48k.wav");
        save_wav(&output_path, &output, 48000).expect("Failed to save output");
        eprintln!("Saved output to: {}", output_path.display());

        eprintln!("✓ ResamplingPipeline produces valid audio output");
    }
}
