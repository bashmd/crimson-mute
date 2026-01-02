//! Test utilities for signal generation and analysis.

use std::f32::consts::PI;

/// Generate a sine wave at the given frequency and sample rate.
pub fn generate_sine(frequency: f32, sample_rate: f32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

/// Generate a sine wave with specified amplitude.
pub fn generate_sine_with_amplitude(
    frequency: f32,
    amplitude: f32,
    sample_rate: f32,
    num_samples: usize,
) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            amplitude * (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

/// Generate an impulse (single sample at 1.0, rest at 0.0).
pub fn generate_impulse(num_samples: usize, impulse_position: usize) -> Vec<f32> {
    let mut signal = vec![0.0; num_samples];
    if impulse_position < num_samples {
        signal[impulse_position] = 1.0;
    }
    signal
}

/// Generate white noise with amplitude in [-amplitude, amplitude].
pub fn generate_noise(num_samples: usize, amplitude: f32, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducible "random" noise
    let mut state = seed;
    (0..num_samples)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let normalized = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
            normalized * amplitude
        })
        .collect()
}

/// Generate a DC signal (constant value).
pub fn generate_dc(value: f32, num_samples: usize) -> Vec<f32> {
    vec![value; num_samples]
}

/// Compute RMS (root mean square) of a signal.
pub fn compute_rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = signal.iter().map(|x| x * x).sum();
    (sum_sq / signal.len() as f32).sqrt()
}

/// Compute peak amplitude of a signal.
pub fn compute_peak(signal: &[f32]) -> f32 {
    signal
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, |a, b| a.max(b))
}

/// Compute SNR (signal-to-noise ratio) in dB between original and processed signals.
/// Assumes the difference is noise.
pub fn compute_snr_db(original: &[f32], processed: &[f32]) -> f32 {
    assert_eq!(original.len(), processed.len());

    let signal_power: f32 = original.iter().map(|x| x * x).sum();
    let noise_power: f32 = original
        .iter()
        .zip(processed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    if noise_power < 1e-20 {
        return 120.0; // Essentially perfect
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Find the delay (in samples) between two signals using cross-correlation.
/// Returns the delay that maximizes correlation.
pub fn find_delay(reference: &[f32], delayed: &[f32], max_delay: usize) -> usize {
    let mut best_delay = 0;
    let mut best_correlation = f32::NEG_INFINITY;

    for delay in 0..=max_delay.min(delayed.len()) {
        let mut correlation = 0.0f32;
        let mut count = 0;

        for i in 0..reference.len() {
            if i + delay < delayed.len() {
                correlation += reference[i] * delayed[i + delay];
                count += 1;
            }
        }

        if count > 0 {
            correlation /= count as f32;
            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        }
    }

    best_delay
}

/// Compare two signals with a given tolerance (max absolute difference).
pub fn signals_equal(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() <= tolerance)
}

/// Compare two signals after accounting for a delay.
/// Returns true if they match within tolerance after the delay.
pub fn signals_equal_with_delay(
    reference: &[f32],
    delayed: &[f32],
    delay: usize,
    tolerance: f32,
) -> bool {
    if delay >= delayed.len() {
        return false;
    }

    let compare_len = reference.len().min(delayed.len() - delay);
    reference[..compare_len]
        .iter()
        .zip(delayed[delay..delay + compare_len].iter())
        .all(|(a, b)| (a - b).abs() <= tolerance)
}

/// Compute the dominant frequency in a signal using zero-crossing rate.
/// Returns approximate frequency in Hz.
pub fn estimate_frequency(signal: &[f32], sample_rate: f32) -> f32 {
    if signal.len() < 3 {
        return 0.0;
    }

    let mut zero_crossings = 0;
    for i in 1..signal.len() {
        if (signal[i - 1] >= 0.0) != (signal[i] >= 0.0) {
            zero_crossings += 1;
        }
    }

    // Each full cycle has 2 zero crossings
    let cycles = zero_crossings as f32 / 2.0;
    let duration = signal.len() as f32 / sample_rate;

    cycles / duration
}

/// Check if a signal is approximately silent (all values near zero).
pub fn is_silent(signal: &[f32], threshold: f32) -> bool {
    signal.iter().all(|x| x.abs() < threshold)
}

/// Count non-zero samples in a signal.
pub fn count_nonzero(signal: &[f32], threshold: f32) -> usize {
    signal.iter().filter(|x| x.abs() >= threshold).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sine() {
        let signal = generate_sine(1000.0, 48000.0, 480);
        assert_eq!(signal.len(), 480);
        // 1kHz at 48kHz = 48 samples per cycle, 480 samples = 10 cycles
        // Check peak is approximately 1.0
        assert!((compute_peak(&signal) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_generate_impulse() {
        let signal = generate_impulse(100, 50);
        assert_eq!(signal.len(), 100);
        assert_eq!(signal[50], 1.0);
        assert_eq!(signal[49], 0.0);
        assert_eq!(signal[51], 0.0);
    }

    #[test]
    fn test_compute_rms() {
        // DC signal of 1.0 has RMS of 1.0
        let dc = generate_dc(1.0, 100);
        assert!((compute_rms(&dc) - 1.0).abs() < 0.001);

        // Sine wave has RMS of 1/sqrt(2) ≈ 0.707
        let sine = generate_sine(100.0, 1000.0, 1000);
        assert!((compute_rms(&sine) - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_compute_snr() {
        let original = generate_sine(100.0, 1000.0, 1000);

        // Identical signals = very high SNR
        let snr = compute_snr_db(&original, &original);
        assert!(snr > 100.0);

        // Add noise
        let noisy: Vec<f32> = original.iter().map(|x| x + 0.01).collect();
        let snr_noisy = compute_snr_db(&original, &noisy);
        assert!(snr_noisy > 0.0 && snr_noisy < 100.0);
    }

    #[test]
    fn test_find_delay() {
        // Use an impulse for unambiguous delay detection
        let original = generate_impulse(100, 10); // impulse at position 10

        // Create delayed version
        let delay = 50;
        let mut delayed = vec![0.0; delay];
        delayed.extend_from_slice(&original);

        let found_delay = find_delay(&original, &delayed, 100);
        assert_eq!(found_delay, delay);
    }

    #[test]
    fn test_estimate_frequency() {
        let signal = generate_sine(1000.0, 48000.0, 4800); // 100 cycles
        let freq = estimate_frequency(&signal, 48000.0);
        // Should be close to 1000 Hz
        assert!((freq - 1000.0).abs() < 50.0); // Within 5% tolerance
    }
}
