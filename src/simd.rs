//! SIMD-optimized DSP routines using multiversion for runtime dispatch.
//!
//! All functions are multiversioned across x86_64 SIMD feature sets:
//! - AVX2+FMA (Haswell+, Ryzen 1+)
//! - AVX (Sandy/Ivy Bridge)
//! - SSE4.1 (Core 2/Nehalem)

use multiversion::multiversion;
use rustfft::num_complex::Complex32;

/// Extract magnitude and phase from complex spectrum.
/// magnitude[i] = sqrt(re² + im²)
/// phase[i] = atan2(im, re)
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn extract_mag_phase(spectrum: &[Complex32], magnitude: &mut [f32], phase: &mut [f32]) {
    debug_assert_eq!(spectrum.len(), magnitude.len());
    debug_assert_eq!(spectrum.len(), phase.len());

    for (i, bin) in spectrum.iter().enumerate() {
        magnitude[i] = bin.norm();
        phase[i] = bin.im.atan2(bin.re);
    }
}

/// Rebuild complex spectrum from magnitude and phase (polar form).
/// output[i] = mag * (cos(phase) + j*sin(phase))
///
/// NOTE: For use with realfft, DC (bin 0) and Nyquist (last bin) must have
/// zero imaginary parts. This function forces them to be purely real.
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn polar_to_complex(magnitude: &[f32], phase: &[f32], output: &mut [Complex32]) {
    debug_assert_eq!(magnitude.len(), phase.len());
    debug_assert_eq!(magnitude.len(), output.len());

    let n = magnitude.len();
    for i in 0..n {
        let (sin_p, cos_p) = phase[i].sin_cos();
        output[i] = Complex32::new(magnitude[i] * cos_p, magnitude[i] * sin_p);
    }

    // Force DC and Nyquist bins to be purely real (required by realfft)
    if n > 0 {
        output[0] = Complex32::new(output[0].re, 0.0);
    }
    if n > 1 {
        output[n - 1] = Complex32::new(output[n - 1].re, 0.0);
    }
}

/// Overlap-add accumulation: dst[i] += src[i]
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn ola_add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());

    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

/// Multiply two f32 slices element-wise: out[i] = a[i] * b[i]
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// Apply mask to packed magnitude/phase array (514 elements).
/// out_mag[i] = packed[i] * mask[i]           (first 257)
/// out_phase[i] = packed[257+i] * mask[257+i] (last 257)
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn apply_pmln_mask(
    packed: &[f32; 514],
    mask: &[f32],
    out_mag: &mut [f32; 257],
    out_phase: &mut [f32; 257],
) {
    debug_assert!(mask.len() >= 514);

    for i in 0..257 {
        out_mag[i] = packed[i] * mask[i];
        out_phase[i] = packed[257 + i] * mask[257 + i];
    }
}

/// Apply power-based strength adjustment to mask values.
/// mask[i] = clamp(mask[i]^strength, 0.0, 1.0)
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn apply_mask_strength(mask: &mut [f32], strength: f32) {
    for v in mask.iter_mut() {
        *v = v.powf(strength).clamp(0.0, 1.0);
    }
}

/// Multiply complex spectrum by real mask: out[i] = spectrum[i] * mask[i]
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn complex_mul_scalar(spectrum: &[Complex32], mask: &[f32], out: &mut [Complex32]) {
    debug_assert_eq!(spectrum.len(), mask.len());
    debug_assert_eq!(spectrum.len(), out.len());

    for i in 0..spectrum.len() {
        out[i] = spectrum[i] * mask[i];
    }
}

/// Scale a buffer in-place: buf[i] *= scale
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn scale_buffer(buf: &mut [f32], scale: f32) {
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

/// Apply window function: dst[i] = src[i] * window[i]
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+avx",
    "x86_64+sse4.1"
))]
pub fn apply_window(src: &[f32], window: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), window.len());
    debug_assert_eq!(src.len(), dst.len());

    for i in 0..src.len() {
        dst[i] = src[i] * window[i];
    }
}
