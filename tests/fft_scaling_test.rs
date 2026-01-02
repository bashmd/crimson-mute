//! Test FFT scaling behavior
use realfft::RealFftPlanner;

#[test]
fn test_realfft_roundtrip_scaling() {
    let n = 512;
    
    // Create test signal
    let mut input: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 10.0 * i as f32 / n as f32).sin())
        .collect();
    
    let input_rms: f32 = (input.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
    eprintln!("Input RMS: {:.6}", input_rms);
    
    let original_input = input.clone();
    
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    
    // Forward FFT
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).unwrap();
    
    eprintln!("Spectrum DC bin: {:?}", spectrum[0]);
    
    // Inverse FFT (WITHOUT manual scaling)
    let mut output = vec![0.0f32; n];
    ifft.process(&mut spectrum, &mut output).unwrap();
    
    let output_rms_no_scale: f32 = (output.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
    eprintln!("Output RMS (no scaling): {:.6}", output_rms_no_scale);
    eprintln!("Ratio (out/in, no scaling): {:.6}", output_rms_no_scale / input_rms);
    
    // Now with 1/N scaling
    for x in output.iter_mut() {
        *x /= n as f32;
    }
    let output_rms_scaled: f32 = (output.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
    eprintln!("Output RMS (with 1/N scaling): {:.6}", output_rms_scaled);
    eprintln!("Ratio (out/in, with scaling): {:.6}", output_rms_scaled / input_rms);
    
    // Check max diff
    let max_diff: f32 = output.iter()
        .zip(original_input.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Max diff after roundtrip: {:.10}", max_diff);
}
