use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

pub const SAMPLE_RATE: u32 = 16_000;

/// Loads an audio file as 16 kHz mono f32 samples in [-1.0, 1.0].
/// First attempts native decoding via `hound` for WAV files.
/// Falls back to `ffmpeg` for other formats (MP3, M4A, FLAC, etc.) or complex WAVs.
pub fn load_audio(path: &Path, verbose: bool) -> Result<Vec<f32>, String> {
    if !path.exists() {
        return Err(format!("File not found: {}", path.display()));
    }

    // Try hound first for WAV files
    match load_wav_hound(path) {
        Ok(samples) => {
            if verbose {
                eprintln!("[info] Successfully decoded WAV with native hound parser: {} samples ({:.2}s)", 
                    samples.len(), samples.len() as f64 / SAMPLE_RATE as f64);
            }
            return Ok(samples);
        }
        Err(err) => {
            if verbose {
                eprintln!("[debug] Native WAV decoding skipped/failed: {err}. Attempting ffmpeg fallback...");
            }
        }
    }

    // Fallback to ffmpeg
    decode_with_ffmpeg(path, verbose)
}

fn load_wav_hound(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("hound open error: {e}"))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err("WAV has 0 channels".into());
    }

    let samples_f32: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = match spec.bits_per_sample {
                8 => 128.0,
                16 => 32768.0,
                24 => 8388608.0,
                32 => 2147483648.0,
                other => return Err(format!("Unsupported integer bit depth: {other}")),
            };
            reader
                .samples::<i32>()
                .map(|s| (s.unwrap_or(0) as f32) / max_val)
                .collect()
        }
    };

    // Mixdown to mono if multiple channels
    let mono: Vec<f32> = if channels == 1 {
        samples_f32
    } else {
        samples_f32
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Resample to 16kHz if needed
    if spec.sample_rate == SAMPLE_RATE {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, spec.sample_rate, SAMPLE_RATE))
    }
}

fn decode_with_ffmpeg(path: &Path, verbose: bool) -> Result<Vec<f32>, String> {
    if verbose {
        eprintln!("[info] Spawning ffmpeg to decode audio to 16kHz mono...");
    }

    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(path)
        .args([
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not execute ffmpeg: {e}. Please ensure ffmpeg is installed if loading non-standard WAV files."))?;

    let mut bytes = Vec::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout
            .read_to_end(&mut bytes)
            .map_err(|e| format!("Failed reading ffmpeg stdout: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed waiting for ffmpeg: {e}"))?;

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg failed to decode {}: {}", path.display(), err_msg.trim()));
    }

    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(samples)
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let dst_len = ((input.len() as f64) / ratio).round() as usize;
    let mut output = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;

        if idx + 1 < input.len() {
            output.push(input[idx] * (1.0 - frac) + input[idx + 1] * frac);
        } else if idx < input.len() {
            output.push(input[idx]);
        } else {
            output.push(0.0);
        }
    }
    output
}
