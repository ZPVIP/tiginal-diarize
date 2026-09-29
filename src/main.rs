mod audio;
mod diarize;
mod model;
mod nemotron;

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "tiginal-diarize")]
#[command(author = "pengzhang")]
#[command(version = "0.1.0")]
#[command(
    about = "Fast, local speaker diarization CLI based on NVIDIA Nemotron-3",
    long_about = "Identifies 'who spoke when' in audio files using a local quantized ONNX model.\n\
                  Outputs clean JSON to stdout, making it easy to pipe to jq, Tiginal, or shell scripts."
)]
struct Cli {
    /// Path to input audio file (WAV preferred, or MP3/M4A/FLAC if ffmpeg is installed)
    #[arg(value_name = "AUDIO_PATH")]
    audio: PathBuf,

    /// Fix number of speakers (optional, e.g. -s 2). By default, clustering detects it automatically.
    #[arg(short = 's', long = "speakers", value_name = "N")]
    speakers: Option<usize>,

    /// Custom path to Nemotron-3 ONNX model (model_quantized.onnx).
    /// Defaults to ~/.cache/tiginal/models/nemotron-3-diarization/model_quantized.onnx
    #[arg(short = 'm', long = "model-path", value_name = "PATH")]
    model_path: Option<PathBuf>,

    /// Automatically download the ~120MB model from Hugging Face if not found in local cache
    #[arg(long = "download")]
    download: bool,

    /// Write output JSON to a file instead of stdout
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    output: Option<PathBuf>,

    /// Print detailed execution and progress logs to stderr
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn main() -> ExitCode {
    let args = Cli::parse();
    let started = Instant::now();

    if args.verbose {
        eprintln!("[info] Starting tiginal-diarize v0.1.0");
        eprintln!("[info] Target audio: {}", args.audio.display());
    }

    // 1. Resolve model
    let model_path = match model::resolve_model(args.model_path.as_deref(), args.download, args.verbose) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("[error] {err}");
            return ExitCode::from(1);
        }
    };

    // 2. Load and decode audio to 16kHz mono f32
    let samples = match audio::load_audio(&args.audio, args.verbose) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("[error] Failed to load audio: {err}");
            return ExitCode::from(2);
        }
    };

    if samples.is_empty() {
        eprintln!("[error] Audio file contains no audio samples.");
        return ExitCode::from(2);
    }

    let audio_duration_s = samples.len() as f64 / audio::SAMPLE_RATE as f64;
    if args.verbose {
        eprintln!("[info] Audio duration: {:.2}s ({:.2} minutes)", audio_duration_s, audio_duration_s / 60.0);
    }

    // 3. Run diarization pipeline
    let turns = match diarize::turns(&samples, args.speakers, &model_path, args.verbose) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("[error] Diarization inference failed: {err}");
            return ExitCode::from(3);
        }
    };

    // 4. Output results as JSON
    let json_output = match serde_json::to_string_pretty(&turns) {
        Ok(j) => j,
        Err(err) => {
            eprintln!("[error] Failed to serialize JSON: {err}");
            return ExitCode::from(4);
        }
    };

    if let Some(out_path) = args.output {
        if let Err(err) = write_output_file(&out_path, &json_output) {
            eprintln!("[error] Failed writing output file {}: {err}", out_path.display());
            return ExitCode::from(5);
        }
        if args.verbose {
            eprintln!("[info] Results saved to: {}", out_path.display());
        }
    } else {
        println!("{json_output}");
    }

    if args.verbose {
        let elapsed = started.elapsed().as_secs_f64();
        let rtf = elapsed / audio_duration_s.max(0.001);
        eprintln!(
            "[done] Finished in {:.2}s (RTF: {:.3}x realtime)",
            elapsed, rtf
        );
    }

    ExitCode::SUCCESS
}

fn write_output_file(path: &PathBuf, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    file.write_all(content.as_bytes())?;
    file.flush()
}
