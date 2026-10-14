mod align;
mod audio;
mod diarize;
mod export;
mod model;
mod nemotron;
mod whisper;

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "tiginal-diarize")]
#[command(author = "pengzhang")]
#[command(version = "0.2.0")]
#[command(
    about = "Fast, local speaker diarization & alignment CLI based on NVIDIA Nemotron-3, MMS-Align, and Whisper",
    long_about = "Identifies 'who spoke when' in audio files using a local quantized ONNX model.\n\
                  Supports high-precision forced alignment with MMS-Align, and local offline transcription with Whisper.\n\
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

    /// Automatically download models from Hugging Face if not found in local cache
    #[arg(long = "download")]
    download: bool,

    /// Align an existing transcript with the audio using MMS-Align (Wav2Vec2-CTC)
    #[arg(short = 't', long = "transcript", value_name = "FILE")]
    transcript: Option<PathBuf>,

    /// Custom path to MMS-Align ONNX model
    #[arg(long = "align-model", value_name = "PATH")]
    align_model: Option<PathBuf>,

    /// Transcribe audio locally using Whisper and align with speakers
    #[arg(long = "transcribe")]
    transcribe: bool,

    /// Custom path to Whisper model (.bin / .gguf)
    #[arg(long = "whisper-model", value_name = "PATH")]
    whisper_model: Option<PathBuf>,

    /// Language for transcription / alignment (e.g. 'en', 'zh', default: 'auto')
    #[arg(short = 'l', long = "language", value_name = "LANG")]
    language: Option<String>,

    /// Automatically save derived artifacts (<stem>-diar.txt, <stem>.srt) beside audio file
    #[arg(long = "write-artifacts")]
    write_artifacts: bool,

    /// Skip speaker diarization (Nemotron-3)
    #[arg(long = "no-diarize")]
    no_diarize: bool,

    /// Skip generating SRT subtitles
    #[arg(long = "no-srt")]
    no_srt: bool,

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
        eprintln!("[info] Starting tiginal-diarize v0.2.0");
        eprintln!("[info] Target audio: {}", args.audio.display());
    }

    // 1. Resolve Nemotron-3 model (only if diarization is enabled)
    let model_path = if !args.no_diarize {
        match model::resolve_model(args.model_path.as_deref(), args.download, args.verbose) {
            Ok(path) => Some(path),
            Err(err) => {
                eprintln!("[error] {err}");
                return ExitCode::from(1);
            }
        }
    } else {
        None
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

    // 3. Run speaker diarization pipeline (Nemotron-3)
    let turns = if !args.no_diarize {
        let m_path = model_path.as_ref().unwrap();
        match diarize::turns(&samples, args.speakers, m_path, args.verbose) {
            Ok(t) => t,
            Err(err) => {
                eprintln!("[error] Diarization inference failed: {err}");
                return ExitCode::from(3);
            }
        }
    } else {
        Vec::new()
    };

    // 4. Handle Alignment or Local Whisper Transcription
    let mut transcript_text: Option<String> = None;
    let mut aligned_words: Option<Vec<export::AlignedWord>> = None;

    if args.transcribe {
        let w_model_path = args.whisper_model.unwrap_or_else(whisper::default_whisper_model_path);
        let transcriber = whisper::WhisperTranscriber::new(&w_model_path, args.language.clone());
        match transcriber.transcribe(&samples, args.verbose) {
            Ok((txt, words)) => {
                transcript_text = Some(txt);
                aligned_words = Some(words);
            }
            Err(err) => {
                eprintln!("[error] Whisper transcription failed: {err}");
                return ExitCode::from(4);
            }
        }
    } else if let Some(ref trans_path) = args.transcript {
        match std::fs::read_to_string(trans_path) {
            Ok(content) => {
                let clean_text = content.trim().to_string();
                let mms_path = args.align_model.unwrap_or_else(align::default_mms_model_path);

                if mms_path.exists() {
                    match align::MmsAligner::load(&mms_path, None, args.verbose) {
                        Ok(mut aligner) => match aligner.align(&samples, &clean_text, args.verbose) {
                            Ok(words) => {
                                aligned_words = Some(words);
                            }
                            Err(e) => {
                                if args.verbose {
                                    eprintln!("[warn] MMS-Align alignment failed: {e}; using fallback");
                                }
                                aligned_words = Some(align::fallback_proportional_alignment(&clean_text, audio_duration_s));
                            }
                        },
                        Err(e) => {
                            if args.verbose {
                                eprintln!("[warn] Failed to load MMS-Align model: {e}; using fallback");
                            }
                            aligned_words = Some(align::fallback_proportional_alignment(&clean_text, audio_duration_s));
                        }
                    }
                } else {
                    if args.verbose {
                        eprintln!("[info] MMS-Align model not found at {}; using proportional fallback", mms_path.display());
                    }
                    aligned_words = Some(align::fallback_proportional_alignment(&clean_text, audio_duration_s));
                }
                transcript_text = Some(clean_text);
            }
            Err(err) => {
                eprintln!("[error] Failed to read transcript file {}: {err}", trans_path.display());
                return ExitCode::from(4);
            }
        }
    }

    // 5. Build output and write optional artifacts
    let json_output = if let Some(words) = aligned_words {
        let segments = export::match_words_to_turns(&words, &turns);
        let diar_text = if !args.no_diarize {
            Some(export::build_diarized_text(&segments))
        } else {
            None
        };
        let srt = if !args.no_srt {
            Some(export::build_srt_content(&segments, !args.no_diarize))
        } else {
            None
        };

        if args.write_artifacts {
            let stem = args.audio.with_extension("");
            if !args.no_diarize {
                if let Some(ref dt) = diar_text {
                    let diar_path = PathBuf::from(format!("{}-diar.txt", stem.display()));
                    if let Err(e) = std::fs::write(&diar_path, dt) {
                        eprintln!("[error] Failed to write {}: {e}", diar_path.display());
                    } else if args.verbose {
                        eprintln!("[info] Saved: {}", diar_path.display());
                    }
                }
            }

            if !args.no_srt {
                if let Some(ref st) = srt {
                    let srt_path = PathBuf::from(format!("{}.srt", stem.display()));
                    if let Err(e) = std::fs::write(&srt_path, st) {
                        eprintln!("[error] Failed to write {}: {e}", srt_path.display());
                    } else if args.verbose {
                        eprintln!("[info] Saved: {}", srt_path.display());
                    }
                }
            }

            if args.transcribe {
                if let Some(ref txt) = transcript_text {
                    let txt_path = PathBuf::from(format!("{}.txt", stem.display()));
                    if !txt_path.exists() {
                        let _ = std::fs::write(&txt_path, txt);
                    }
                }
            }
        }

        let full_output = export::FullDiarizationOutput {
            turns,
            transcript: transcript_text,
            segments: Some(segments),
            diar_text,
            srt,
        };

        serde_json::to_string_pretty(&full_output)
            .map_err(|e| format!("Failed to serialize full JSON: {e}"))
    } else {
        // Pure diarization mode: output turns directly for backward compatibility
        serde_json::to_string_pretty(&turns)
            .map_err(|e| format!("Failed to serialize JSON: {e}"))
    };

    let output_str = match json_output {
        Ok(s) => s,
        Err(err) => {
            eprintln!("[error] {err}");
            return ExitCode::from(5);
        }
    };

    if let Some(out_path) = args.output {
        if let Err(err) = write_output_file(&out_path, &output_str) {
            eprintln!("[error] Failed writing output file {}: {err}", out_path.display());
            return ExitCode::from(6);
        }
        if args.verbose {
            eprintln!("[info] Results saved to: {}", out_path.display());
        }
    } else {
        println!("{output_str}");
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

fn write_output_file(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    file.write_all(content.as_bytes())?;
    file.flush()
}
