use std::path::{Path, PathBuf};
use crate::export::AlignedWord;

#[allow(dead_code)]
pub struct WhisperTranscriber {
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    language: Option<String>,
}

impl WhisperTranscriber {
    pub fn new(model_path: &Path, language: Option<String>) -> Self {
        Self {
            model_path: model_path.to_path_buf(),
            language,
        }
    }

    /// Transcribes audio samples and extracts words with token/word-level timestamps.
    pub fn transcribe(&self, samples: &[f32], verbose: bool) -> Result<(String, Vec<AlignedWord>), String> {
        #[cfg(feature = "whisper")]
        {
            self.transcribe_internal(samples, verbose)
        }
        #[cfg(not(feature = "whisper"))]
        {
            self.transcribe_via_cli(samples, verbose)
        }
    }

    #[cfg(feature = "whisper")]
    fn transcribe_internal(&self, samples: &[f32], verbose: bool) -> Result<(String, Vec<AlignedWord>), String> {
        use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

        if verbose {
            eprintln!("[info] Loading Whisper model from: {}", self.model_path.display());
        }

        let ctx_params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(&self.model_path, ctx_params)
            .map_err(|e| format!("Failed to load Whisper model: {e}"))?;

        let mut state = ctx.create_state()
            .map_err(|e| format!("Failed to create Whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        params.set_n_threads(threads.min(16) as i32);
        params.set_token_timestamps(true);
        params.set_split_on_word(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_no_speech_thold(0.6);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_progress_callback_safe(|progress| {
            eprintln!("[progress] whisper: {progress}");
        });

        if let Some(ref lang) = self.language {
            if lang != "auto" {
                params.set_language(Some(lang));
            }
        }

        if verbose {
            eprintln!("[info] Running Whisper transcription on {} samples...", samples.len());
        }

        state.full(params, samples)
            .map_err(|e| format!("Whisper transcription failed: {e}"))?;

        let eot = ctx.token_eot();
        let mut full_text = String::new();
        let mut aligned_words = Vec::new();

        let num_segments = state.full_n_segments();
        for s in 0..num_segments {
            let Some(seg) = state.get_segment(s) else {
                continue;
            };

            let no_speech = seg.no_speech_probability();
            if no_speech > 0.85 {
                continue;
            }

            if let Ok(seg_text) = seg.to_str_lossy() {
                let trimmed = seg_text.trim();
                if !trimmed.is_empty() {
                    if !full_text.is_empty() {
                        full_text.push(' ');
                    }
                    full_text.push_str(trimmed);
                }
            }

            let mut current: Option<(Vec<u8>, i64, i64, f32)> = None;
            let flush = |current: &mut Option<(Vec<u8>, i64, i64, f32)>, words: &mut Vec<AlignedWord>| {
                if let Some((bytes, start_ms, end_ms, score)) = current.take() {
                    let text = String::from_utf8_lossy(&bytes).trim().to_owned();
                    if !text.is_empty() && !text.starts_with("[_") {
                        words.push(AlignedWord {
                            word: text,
                            start: start_ms as f64 / 1000.0,
                            end: end_ms as f64 / 1000.0,
                            start_ms,
                            end_ms: end_ms.max(start_ms + 60),
                            speaker: 0,
                            score: Some(score),
                        });
                    }
                }
            };

            let num_tokens = seg.n_tokens();
            for t in 0..num_tokens {
                let Some(token) = seg.get_token(t) else {
                    continue;
                };
                if token.token_id() >= eot {
                    continue;
                }
                let Ok(bytes) = token.to_bytes() else {
                    continue;
                };
                let data = token.token_data();
                let (t0, t1) = if data.t_dtw >= 0 {
                    (data.t_dtw * 10, data.t_dtw * 10)
                } else {
                    (data.t0 * 10, data.t1 * 10)
                };

                let is_cjk_start = std::str::from_utf8(bytes)
                    .ok()
                    .and_then(|s| s.chars().next())
                    .map_or(false, crate::export::is_cjk);

                let prev_incomplete = current
                    .as_ref()
                    .is_some_and(|(wb, _, _, _)| std::str::from_utf8(wb).is_err());

                if prev_incomplete {
                    if let Some((word_bytes, _, end, _)) = current.as_mut() {
                        word_bytes.extend_from_slice(bytes);
                        *end = t1.max(*end);
                    }
                } else if !bytes.starts_with(b" ") && !is_cjk_start {
                    match current.as_mut() {
                        Some((word_bytes, _, end, _)) => {
                            word_bytes.extend_from_slice(bytes);
                            *end = t1.max(*end);
                        }
                        None => {
                            current = Some((bytes.to_vec(), t0, t1, token.token_probability()));
                        }
                    }
                } else {
                    flush(&mut current, &mut aligned_words);
                    current = Some((bytes.to_vec(), t0, t1, token.token_probability()));
                }
            }
            flush(&mut current, &mut aligned_words);
        }

        Ok((full_text, aligned_words))
    }

    #[cfg(not(feature = "whisper"))]
    fn transcribe_via_cli(&self, _samples: &[f32], _verbose: bool) -> Result<(String, Vec<AlignedWord>), String> {
        Err("Local Whisper transcription requires compiling with `--features whisper` or `--features metal`.".to_string())
    }
}

/// Resolves default path for Whisper model.
pub fn default_whisper_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tiginal")
        .join("models")
        .join("whisper")
        .join("ggml-large-v3-turbo.bin")
}
