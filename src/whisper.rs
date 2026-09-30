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
        params.set_token_timestamps(true);
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

        let _num_segments = state.full_n_segments();

        let mut full_text = String::new();
        let mut aligned_words = Vec::new();

        let num_segments = state.full_n_segments();
        for s in 0..num_segments {
            if let Some(seg) = state.get_segment(s) {
                if let Ok(seg_text) = seg.to_str_lossy() {
                    let trimmed = seg_text.trim();
                    if !trimmed.is_empty() {
                        if !full_text.is_empty() {
                            full_text.push(' ');
                        }
                        full_text.push_str(trimmed);
                    }
                }

                let num_tokens = seg.n_tokens();
                for t in 0..num_tokens {
                    if let Some(token) = seg.get_token(t) {
                        let token_text = token.to_str_lossy().unwrap_or_default();
                        let clean = token_text.trim();
                        if clean.is_empty() || clean.starts_with("[_") {
                            continue;
                        }

                        let data = token.token_data();
                        let start_ms = data.t0 * 10;
                        let end_ms = data.t1 * 10;

                        aligned_words.push(AlignedWord {
                            word: clean.to_string(),
                            start: start_ms as f64 / 1000.0,
                            end: end_ms as f64 / 1000.0,
                            start_ms,
                            end_ms: end_ms.max(start_ms + 60),
                            speaker: 0,
                            score: Some(token.token_probability()),
                        });
                    }
                }
            }
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
