use std::path::{Path, PathBuf};
use crate::export::AlignedWord;
#[cfg(feature = "whisper")]
use std::{ffi::c_void, sync::Arc};

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

        if verbose {
            eprintln!("[info] Running Whisper transcription on {} samples...", samples.len());
        }

        let eot = ctx.token_eot();
        let mut full_text = String::new();
        let mut aligned_words = Vec::new();
        let mut context: Vec<i32> = Vec::new();
        let mut punctuated_len = 0;
        let mut styled_seen = false;
        let mut retrying_loop = false;
        let mut start = 0;

        loop {
            let guard = Arc::new(WindowGuard::new(styled_seen, retrying_loop));
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
            params.set_tokens(&context);
            let (done, total) = (start, samples.len());
            params.set_progress_callback_safe(move |progress: i32| {
                let overall = (done + (total - done) * progress.max(0) as usize / 100) * 100 / total;
                eprintln!("[progress] whisper: {overall}");
            });
            let watched = Arc::clone(&guard);
            params.set_segment_callback_safe_lossy(move |segment: whisper_rs::SegmentCallbackData| {
                watched.state.lock().unwrap().window.push(segment.text);
            });
            // SAFETY: `guard` outlives `state.full`, the only caller of the callback. whisper-rs's `set_abort_callback_safe` casts its boxed closure back to the wrong type, so the raw setter is used.
            unsafe {
                params.set_abort_callback(Some(stop_at_bad_window));
                params.set_abort_callback_user_data(Arc::as_ptr(&guard) as *mut c_void);
            }

            if let Some(ref lang) = self.language {
                if lang != "auto" {
                    params.set_language(Some(lang));
                }
            }

            let result = state.full(params, &samples[start..]);
            let (stop, seen) = {
                let guarded = guard.state.lock().unwrap();
                (guarded.stop, guarded.styled_seen)
            };
            styled_seen = seen;
            if stop.is_none() {
                result.map_err(|e| format!("Whisper transcription failed: {e}"))?;
            }

            let offset_ms = (start / SAMPLES_PER_MS) as i64;
            let looped_segments = match stop {
                Some(Stop::Looped { segments }) => segments as i32,
                _ => 0,
            };
            let mut end_ms = offset_ms;
            for s in 0..state.full_n_segments() - looped_segments {
                let Some(seg) = state.get_segment(s) else {
                    continue;
                };
                end_ms = seg.end_timestamp() * 10 + offset_ms;
                collect_segment(&seg, eot, offset_ms, &mut full_text, &mut aligned_words, &mut context, &mut punctuated_len);
            }

            let next = end_ms as usize * SAMPLES_PER_MS;
            match stop {
                None => break,
                Some(Stop::LostStyle) => {
                    if next <= start {
                        return Err(format!("Whisper made no progress past {offset_ms} ms"));
                    }
                    if verbose {
                        eprintln!("[info] Whisper dropped punctuation in the window ending at {end_ms} ms; resuming with the last punctuated text as context");
                    }
                    context.truncate(punctuated_len);
                    context.drain(..context.len().saturating_sub(CONTEXT_TOKENS));
                    punctuated_len = context.len();
                    retrying_loop = false;
                }
                Some(Stop::Looped { .. }) => {
                    if verbose {
                        eprintln!("[info] Whisper repeated itself in the window after {end_ms} ms; decoding it again without context");
                    }
                    context.clear();
                    punctuated_len = 0;
                    retrying_loop = true;
                }
            }
            start = next;
        }

        Ok((full_text, aligned_words))
    }

    #[cfg(not(feature = "whisper"))]
    fn transcribe_via_cli(&self, _samples: &[f32], _verbose: bool) -> Result<(String, Vec<AlignedWord>), String> {
        Err("Local Whisper transcription requires compiling with `--features whisper` or `--features metal`.".to_string())
    }
}

#[cfg(feature = "whisper")]
const SAMPLES_PER_MS: usize = 16;

/// whisper.cpp keeps at most half of its 448-token text context as the prompt.
#[cfg(feature = "whisper")]
const CONTEXT_TOKENS: usize = 224;

/// The longest run of words without punctuation in punctuated meeting transcripts is ~34; once Whisper drops punctuation, runs reach hundreds.
#[cfg(feature = "whisper")]
const LOST_STYLE_RUN_WORDS: usize = 40;

/// Words since the last punctuation mark, continuing the `run` carried over from earlier text.
#[cfg(feature = "whisper")]
fn unpunctuated_run(run: usize, text: &str) -> usize {
    text.split_whitespace()
        .fold(run, |run, word| if crate::export::is_punctuated(word) { 0 } else { run + 1 })
}

/// Speakers stutter words and short phrases ("uh, do we need, uh, do we need"), but a phrase of three or four words said four times in a row, or a longer one said three times, is Whisper looping, not speech.
#[cfg(feature = "whisper")]
fn repeats_phrase(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').flat_map(char::to_lowercase).collect())
        .collect();
    (3..=12).any(|n| {
        let times = if n < 5 { 4 } else { 3 };
        (0..(words.len() + 1).saturating_sub(n * times))
            .any(|i| (1..times).all(|k| words[i + k * n..i + (k + 1) * n] == words[i..i + n]))
    })
}

/// Why a pass stopped early.
#[cfg(feature = "whisper")]
#[derive(Clone, Copy)]
enum Stop {
    /// The text ran on without punctuation after text that had it. whisper.cpp decodes a temperature fallback of 0.5 or more without context but still feeds its text to the next window, so one window in the unpunctuated style makes every later window copy it. The pass resumes after that window with the text up to the last punctuation mark as context.
    LostStyle,
    /// The last window, made of `segments` segments, repeats a phrase. Its text is discarded and the window decoded again without the context that led into the loop.
    Looped { segments: usize },
}

#[cfg(feature = "whisper")]
#[derive(Default)]
struct GuardState {
    window: Vec<String>,
    windows: usize,
    run: usize,
    styled_seen: bool,
    retrying_loop: bool,
    stop: Option<Stop>,
}

/// Judges each 30 s window as whisper.cpp finishes it and stops the pass before the next window can be conditioned on a bad one.
#[cfg(feature = "whisper")]
struct WindowGuard {
    state: std::sync::Mutex<GuardState>,
}

#[cfg(feature = "whisper")]
impl WindowGuard {
    /// `retrying_loop` exempts the first window from the loop check, so a window that loops even without context is kept rather than retried forever.
    fn new(styled_seen: bool, retrying_loop: bool) -> Self {
        Self { state: std::sync::Mutex::new(GuardState { styled_seen, retrying_loop, ..Default::default() }) }
    }

    /// whisper.cpp polls this during every encode and decode; segments only arrive between a window's last decode and the next window's encode, so a non-empty buffer holds exactly one finished window.
    fn should_stop(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.stop.is_some() {
            return true;
        }
        if state.window.is_empty() {
            return false;
        }
        let window = std::mem::take(&mut state.window);
        let text = window.join(" ");
        let first_retry_window = state.retrying_loop && state.windows == 0;
        state.windows += 1;
        if !first_retry_window && repeats_phrase(&text) {
            state.stop = Some(Stop::Looped { segments: window.len() });
            return true;
        }
        state.styled_seen |= text.split_whitespace().any(crate::export::is_punctuated);
        state.run = unpunctuated_run(state.run, &text);
        if state.styled_seen && state.run >= LOST_STYLE_RUN_WORDS {
            state.stop = Some(Stop::LostStyle);
        }
        state.stop.is_some()
    }
}

#[cfg(feature = "whisper")]
unsafe extern "C" fn stop_at_bad_window(user_data: *mut c_void) -> bool {
    // SAFETY: `user_data` is the `WindowGuard` the caller keeps alive for the whole pass.
    unsafe { &*(user_data as *const WindowGuard) }.should_stop()
}

/// Appends a segment's text and words, and its text tokens to `context`, moving `punctuated_len` past the last token holding punctuation. Skips a segment Whisper thinks is not speech.
#[cfg(feature = "whisper")]
fn collect_segment(
    seg: &whisper_rs::WhisperSegment<'_>,
    eot: i32,
    offset_ms: i64,
    full_text: &mut String,
    aligned_words: &mut Vec<AlignedWord>,
    context: &mut Vec<i32>,
    punctuated_len: &mut usize,
) {
    let no_speech = seg.no_speech_probability();
    if no_speech > 0.85 {
        return;
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
        context.push(token.token_id());
        if std::str::from_utf8(bytes).is_ok_and(crate::export::is_punctuated) {
            *punctuated_len = context.len();
        }
        let data = token.token_data();
        let (t0, t1) = if data.t_dtw >= 0 {
            (data.t_dtw * 10 + offset_ms, data.t_dtw * 10 + offset_ms)
        } else {
            (data.t0 * 10 + offset_ms, data.t1 * 10 + offset_ms)
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
            flush(&mut current, aligned_words);
            current = Some((bytes.to_vec(), t0, t1, token.token_probability()));
        }
    }
    flush(&mut current, aligned_words);
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

#[cfg(all(test, feature = "whisper"))]
mod tests {
    use super::{LOST_STYLE_RUN_WORDS, repeats_phrase, unpunctuated_run};

    #[test]
    fn test_run_continues_across_windows_and_resets_at_punctuation() {
        assert_eq!(unpunctuated_run(0, "Super caps seem to be falling out of favor for that purpose. because lithium coin cell"), 4);
        assert_eq!(unpunctuated_run(4, "technology has advanced so Joe and"), 10);
        assert_eq!(unpunctuated_run(10, "I'm sorry, guys"), 1);
    }

    #[test]
    fn test_dropped_punctuation_reaches_the_limit_but_punctuated_speech_does_not() {
        let dropped = "here's what you get it rolls over from there if you want more great pay us more yeah for example that the the the uh memory in ipad they they they they send the user i mean they to give the user always a trail";
        assert!(unpunctuated_run(0, dropped) >= LOST_STYLE_RUN_WORDS);
        let punctuated = "Do we have a cap limit, uh, for the modems and the service we're using? Uh, no, currently we can see unlimited. I think the storage is, is so cheap.";
        assert!(unpunctuated_run(0, punctuated) < LOST_STYLE_RUN_WORDS);
    }

    #[test]
    fn test_repeats_phrase_flags_whisper_loops_but_not_stutters() {
        let looped = format!("forward. I think it's going to be a good time.{} Yeah. Um, I have, like, two tickets left.", " I'm going to be a good time.".repeat(8));
        assert!(repeats_phrase(&looped));
        assert!(!repeats_phrase("for example that the the the uh memory in ipad they they they they send the user"));
        assert!(!repeats_phrase("Yeah. Yeah. Yeah. Yeah. Yeah. So we need to use the raw data."));
        assert!(repeats_phrase("the fundamental endpoints. I'm not going to be able to do that. And I'm not going to be able to do that. And I'm not going to be able to do that. And I'm not going to be able to do that. because I can just try"));
        assert!(!repeats_phrase("Is there any... Is there any... verging number of the hardware"));
        assert!(!repeats_phrase("Uh, and I, I was wondering, uh, uh, do we need, uh, do we need, uh, do we need to distribute the logs?"));
    }
}
