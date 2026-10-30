use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::export::AlignedWord;

const SAMPLE_RATE: usize = 16_000;
const INFERENCE_CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE;
const INFERENCE_OVERLAP_SAMPLES: usize = 2 * SAMPLE_RATE;
const ALIGNMENT_BEAM_STATES: usize = 512;

struct AcousticFrames {
    logits: Vec<f32>,
    times_ms: Vec<i64>,
    vocab_size: usize,
}

#[derive(Clone, Copy)]
struct TranscriptToken {
    word_index: usize,
    token_id: usize,
}

struct TraceFrame {
    states: Vec<u32>,
    previous_offsets: Vec<u8>,
}

pub struct MmsAligner {
    session: Session,
    vocab: HashMap<char, usize>,
    blank_id: usize,
}

impl MmsAligner {
    pub fn load(
        model_path: &Path,
        vocab_path: Option<&Path>,
        verbose: bool,
    ) -> Result<Self, String> {
        if verbose {
            eprintln!(
                "[info] Loading MMS-Align ONNX model from: {}",
                model_path.display()
            );
        }

        let session = Session::builder()
            .map_err(|e| format!("Failed to create ort SessionBuilder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| format!("Failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load MMS-Align session: {e}"))?;

        let resolved_vocab_path = resolve_vocab_path(model_path, vocab_path).ok_or_else(|| {
            format!(
                "MMS-Align requires the vocab.json shipped with its forced-alignment model; no matching vocabulary was found for {}",
                model_path.display(),
            )
        })?;
        if verbose {
            eprintln!(
                "[info] Loading MMS-Align vocabulary from: {}",
                resolved_vocab_path.display(),
            );
        }
        let (vocab, blank_id) = load_vocab_json(&resolved_vocab_path)?;

        Ok(Self {
            session,
            vocab,
            blank_id,
        })
    }

    /// Aligns transcript text with audio samples.
    /// Returns word-level aligned timestamps in milliseconds.
    pub fn align(
        &mut self,
        samples: &[f32],
        transcript: &str,
        verbose: bool,
    ) -> Result<Vec<AlignedWord>, String> {
        let clean_text = transcript.trim();
        if clean_text.is_empty() || samples.is_empty() {
            return Ok(Vec::new());
        }

        let frames = self.infer_chunked(samples, verbose)?;
        align_tokens_with_trellis(
            clean_text,
            &frames.logits,
            &frames.times_ms,
            frames.vocab_size,
            &self.vocab,
            self.blank_id,
            verbose,
        )
    }

    fn infer_chunked(&mut self, samples: &[f32], verbose: bool) -> Result<AcousticFrames, String> {
        let step_samples = INFERENCE_CHUNK_SAMPLES - INFERENCE_OVERLAP_SAMPLES;
        let total_chunks = samples.len().saturating_sub(1) / step_samples + 1;
        let mut logits = Vec::new();
        let mut times_ms = Vec::new();
        let mut expected_vocab_size = None;
        let mut chunk_start = 0;
        let mut chunk_index = 0;

        while chunk_start < samples.len() {
            let chunk_end = (chunk_start + INFERENCE_CHUNK_SAMPLES).min(samples.len());
            let chunk = &samples[chunk_start..chunk_end];
            if verbose {
                eprintln!(
                    "[progress] alignment: {}",
                    (chunk_index * 100 / total_chunks).min(99),
                );
                eprintln!(
                    "[info] MMS-Align acoustic chunk {}/{}: {:.2}s to {:.2}s",
                    chunk_index + 1,
                    total_chunks,
                    chunk_start as f64 / SAMPLE_RATE as f64,
                    chunk_end as f64 / SAMPLE_RATE as f64,
                );
            }

            let input_tensor = Tensor::from_array(([1, chunk.len()], chunk.to_vec()))
                .map_err(|e| format!("Failed to create input tensor: {e}"))?;
            let outputs = self
                .session
                .run(ort::inputs!["input_values" => input_tensor])
                .map_err(|e| format!("MMS-Align inference failed: {e}"))?;
            let (_name, tensor) = outputs
                .iter()
                .next()
                .ok_or_else(|| "No output tensor from MMS-Align model".to_string())?;
            let (shape, chunk_logits) = tensor
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Failed to extract logits tensor: {e}"))?;
            if shape.len() != 3 {
                return Err(format!(
                    "Expected 3D logits [1, frames, vocab], got {:?}",
                    shape
                ));
            }
            let frame_count = shape[1] as usize;
            let vocab_size = shape[2] as usize;
            if frame_count == 0 || vocab_size == 0 {
                return Err("MMS-Align returned an empty acoustic chunk".to_string());
            }
            if let Some(expected) = expected_vocab_size {
                if expected != vocab_size {
                    return Err(format!(
                        "MMS-Align vocabulary changed between chunks: {expected} then {vocab_size}",
                    ));
                }
            } else {
                expected_vocab_size = Some(vocab_size);
                let largest_token_id = self
                    .vocab
                    .values()
                    .copied()
                    .chain(std::iter::once(self.blank_id))
                    .max()
                    .unwrap_or(0);
                if largest_token_id + 1 != vocab_size {
                    return Err(format!(
                        "MMS-Align model exposes {vocab_size} output labels, but its vocabulary defines {} labels; the model and vocab.json do not match",
                        largest_token_id + 1,
                    ));
                }
            }

            let half_overlap = INFERENCE_OVERLAP_SAMPLES / 2;
            let owned_start = if chunk_start == 0 {
                0
            } else {
                chunk_start + half_overlap
            };
            let owned_end = if chunk_end == samples.len() {
                samples.len()
            } else {
                chunk_end - half_overlap
            };
            for frame in 0..frame_count {
                let center_in_chunk = ((2 * frame + 1) * chunk.len()) / (2 * frame_count);
                let center_sample = chunk_start + center_in_chunk;
                if center_sample < owned_start || center_sample >= owned_end {
                    continue;
                }
                let row_start = frame * vocab_size;
                logits.extend_from_slice(&chunk_logits[row_start..row_start + vocab_size]);
                times_ms.push((center_sample * 1000 / SAMPLE_RATE) as i64);
            }

            if chunk_end == samples.len() {
                break;
            }
            chunk_start += step_samples;
            chunk_index += 1;
        }

        if times_ms.is_empty() {
            return Err("MMS-Align produced no usable acoustic frames".to_string());
        }
        if verbose {
            eprintln!("[progress] alignment: 100");
            eprintln!(
                "[info] MMS-Align produced {} global acoustic frames from {} chunks",
                times_ms.len(),
                total_chunks,
            );
        }
        Ok(AcousticFrames {
            logits,
            times_ms,
            vocab_size: expected_vocab_size.unwrap_or(0),
        })
    }
}

/// Builds CTC Trellis and backtracks to get token/word boundaries.
fn align_tokens_with_trellis(
    transcript: &str,
    logits: &[f32],
    frame_times_ms: &[i64],
    vocab_size: usize,
    vocab: &HashMap<char, usize>,
    blank_id: usize,
    verbose: bool,
) -> Result<Vec<AlignedWord>, String> {
    let num_frames = frame_times_ms.len();
    if logits.len() != num_frames * vocab_size {
        return Err("Acoustic logits and frame timestamps have inconsistent sizes".to_string());
    }
    let raw_words: Vec<&str> = transcript.split_whitespace().collect();
    if raw_words.is_empty() {
        return Ok(Vec::new());
    }

    let mut target_tokens = Vec::new();
    for (w_idx, w) in raw_words.iter().enumerate() {
        for c in w.chars() {
            let lower = c.to_lowercase().next().unwrap_or(c);
            if let Some(&token_id) = vocab.get(&lower) {
                target_tokens.push(TranscriptToken {
                    word_index: w_idx,
                    token_id,
                });
            }
        }
    }

    if target_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let num_tokens = target_tokens.len();
    let state_count = num_tokens * 2 + 1;
    let final_token_state = state_count - 2;
    let final_blank_state = state_count - 1;
    if num_frames < num_tokens {
        return Err(format!(
            "MMS-Align has fewer acoustic frames ({num_frames}) than transcript tokens ({num_tokens})",
        ));
    }

    // 2. Precompute Log-Sum-Exp once per frame: O(num_frames * vocab_size)
    // Avoid re-scanning vocab_size in the O(num_frames * num_tokens) inner loop!
    let neg_inf = -1e9f32;
    let mut log_sum_exps = Vec::with_capacity(num_frames);
    for t in 0..num_frames {
        let row_start = t * vocab_size;
        let row = &logits[row_start..row_start + vocab_size];
        let max_val = row.iter().fold(neg_inf, |a, &b| a.max(b));
        let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        log_sum_exps.push(max_val + sum_exp.ln());
    }

    let get_log_prob =
        |t: usize, token_id: usize| -> f32 { logits[t * vocab_size + token_id] - log_sum_exps[t] };

    let label_for_state = |state: usize| -> usize {
        if state.is_multiple_of(2) {
            blank_id
        } else {
            target_tokens[(state - 1) / 2].token_id
        }
    };
    let mut previous_states = vec![0usize];
    let mut previous_scores = vec![0.0f32];
    let mut traces = Vec::with_capacity(num_frames);
    let mut candidate_scores = vec![neg_inf; state_count];
    let mut candidate_offsets = vec![u8::MAX; state_count];
    let mut touched_states = Vec::with_capacity(ALIGNMENT_BEAM_STATES * 3);

    for frame in 0..num_frames {
        let remaining_frames = num_frames - frame - 1;
        let minimum_completable_state = final_token_state.saturating_sub(remaining_frames * 2);
        for (&previous_state, &previous_score) in
            previous_states.iter().zip(previous_scores.iter())
        {
            for offset in 0..=2usize {
                let state = previous_state + offset;
                if state >= state_count || state < minimum_completable_state {
                    continue;
                }
                if offset == 2 {
                    if state.is_multiple_of(2) {
                        continue;
                    }
                    let token_index = (state - 1) / 2;
                    if target_tokens[token_index].token_id
                        == target_tokens[token_index - 1].token_id
                    {
                        continue;
                    }
                }
                let score = previous_score + get_log_prob(frame, label_for_state(state));
                if candidate_offsets[state] == u8::MAX {
                    touched_states.push(state);
                }
                if score > candidate_scores[state] {
                    candidate_scores[state] = score;
                    candidate_offsets[state] = offset as u8;
                }
            }
        }

        if touched_states.is_empty() {
            return Err(format!(
                "MMS-Align lost every CTC path at acoustic frame {frame}"
            ));
        }
        let all_touched_states = touched_states.clone();
        touched_states.sort_unstable_by(|left, right| {
            candidate_scores[*right].total_cmp(&candidate_scores[*left])
        });
        touched_states.truncate(ALIGNMENT_BEAM_STATES);
        touched_states.sort_unstable();

        previous_states.clear();
        previous_scores.clear();
        let mut retained_offsets = Vec::with_capacity(touched_states.len());
        for &state in &touched_states {
            previous_states.push(state);
            previous_scores.push(candidate_scores[state]);
            retained_offsets.push(candidate_offsets[state]);
        }
        traces.push(TraceFrame {
            states: previous_states
                .iter()
                .map(|state| *state as u32)
                .collect(),
            previous_offsets: retained_offsets,
        });

        touched_states.clear();
        for state in all_touched_states {
            candidate_scores[state] = neg_inf;
            candidate_offsets[state] = u8::MAX;
        }
    }

    let final_score = |state: usize| -> f32 {
        previous_states
            .binary_search(&state)
            .ok()
            .map(|index| previous_scores[index])
            .unwrap_or(neg_inf)
    };
    let mut state = if final_score(final_blank_state) >= final_score(final_token_state) {
        final_blank_state
    } else {
        final_token_state
    };
    if final_score(state) <= neg_inf / 2.0 {
        return Err(format!(
            "MMS-Align could not match the complete transcript within a {}-state acoustic beam",
            ALIGNMENT_BEAM_STATES,
        ));
    }

    let mut token_spans: Vec<Option<(usize, usize, f64, usize)>> = vec![None; num_tokens];
    for frame in (0..num_frames).rev() {
        if !state.is_multiple_of(2) {
            let token_index = (state - 1) / 2;
            let probability = get_log_prob(frame, target_tokens[token_index].token_id).exp() as f64;
            let span = token_spans[token_index].get_or_insert((frame, frame, 0.0, 0));
            span.0 = span.0.min(frame);
            span.1 = span.1.max(frame);
            span.2 += probability;
            span.3 += 1;
        }
        let trace = &traces[frame];
        let state_index = trace
            .states
            .binary_search(&(state as u32))
            .map_err(|_| "MMS-Align backtracking left the retained acoustic beam".to_string())?;
        let offset = trace.previous_offsets[state_index];
        if offset == u8::MAX || offset as usize > state {
            return Err("MMS-Align encountered an invalid CTC backpointer".to_string());
        }
        state -= offset as usize;
    }
    if state != 0 {
        return Err("MMS-Align backtracking did not return to the initial blank state".to_string());
    }

    let default_frame_ms = frame_times_ms
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .find(|duration| *duration > 0)
        .unwrap_or(20);
    let mut word_spans: HashMap<usize, (usize, usize, f64, usize)> = HashMap::new();
    for (token_index, span) in token_spans.into_iter().enumerate() {
        let Some((start_frame, end_frame, score_sum, score_count)) = span else {
            return Err(format!(
                "MMS-Align did not emit transcript token {token_index}"
            ));
        };
        let word_index = target_tokens[token_index].word_index;
        let entry = word_spans
            .entry(word_index)
            .or_insert((start_frame, end_frame, 0.0, 0));
        entry.0 = entry.0.min(start_frame);
        entry.1 = entry.1.max(end_frame);
        entry.2 += score_sum;
        entry.3 += score_count;
    }

    let mut aligned_words = Vec::new();
    for (w_idx, &word_str) in raw_words.iter().enumerate() {
        let Some((min_f, max_f, score_sum, score_count)) = word_spans.get(&w_idx).copied() else {
            continue;
        };
        let start_ms = frame_times_ms[min_f];
        let end_ms = frame_times_ms
            .get(max_f + 1)
            .copied()
            .unwrap_or(frame_times_ms[max_f] + default_frame_ms);

        aligned_words.push(AlignedWord {
            word: word_str.to_string(),
            start: start_ms as f64 / 1000.0,
            end: end_ms as f64 / 1000.0,
            start_ms,
            end_ms: end_ms.max(start_ms + 100),
            speaker: 0,
            score: Some((score_sum / score_count.max(1) as f64) as f32),
        });
    }
    if verbose {
        eprintln!(
            "[info] CTC forced alignment matched {} words across {} acoustic frames",
            aligned_words.len(),
            num_frames,
        );
    }
    Ok(aligned_words)
}

fn resolve_vocab_path(model_path: &Path, explicit_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit_path {
        return path.is_file().then(|| path.to_path_buf());
    }

    let sibling = model_path.with_file_name("vocab.json");
    if sibling.is_file() {
        return Some(sibling);
    }

    let parent_sibling = model_path.parent()?.parent()?.join("vocab.json");
    parent_sibling.is_file().then_some(parent_sibling)
}

fn parse_vocab_json(content: &str) -> Result<(HashMap<char, usize>, usize), String> {
    let raw: HashMap<String, usize> =
        serde_json::from_str(content).map_err(|e| format!("Failed to parse vocab JSON: {e}"))?;
    let blank_id = raw
        .get("<blank>")
        .or_else(|| raw.get("[blank]"))
        .copied()
        .ok_or_else(|| "Forced-alignment vocabulary does not define <blank>".to_string())?;

    let vocab = raw
        .into_iter()
        .filter_map(|(label, token_id)| {
            let mut characters = label.chars();
            let character = characters.next()?;
            characters.next().is_none().then_some((character, token_id))
        })
        .collect();

    Ok((vocab, blank_id))
}

fn load_vocab_json(path: &Path) -> Result<(HashMap<char, usize>, usize), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read vocab file {}: {e}", path.display()))?;
    parse_vocab_json(&content)
}

/// Resolves default path for MMS-Align ONNX model.
pub fn default_mms_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tiginal")
        .join("models")
        .join("mms-align")
        .join("model.onnx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_alignment_follows_acoustic_peaks_instead_of_text_proportion() {
        let num_frames = 12;
        let vocab_size = 3;
        let mut logits = vec![-5.0; num_frames * vocab_size];
        for frame in 0..num_frames {
            logits[frame * vocab_size] = 5.0;
        }
        logits[2 * vocab_size] = -5.0;
        logits[2 * vocab_size + 1] = 10.0;
        logits[8 * vocab_size] = -5.0;
        logits[8 * vocab_size + 2] = 10.0;

        let vocab = HashMap::from([('a', 1), ('b', 2)]);
        let frame_times_ms: Vec<i64> = (0..num_frames).map(|frame| frame as i64 * 20).collect();
        let words = align_tokens_with_trellis(
            "a b",
            &logits,
            &frame_times_ms,
            vocab_size,
            &vocab,
            0,
            false,
        )
        .expect("alignment should succeed");

        assert_eq!(words[0].start_ms, 40);
        assert_eq!(words[1].start_ms, 160);
    }

    #[test]
    fn forced_alignment_vocab_uses_blank_instead_of_pad() {
        let (vocab, blank_id) = parse_vocab_json(
            r#"{
                "<blank>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "a": 4,
                "z": 26,
                "'": 28
            }"#,
        )
        .expect("forced-alignment vocabulary should parse");

        assert_eq!(blank_id, 0);
        assert_eq!(vocab.get(&'a'), Some(&4));
        assert_eq!(vocab.get(&'z'), Some(&26));
        assert_eq!(vocab.get(&'\''), Some(&28));
        assert!(!vocab.contains_key(&'<'));
    }
}
