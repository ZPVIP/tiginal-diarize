use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::export::AlignedWord;

pub struct MmsAligner {
    session: Session,
    vocab: HashMap<char, usize>,
    blank_id: usize,
}

impl MmsAligner {
    pub fn load(model_path: &Path, vocab_path: Option<&Path>, verbose: bool) -> Result<Self, String> {
        if verbose {
            eprintln!("[info] Loading MMS-Align ONNX model from: {}", model_path.display());
        }

        let session = Session::builder()
            .map_err(|e| format!("Failed to create ort SessionBuilder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| format!("Failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load MMS-Align session: {e}"))?;

        let (vocab, blank_id) = if let Some(v_path) = vocab_path {
            if v_path.exists() {
                load_vocab_json(v_path)?
            } else {
                default_mms_vocab()
            }
        } else {
            // Check if vocab.json exists beside model_path
            let sibling_vocab = model_path.with_file_name("vocab.json");
            if sibling_vocab.exists() {
                load_vocab_json(&sibling_vocab)?
            } else {
                default_mms_vocab()
            }
        };

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

        if verbose {
            eprintln!("[info] Running MMS-Align forward pass on {} samples ({:.2}s)...", 
                samples.len(), samples.len() as f32 / 16000.0);
        }

        // 1. Prepare input tensor: [1, num_samples]
        let input_tensor = Tensor::from_array(([1, samples.len()], samples.to_vec()))
            .map_err(|e| format!("Failed to create input tensor: {e}"))?;

        let outputs = self.session.run(ort::inputs!["input_values" => input_tensor])
            .map_err(|e| format!("MMS-Align inference failed: {e}"))?;

        let (_name, tensor) = outputs.iter().next()
            .ok_or_else(|| "No output tensor from MMS-Align model".to_string())?;

        // Extract logits: [1, num_frames, vocab_size]
        let (shape, logits_data) = tensor.try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract logits tensor: {e}"))?;

        if shape.len() != 3 {
            return Err(format!("Expected 3D logits [1, frames, vocab], got {:?}", shape));
        }

        let num_frames = shape[1] as usize;
        let vocab_size = shape[2] as usize;

        if verbose {
            eprintln!("[info] Computed logits: {} frames, {} vocab size", num_frames, vocab_size);
        }

        // 2. Perform CTC Trellis Alignment
        let words = align_tokens_with_trellis(
            clean_text,
            logits_data,
            num_frames,
            vocab_size,
            &self.vocab,
            self.blank_id,
            verbose,
        )?;

        Ok(words)
    }
}

/// Builds CTC Trellis and backtracks to get token/word boundaries.
fn align_tokens_with_trellis(
    transcript: &str,
    logits: &[f32],
    num_frames: usize,
    vocab_size: usize,
    vocab: &HashMap<char, usize>,
    blank_id: usize,
    verbose: bool,
) -> Result<Vec<AlignedWord>, String> {
    // 1. Parse transcript into words and target characters
    let raw_words: Vec<&str> = transcript.split_whitespace().collect();
    if raw_words.is_empty() {
        return Ok(Vec::new());
    }

    let mut target_tokens: Vec<(usize, char, usize)> = Vec::new(); // (word_idx, char, token_id)
    for (w_idx, w) in raw_words.iter().enumerate() {
        for c in w.chars() {
            let lower = c.to_lowercase().next().unwrap_or(c);
            let token_id = vocab.get(&lower).copied().unwrap_or(blank_id);
            target_tokens.push((w_idx, c, token_id));
        }
    }

    if target_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let num_tokens = target_tokens.len();
    if num_frames < num_tokens {
        // Fallback: If audio frames are fewer than tokens, use uniform spacing
        if verbose {
            eprintln!("[warn] Audio frames ({}) fewer than tokens ({}); using proportional fallback", num_frames, num_tokens);
        }
        return Ok(fallback_proportional_alignment(transcript, num_frames as f64 * 0.02));
    }

    // 2. Compute Log-Softmax in-place or row-by-row
    // Trellis table: [num_frames, num_tokens]
    // trellis[t][j] = log probability of reaching token j at frame t
    let neg_inf = -1e9f32;
    let mut trellis = vec![neg_inf; num_frames * num_tokens];

    // Helper to get log_prob from logits: log_softmax(logits[t, token])
    let get_log_prob = |t: usize, token_id: usize| -> f32 {
        let row_start = t * vocab_size;
        let row = &logits[row_start..row_start + vocab_size];
        let max_val = row.iter().fold(neg_inf, |a, &b| a.max(b));
        let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum_exp = max_val + sum_exp.ln();
        row[token_id] - log_sum_exp
    };

    // Initialize t=0
    let first_token_id = target_tokens[0].2;
    trellis[0] = get_log_prob(0, first_token_id);

    // Forward Trellis
    for t in 1..num_frames {
        let blank_prob = get_log_prob(t, blank_id);

        for j in 0..num_tokens.min(t + 1) {
            let token_id = target_tokens[j].2;
            let emit_prob = get_log_prob(t, token_id);

            // Path 1: stay on token j (consume blank or same)
            let prev_same = trellis[(t - 1) * num_tokens + j] + blank_prob;

            // Path 2: transition from token j-1
            let prev_trans = if j > 0 {
                trellis[(t - 1) * num_tokens + (j - 1)] + emit_prob
            } else {
                neg_inf
            };

            trellis[t * num_tokens + j] = prev_same.max(prev_trans);
        }
    }

    // 3. Backtracking: from (num_frames - 1, num_tokens - 1) down to (0, 0)
    let mut path = Vec::with_capacity(num_tokens);
    let mut curr_t = num_frames - 1;
    let mut curr_j = num_tokens - 1;

    while curr_j > 0 && curr_t > 0 {
        path.push((curr_t, curr_j));
        let prev_same = trellis[(curr_t - 1) * num_tokens + curr_j];
        let prev_trans = trellis[(curr_t - 1) * num_tokens + (curr_j - 1)];

        if prev_trans >= prev_same {
            curr_j -= 1;
        }
        curr_t -= 1;
    }
    path.push((curr_t, 0));
    path.reverse();

    // 4. Group into word boundaries
    // Each frame = 20ms (downsample factor 320 at 16kHz)
    let frame_duration_ms = 20.0f64;

    let mut word_spans: HashMap<usize, (usize, usize)> = HashMap::new(); // word_idx -> (min_frame, max_frame)
    for (t, j) in path {
        let (word_idx, _, _) = target_tokens[j];
        let entry = word_spans.entry(word_idx).or_insert((t, t));
        entry.0 = entry.0.min(t);
        entry.1 = entry.1.max(t);
    }

    let mut aligned_words = Vec::new();
    for (w_idx, &word_str) in raw_words.iter().enumerate() {
        let (min_f, max_f) = word_spans.get(&w_idx).copied().unwrap_or((0, 0));
        let start_ms = (min_f as f64 * frame_duration_ms).round() as i64;
        let end_ms = ((max_f + 1) as f64 * frame_duration_ms).round() as i64;

        aligned_words.push(AlignedWord {
            word: word_str.to_string(),
            start: start_ms as f64 / 1000.0,
            end: end_ms as f64 / 1000.0,
            start_ms,
            end_ms: end_ms.max(start_ms + 100),
            speaker: 0,
            score: Some(0.95),
        });
    }

    Ok(aligned_words)
}

/// Fallback proportional alignment when CTC cannot be run or model is missing.
pub fn fallback_proportional_alignment(transcript: &str, duration_seconds: f64) -> Vec<AlignedWord> {
    let raw_words: Vec<&str> = transcript.split_whitespace().collect();
    if raw_words.is_empty() {
        return Vec::new();
    }

    let total_chars: usize = raw_words.iter().map(|w| w.chars().count().max(1)).sum();
    let total_ms = (duration_seconds * 1000.0).max(1000.0) as i64;

    let mut current_ms = 0i64;
    let mut result = Vec::new();

    for w in raw_words {
        let c_len = w.chars().count().max(1);
        let word_duration_ms = ((c_len as f64 / total_chars as f64) * total_ms as f64).round() as i64;
        let start_ms = current_ms;
        let end_ms = (current_ms + word_duration_ms).min(total_ms);

        result.push(AlignedWord {
            word: w.to_string(),
            start: start_ms as f64 / 1000.0,
            end: end_ms as f64 / 1000.0,
            start_ms,
            end_ms: end_ms.max(start_ms + 60),
            speaker: 0,
            score: None,
        });

        current_ms = end_ms;
    }

    result
}

fn load_vocab_json(path: &Path) -> Result<(HashMap<char, usize>, usize), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read vocab file {}: {e}", path.display()))?;
    let raw: HashMap<String, usize> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse vocab JSON: {e}"))?;

    let mut vocab = HashMap::new();
    let mut blank_id = 0;

    for (k, v) in raw {
        if k == "<pad>" || k == "[pad]" || k == "<blank>" {
            blank_id = v;
        } else if let Some(c) = k.chars().next() {
            vocab.insert(c, v);
        }
    }

    Ok((vocab, blank_id))
}

fn default_mms_vocab() -> (HashMap<char, usize>, usize) {
    let mut vocab = HashMap::new();
    // Default Wav2Vec2/MMS English mapping: 0 = pad, 1-26 = a-z, 27 = ' ' or '|'
    let letters = "abcdefghijklmnopqrstuvwxyz";
    for (i, c) in letters.chars().enumerate() {
        vocab.insert(c, i + 1);
    }
    vocab.insert('|', 27);
    vocab.insert('\'', 28);
    (vocab, 0)
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
