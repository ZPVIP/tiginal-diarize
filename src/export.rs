use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlignedWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiarizedSegment {
    pub speaker: usize,
    pub speaker_label: String,
    pub start: f64,
    pub end: f64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub words: Vec<AlignedWord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullDiarizationOutput {
    pub turns: Vec<crate::diarize::Turn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<DiarizedSegment>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diar_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub srt: Option<String>,
}

pub fn format_srt_time(ms: i64) -> String {
    let ms = ms.max(0);
    let total_seconds = ms / 1000;
    let millis = ms % 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

/// Builds human-readable speaker turns, merging consecutive lines from the same speaker.
pub fn build_diarized_text(segments: &[DiarizedSegment]) -> String {
    let mut lines = Vec::new();
    let mut current_speaker: Option<usize> = None;
    let mut current_texts: Vec<&str> = Vec::new();

    for seg in segments {
        let trimmed = seg.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        if Some(seg.speaker) == current_speaker {
            current_texts.push(trimmed);
        } else {
            if let Some(spk) = current_speaker {
                lines.push(format!("Speaker {}: {}", spk + 1, current_texts.join(" ")));
                current_texts.clear();
            }
            current_speaker = Some(seg.speaker);
            current_texts.push(trimmed);
        }
    }

    if let Some(spk) = current_speaker {
        lines.push(format!("Speaker {}: {}", spk + 1, current_texts.join(" ")));
    }

    lines.join("\n\n")
}

/// Builds standard SRT content from segments.
pub fn build_srt_content(segments: &[DiarizedSegment]) -> String {
    let mut srt_blocks = Vec::new();
    let mut index = 1;

    for seg in segments {
        let trimmed = seg.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        let start_str = format_srt_time(seg.start_ms);
        let end_str = format_srt_time(seg.end_ms.max(seg.start_ms + 100));
        let speaker_tag = format!("[Speaker {}]", seg.speaker + 1);

        srt_blocks.push(format!(
            "{}\n{} --> {}\n{} {}",
            index, start_str, end_str, speaker_tag, trimmed
        ));
        index += 1;
    }

    srt_blocks.join("\n\n")
}

/// Assigns words to speaker turns using midpoint intersection.
pub fn match_words_to_turns(
    words: &[AlignedWord],
    turns: &[crate::diarize::Turn],
) -> Vec<DiarizedSegment> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut words_with_speaker: Vec<AlignedWord> = Vec::with_capacity(words.len());

    for w in words {
        let mid = (w.start_ms + w.end_ms) / 2;
        // Find turn containing mid point
        let matched_turn = turns.iter().find(|t| mid >= t.start_ms && mid <= t.end_ms);
        let speaker = if let Some(t) = matched_turn {
            t.speaker
        } else {
            // Find closest turn
            turns
                .iter()
                .min_by_key(|t| {
                    if mid < t.start_ms {
                        t.start_ms - mid
                    } else if mid > t.end_ms {
                        mid - t.end_ms
                    } else {
                        0
                    }
                })
                .map(|t| t.speaker)
                .unwrap_or(0)
        };

        let mut assigned = w.clone();
        assigned.speaker = speaker;
        words_with_speaker.push(assigned);
    }

    // Group adjacent words of same speaker into segments (and split on large pauses > 1.5s)
    let mut segments = Vec::new();
    let mut current_words: Vec<AlignedWord> = Vec::new();

    for w in words_with_speaker {
        if let Some(prev) = current_words.last() {
            let pause = w.start_ms - prev.end_ms;
            let speaker_changed = w.speaker != prev.speaker;
            let long_pause = pause > 1500;

            if speaker_changed || long_pause {
                let start_ms = current_words.first().unwrap().start_ms;
                let end_ms = current_words.last().unwrap().end_ms;
                let spk = current_words.first().unwrap().speaker;
                let text = join_words_cleanly(&current_words);

                segments.push(DiarizedSegment {
                    speaker: spk,
                    speaker_label: format!("Speaker {}", spk + 1),
                    start: start_ms as f64 / 1000.0,
                    end: end_ms as f64 / 1000.0,
                    start_ms,
                    end_ms,
                    text,
                    words: std::mem::take(&mut current_words),
                });
            }
        }
        current_words.push(w);
    }

    if !current_words.is_empty() {
        let start_ms = current_words.first().unwrap().start_ms;
        let end_ms = current_words.last().unwrap().end_ms;
        let spk = current_words.first().unwrap().speaker;
        let text = join_words_cleanly(&current_words);

        segments.push(DiarizedSegment {
            speaker: spk,
            speaker_label: format!("Speaker {}", spk + 1),
            start: start_ms as f64 / 1000.0,
            end: end_ms as f64 / 1000.0,
            start_ms,
            end_ms,
            text,
            words: current_words,
        });
    }

    segments
}

fn join_words_cleanly(words: &[AlignedWord]) -> String {
    let mut text = String::new();
    for (i, w) in words.iter().enumerate() {
        let word = w.word.trim();
        if word.is_empty() {
            continue;
        }
        if i == 0 {
            text.push_str(word);
        } else {
            // Check if CJK or English space is needed
            let prev_last = text.chars().last().unwrap_or(' ');
            let curr_first = word.chars().next().unwrap_or(' ');
            let prev_is_cjk = is_cjk(prev_last);
            let curr_is_cjk = is_cjk(curr_first);

            if !prev_is_cjk && !curr_is_cjk && !word.starts_with(|c: char| c.is_ascii_punctuation()) {
                text.push(' ');
            }
            text.push_str(word);
        }
    }
    text
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |
        '\u{3400}'..='\u{4DBF}' |
        '\u{20000}'..='\u{2A6DF}' |
        '\u{3040}'..='\u{309F}' |
        '\u{30A0}'..='\u{30FF}' |
        '\u{AC00}'..='\u{D7AF}'
    )
}
