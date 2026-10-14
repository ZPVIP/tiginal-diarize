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
pub fn build_srt_content(segments: &[DiarizedSegment], include_speaker_tag: bool) -> String {
    let mut srt_blocks = Vec::new();
    let mut index = 1;

    for seg in segments {
        let trimmed = seg.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        let start_str = format_srt_time(seg.start_ms);
        let end_str = format_srt_time(seg.end_ms.max(seg.start_ms + 100));

        let line = if include_speaker_tag {
            format!("[Speaker {}] {}", seg.speaker + 1, trimmed)
        } else {
            trimmed.to_string()
        };

        srt_blocks.push(format!(
            "{}\n{} --> {}\n{}",
            index, start_str, end_str, line
        ));
        index += 1;
    }

    srt_blocks.join("\n\n")
}

/// Whether `text` ends a sentence: a full stop, question or exclamation mark,
/// possibly followed by a closing quote or bracket.
pub fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(['"', '\'', ')', ']', '}', '”', '’', '）', '】'])
        .ends_with(['.', '?', '!', '…', '。', '？', '！'])
}

/// Assigns words to speaker turns using sentence-level boundaries and maximum time overlap.
pub fn match_words_to_turns(
    words: &[AlignedWord],
    turns: &[crate::diarize::Turn],
) -> Vec<DiarizedSegment> {
    if words.is_empty() {
        return Vec::new();
    }

    struct Phrase {
        words: Vec<AlignedWord>,
        start_ms: i64,
        end_ms: i64,
    }

    // Step 1: Initial cuts at sentence ends, speaker changes at pauses (>= 250ms), or long silences (>= 3000ms).
    // A run of continuous speech without pause is NEVER cut in the middle.
    let mut pieces: Vec<Phrase> = Vec::new();
    for w in words {
        let sentence_ended = pieces.last().is_some_and(|p| {
            p.words.last().is_some_and(|last| ends_sentence(&last.word))
        });
        let pause = pieces.last().map_or(0, |p| w.start_ms.saturating_sub(p.end_ms));
        let long_pause = pause >= 3000;
        let turn_at_pause = !turns.is_empty()
            && pieces.last().is_some_and(|p| {
                pause >= 250
                    && crate::diarize::speaker_at(turns, p.start_ms, p.end_ms)
                        != crate::diarize::speaker_at(turns, w.start_ms, w.end_ms)
            });

        match pieces.last_mut() {
            Some(p) if !sentence_ended && !turn_at_pause && !long_pause => {
                p.end_ms = w.end_ms.max(p.end_ms);
                p.words.push(w.clone());
            }
            _ => {
                pieces.push(Phrase {
                    start_ms: w.start_ms,
                    end_ms: w.end_ms.max(w.start_ms + 60),
                    words: vec![w.clone()],
                });
            }
        }
    }

    // Step 2: Sentence integrity protection:
    // A sentence is NEVER split between speakers. If a piece stopped without a sentence-ending
    // punctuation mark and the next piece follows within a normal pause (< 3000ms), join them.
    let mut sentences: Vec<Phrase> = Vec::new();
    for piece in pieces {
        match sentences.last_mut() {
            Some(previous)
                if !ends_sentence(previous.words.last().map_or("", |w| &w.word))
                    && piece.start_ms.saturating_sub(previous.end_ms) < 3000 =>
            {
                previous.end_ms = piece.end_ms.max(previous.end_ms);
                previous.words.extend(piece.words);
            }
            _ => sentences.push(piece),
        }
    }
    let pieces = sentences;

    // Step 3: Assign speakers based on maximum time overlap over the whole sentence,
    // align start time to takeover if speaker changed, and assemble into DiarizedSegments.
    let mut segments: Vec<DiarizedSegment> = Vec::new();
    for mut piece in pieces {
        let speaker = crate::diarize::speaker_at(turns, piece.start_ms, piece.end_ms);
        for w in &mut piece.words {
            w.speaker = speaker;
        }

        let changed = segments.last().is_none_or(|last: &DiarizedSegment| last.speaker != speaker);
        if changed {
            if let Some(start) = crate::diarize::turn_start_near(turns, speaker, piece.start_ms) {
                let floor = segments.last().map_or(0, |last| last.start_ms + 1);
                piece.start_ms = start.max(floor);
                piece.end_ms = piece.end_ms.max(piece.start_ms + 60);
            }
        }

        let text = join_words_cleanly(&piece.words);
        if text.trim().is_empty() {
            continue;
        }

        segments.push(DiarizedSegment {
            speaker,
            speaker_label: format!("Speaker {}", speaker + 1),
            start: piece.start_ms as f64 / 1000.0,
            end: piece.end_ms as f64 / 1000.0,
            start_ms: piece.start_ms,
            end_ms: piece.end_ms,
            text,
            words: piece.words,
        });
    }

    segments
}

pub fn join_words_cleanly(words: &[AlignedWord]) -> String {
    let mut text = String::new();
    for (i, w) in words.iter().enumerate() {
        let word = w.word.trim();
        if word.is_empty() {
            continue;
        }
        if i == 0 || text.is_empty() {
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

pub fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |
        '\u{3400}'..='\u{4DBF}' |
        '\u{20000}'..='\u{2A6DF}' |
        '\u{3040}'..='\u{309F}' |
        '\u{30A0}'..='\u{30FF}' |
        '\u{AC00}'..='\u{D7AF}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::Turn;

    #[test]
    fn test_ends_sentence() {
        assert!(ends_sentence("Hello world."));
        assert!(ends_sentence("Is this working?"));
        assert!(ends_sentence("Yes!"));
        assert!(ends_sentence("Wait..."));
        assert!(ends_sentence("你好世界。"));
        assert!(ends_sentence("真的吗？"));
        assert!(ends_sentence(r#""Quote.""#));
        assert!(!ends_sentence("Hello world"));
        assert!(!ends_sentence("Good"));
        assert!(!ends_sentence("So Hall"));
    }

    #[test]
    fn test_sentence_not_split_across_speakers() {
        // Simulates: "Good thing everyone speaks German."
        // Where "Good" happened to overlap with speaker 1, but the rest of the sentence was speaker 0.
        let words = vec![
            AlignedWord {
                word: "Good".to_string(),
                start: 0.0,
                end: 0.3,
                start_ms: 0,
                end_ms: 300,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "thing".to_string(),
                start: 0.35,
                end: 0.6,
                start_ms: 350,
                end_ms: 600,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "everyone".to_string(),
                start: 0.65,
                end: 1.0,
                start_ms: 650,
                end_ms: 1000,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "speaks".to_string(),
                start: 1.05,
                end: 1.4,
                start_ms: 1050,
                end_ms: 1400,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "German.".to_string(),
                start: 1.45,
                end: 1.9,
                start_ms: 1450,
                end_ms: 1900,
                speaker: 0,
                score: None,
            },
        ];

        let turns = vec![
            Turn {
                speaker: 1,
                start: 0.0,
                end: 0.32,
                start_ms: 0,
                end_ms: 320,
            },
            Turn {
                speaker: 0,
                start: 0.32,
                end: 2.0,
                start_ms: 320,
                end_ms: 2000,
            },
        ];

        let segments = match_words_to_turns(&words, &turns);
        // Sentence must NOT be split into Speaker 2: "Good" and Speaker 1: "thing everyone speaks German."
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, 0); // Majority overlap is Speaker 0
        assert_eq!(segments[0].text, "Good thing everyone speaks German.");
    }

    #[test]
    fn test_distinct_turns_with_pause_split_properly() {
        let words = vec![
            AlignedWord {
                word: "Hello.".to_string(),
                start: 0.0,
                end: 0.5,
                start_ms: 0,
                end_ms: 500,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "Hi there.".to_string(),
                start: 1.0,
                end: 1.6,
                start_ms: 1000,
                end_ms: 1600,
                speaker: 0,
                score: None,
            },
        ];

        let turns = vec![
            Turn {
                speaker: 0,
                start: 0.0,
                end: 0.6,
                start_ms: 0,
                end_ms: 600,
            },
            Turn {
                speaker: 1,
                start: 0.9,
                end: 1.7,
                start_ms: 900,
                end_ms: 1700,
            },
        ];

        let segments = match_words_to_turns(&words, &turns);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, 0);
        assert_eq!(segments[0].text, "Hello.");
        assert_eq!(segments[1].speaker, 1);
        assert_eq!(segments[1].text, "Hi there.");
    }

    #[test]
    fn test_join_words_cleanly_punctuation() {
        let words = vec![
            AlignedWord {
                word: "Here's".to_string(),
                start: 0.0,
                end: 0.4,
                start_ms: 0,
                end_ms: 400,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "modalities,".to_string(),
                start: 0.45,
                end: 0.9,
                start_ms: 450,
                end_ms: 900,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "loosening".to_string(),
                start: 0.95,
                end: 1.3,
                start_ms: 950,
                end_ms: 1300,
                speaker: 0,
                score: None,
            },
            AlignedWord {
                word: "piezo.".to_string(),
                start: 1.35,
                end: 1.7,
                start_ms: 1350,
                end_ms: 1700,
                speaker: 0,
                score: None,
            },
        ];
        assert_eq!(join_words_cleanly(&words), "Here's modalities, loosening piezo.");
    }
}
