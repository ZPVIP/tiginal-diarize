use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::nemotron::Model;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Identifies speaker turns in 16 kHz mono samples.
/// If `speakers` is `Some(n)`, restricts to the `n` most active speakers.
/// If `None`, automatically drops/absorbs tiny clusters (<4s or <4% speech).
pub fn turns(
    samples: &[f32],
    speakers: Option<usize>,
    model_path: &Path,
    verbose: bool,
) -> Result<Vec<Turn>, String> {
    let mut model = Model::load(model_path, verbose)?;
    let probs = model.probabilities(samples, verbose)?;

    if verbose {
        eprintln!("[info] Thresholding speaker probabilities and bridging short pauses...");
    }

    let raw = segments(&probs, 8);
    let raw = match speakers {
        Some(n) => keep_largest(raw, n),
        None => absorb_small_clusters(raw),
    };

    let turns = renumber(raw);
    if verbose {
        eprintln!("[info] Identified {} speaker turns across {} unique speakers", 
            turns.len(), 
            turns.iter().map(|t| t.speaker).max().map(|m| m + 1).unwrap_or(0));
    }

    Ok(turns)
}

/// Stretches where a speaker's probability is > 0.5.
/// A frame is 10 ms; pauses < 500 ms are bridged, blips < 300 ms dropped.
fn segments(probs: &[f32], speakers: usize) -> Vec<(i64, i64, i32)> {
    let frames = probs.len() / speakers;
    let mut raw = Vec::new();

    for s in 0..speakers {
        let mut runs: Vec<(i64, i64)> = Vec::new();
        let mut start = None;

        for f in 0..=frames {
            let on = f < frames && probs[f * speakers + s] > 0.5;
            match (on, start) {
                (true, None) => start = Some(f),
                (false, Some(from)) => {
                    let (from_ms, to_ms) = (from as i64 * 10, f as i64 * 10);
                    match runs.last_mut() {
                        Some(last) if from_ms - last.1 < 500 => last.1 = to_ms,
                        _ => runs.push((from_ms, to_ms)),
                    }
                    start = None;
                }
                _ => {}
            }
        }

        raw.extend(
            runs.into_iter()
                .filter(|(from_ms, to_ms)| to_ms - from_ms >= 300)
                .map(|(from_ms, to_ms)| (from_ms, to_ms, s as i32)),
        );
    }
    raw
}

fn keep_largest(raw: Vec<(i64, i64, i32)>, n: usize) -> Vec<(i64, i64, i32)> {
    let mut spoken = HashMap::<i32, i64>::new();
    for (start, end, id) in &raw {
        *spoken.entry(*id).or_default() += end - start;
    }
    let mut ranked: Vec<(i32, i64)> = spoken.into_iter().collect();
    ranked.sort_by_key(|(id, ms)| (-ms, *id));
    let kept: Vec<i32> = ranked.iter().take(n.max(1)).map(|(id, _)| *id).collect();
    reassign(raw, |id| kept.contains(id))
}

fn absorb_small_clusters(raw: Vec<(i64, i64, i32)>) -> Vec<(i64, i64, i32)> {
    let mut spoken = HashMap::<i32, i64>::new();
    for (start, end, id) in &raw {
        *spoken.entry(*id).or_default() += end - start;
    }
    let total: i64 = spoken.values().sum();
    let floor = (total * 4 / 100).max(4000);
    reassign(raw, |id| spoken.get(id).is_some_and(|ms| *ms >= floor))
}

fn reassign(raw: Vec<(i64, i64, i32)>, keeps: impl Fn(&i32) -> bool) -> Vec<(i64, i64, i32)> {
    let anchors: Vec<(i64, i64, i32)> =
        raw.iter().copied().filter(|(_, _, id)| keeps(id)).collect();
    if anchors.is_empty() {
        return raw;
    }
    raw.iter()
        .map(|&(start, end, id)| {
            if keeps(&id) {
                return (start, end, id);
            }
            let middle = (start + end) / 2;
            let nearest = anchors
                .iter()
                .min_by_key(|(s, e, _)| {
                    if middle < *s {
                        s - middle
                    } else {
                        (middle - e).max(0)
                    }
                })
                .map_or(id, |(_, _, other)| *other);
            (start, end, nearest)
        })
        .collect()
}

fn renumber(mut raw: Vec<(i64, i64, i32)>) -> Vec<Turn> {
    raw.sort_by_key(|(start, _, _)| *start);
    let mut order: Vec<i32> = Vec::new();
    raw.into_iter()
        .map(|(start_ms, end_ms, id)| {
            let speaker = order.iter().position(|o| *o == id).unwrap_or_else(|| {
                order.push(id);
                order.len() - 1
            });
            Turn {
                speaker,
                start: start_ms as f64 / 1000.0,
                end: end_ms as f64 / 1000.0,
                start_ms,
                end_ms,
            }
        })
        .collect()
}
