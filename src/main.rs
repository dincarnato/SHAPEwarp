use std::ops::Range;

fn main() {
    println!("Hello, world!");
}

struct Scores {
    align_match: Range<f32>,
    align_mismatch: Range<f32>,
}

struct SequenceScores {
    align_match: f32,
    align_mismatch: f32,
}

fn calc_seed_alignment_score_from_reactivity(
    query: &[f32],
    target: &[f32],
    scores: &Scores,
    max_reactivity: f32,
) -> f32 {
    assert_eq!(query.len(), target.len());

    query
        .iter()
        .zip(target)
        .map(|(&query, &target)| calc_base_alignment_score(query, target, scores, max_reactivity))
        .sum()
}

fn calc_seed_alignment_score_from_sequence(
    query: &[Base],
    target: &[Base],
    scores: &SequenceScores,
) -> f32 {
    assert_eq!(query.len(), target.len());

    query
        .iter()
        .zip(target)
        .map(|(&query, &target)| get_sequence_base_alignment_score(query, target, scores))
        .sum()
}

#[inline]
fn calc_base_alignment_score(query: f32, target: f32, scores: &Scores, max_reactivity: f32) -> f32 {
    if query > 1. && target > 1. {
        0f32
    } else if query.is_nan() || target.is_nan() {
        scores.align_match.start
    } else {
        let diff = (query - target).abs();
        if diff < 0.5 {
            diff * (scores.align_match.end - scores.align_match.start) / 0.5
                + scores.align_match.start
        } else {
            (diff - 0.5) * (scores.align_mismatch.end - scores.align_mismatch.start)
                / (max_reactivity - 0.5)
                + scores.align_mismatch.start
        }
    }
}

// TODO: use an appropriate type
type Base = u8;

#[inline]
fn get_sequence_base_alignment_score(query: Base, target: Base, scores: &SequenceScores) -> f32 {
    if query == target {
        scores.align_match
    } else {
        scores.align_mismatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_alignment_score() {
        let scores = Scores {
            align_match: -0.5..2.,
            align_mismatch: -6.0..-0.5,
        };
        let max_reactivity = 1.2;

        assert_eq!(
            calc_base_alignment_score(1.1, 1.2, &scores, max_reactivity),
            0.
        );
        assert_eq!(
            calc_base_alignment_score(f32::NAN, 1.2, &scores, max_reactivity),
            scores.align_match.start,
        );
        assert_eq!(
            calc_base_alignment_score(1.1, f32::NAN, &scores, max_reactivity),
            scores.align_match.start,
        );
        assert!(
            (calc_base_alignment_score(0.2, 0.4, &scores, max_reactivity) - 0.5).abs()
                < f32::EPSILON
        );
        assert!(
            (calc_base_alignment_score(0.4, 0.2, &scores, max_reactivity) - 0.5).abs()
                < f32::EPSILON
        );
        assert!(
            (calc_base_alignment_score(0.1, 0.6, &scores, max_reactivity) + 6.).abs()
                < f32::EPSILON
        );
        assert!(
            (calc_base_alignment_score(0.1, 0.8, &scores, max_reactivity) + 4.428_571).abs()
                < f32::EPSILON * 10.
        );
        assert!(
            (calc_base_alignment_score(0., max_reactivity, &scores, max_reactivity) + 0.5).abs()
                < f32::EPSILON
        );
    }
}
