mod cli;
mod db_file;
mod query_file;

use clap::Parser;
use std::ops::Range;

use crate::cli::Cli;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let cli = Cli::parse();
    let query = query_file::read_file(&cli.query, cli.max_reactivity)?;
    let db = db_file::read_file(&cli.database)?;
    dbg!(query, db);

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Base {
    A,
    C,
    G,
    T,
    N,
}

impl Base {
    fn try_from_nibble(nibble: u8) -> Result<Self, InvalidEncodedBase> {
        use Base::*;
        Ok(match nibble {
            0 => A,
            1 => C,
            2 => G,
            3 => T,
            4 => N,
            _ => return Err(InvalidEncodedBase),
        })
    }

    fn try_pair_from_byte(byte: u8) -> Result<[Self; 2], InvalidEncodedBase> {
        let first = Base::try_from_nibble(byte >> 4)?;
        let second = Base::try_from_nibble(byte & 0x0F)?;

        Ok([first, second])
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::A => b'A',
            Self::C => b'C',
            Self::G => b'G',
            Self::T => b'T',
            Self::N => b'N',
        }
    }
}

impl TryFrom<u8> for Base {
    type Error = InvalidEncodedBase;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            b'A' => Self::A,
            b'C' => Self::C,
            b'G' => Self::G,
            b'T' => Self::T,
            b'N' => Self::N,
            _ => return Err(InvalidEncodedBase),
        })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InvalidEncodedBase;

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
