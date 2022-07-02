use std::mem;

use num_traits::{cast, NumCast};
use rand::prelude::*;

use crate::db_file::Entry;

pub fn make_shuffled_db(
    db: &[Entry],
    offset: usize,
    block_size: usize,
    shuffle_iterations: usize,
) -> Vec<Entry> {
    let mut rng = thread_rng();
    let mut chunk_indices = Vec::new();

    fn resize_chunk_indices(chunk_indices: &mut Vec<usize>, new_len: usize) {
        let old_len = chunk_indices.len();

        let mut index = old_len;
        chunk_indices.resize_with(new_len, move || {
            let new_index = index + 1;
            mem::replace(&mut index, new_index)
        });

        let mut index = 0;
        chunk_indices[..new_len.min(old_len)].fill_with(move || {
            let new_index = index + 1;
            mem::replace(&mut index, new_index)
        });
    }

    match offset {
        0 => db
            .iter()
            .map(|entry| {
                resize_chunk_indices(&mut chunk_indices, entry.sequence.len() / block_size);
                for _ in 0..shuffle_iterations {
                    chunk_indices.shuffle(&mut rng);
                }

                let mut sequence = Vec::with_capacity(entry.sequence.len());
                let mut reactivity = Vec::with_capacity(entry.reactivity.len());
                for &chunk_index in &chunk_indices {
                    sequence.extend_from_slice(get_chunk_without_offset(
                        chunk_index,
                        block_size,
                        &entry.sequence,
                    ));
                    reactivity.extend_from_slice(get_chunk_without_offset(
                        chunk_index,
                        block_size,
                        &entry.reactivity,
                    ));
                }

                Entry {
                    id: entry.id.clone(),
                    sequence,
                    reactivity,
                }
            })
            .collect(),
        _ => db
            .iter()
            .map(|entry| {
                let len = entry.sequence.len() - offset;
                let mut chunks = len / block_size + 1;
                if len % offset != 0 {
                    chunks += 1;
                }

                resize_chunk_indices(&mut chunk_indices, chunks);

                for _ in 0..shuffle_iterations {
                    chunk_indices.shuffle(&mut rng);
                }

                let mut sequence = Vec::with_capacity(entry.sequence.len());
                let mut reactivity = Vec::with_capacity(entry.reactivity.len());
                for &chunk_index in &chunk_indices {
                    sequence.extend_from_slice(get_chunk_with_offset(
                        chunk_index,
                        offset,
                        block_size,
                        &entry.sequence,
                    ));
                    reactivity.extend_from_slice(get_chunk_with_offset(
                        chunk_index,
                        offset,
                        block_size,
                        &entry.reactivity,
                    ));
                }

                Entry {
                    id: entry.id.clone(),
                    sequence,
                    reactivity,
                }
            })
            .collect(),
    }
}

pub(crate) struct ExtremeDistribution {
    pub(crate) mean: f64,
    pub(crate) stddev: f64,
}

impl ExtremeDistribution {
    pub(crate) fn from_sample<T>(sample: &[T]) -> Self
    where
        T: NumCast + Copy,
    {
        let len = sample.len() as f64;
        let len_inv = 1. / len;
        let mean: f64 = sample
            .iter()
            .copied()
            .map(|x| cast::<_, f64>(x).unwrap() * len_inv)
            .sum();

        let stddev = sample
            .iter()
            .copied()
            .map(|x| (cast::<_, f64>(x).unwrap() - mean).powi(2))
            .sum::<f64>()
            / (len - 1.);

        Self { mean, stddev }
    }

    pub(crate) fn p_value<T>(&self, value: T) -> f64
    where
        T: NumCast,
    {
        use std::f64::consts::PI;

        const INV_SQRT_6: f64 = 1. / 2.449489742783178;
        const EULER: f64 = 0.5772156649015329;

        let z_score = (cast::<_, f64>(value).unwrap() - self.mean) / self.stddev;
        1. - (-(-z_score * PI * INV_SQRT_6 - EULER).exp()).exp()
    }
}

fn get_chunk_with_offset<T>(index: usize, offset: usize, block_size: usize, data: &[T]) -> &[T] {
    index
        .checked_sub(1)
        .map(|index| {
            data.get(offset..).map(|data| {
                data.chunks(block_size)
                    .nth(index)
                    .expect("chunk index out of bound")
            })
        })
        .unwrap_or_else(|| data.get(..offset))
        .unwrap_or_default()
}

fn get_chunk_without_offset<T>(index: usize, block_size: usize, data: &[T]) -> &[T] {
    data.chunks(block_size)
        .nth(index)
        .expect("chunk index out of bound")
}
