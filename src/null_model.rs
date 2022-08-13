use std::mem;

use num_traits::{cast, NumCast};
use rand::prelude::*;

use crate::db_file::Entry;

#[inline]
pub fn make_shuffled_db(db: &[Entry], block_size: usize, shuffle_iterations: usize) -> Vec<Entry> {
    make_shuffled_db_inner(db, block_size, shuffle_iterations, rand::thread_rng())
}

fn make_shuffled_db_inner<R: Rng>(
    db: &[Entry],
    block_size: usize,
    shuffle_iterations: usize,
    mut rng: R,
) -> Vec<Entry> {
    let mut chunk_indices = Vec::new();

    db.iter()
        .map(|entry| {
            let (offset, chunks) =
                get_random_offset_and_chunks(entry.sequence.len(), block_size, &mut rng);

            resize_indices(&mut chunk_indices, chunks);
            for _ in 0..shuffle_iterations {
                chunk_indices.shuffle(&mut rng);
            }

            let mut sequence = Vec::with_capacity(entry.sequence.len());
            let mut reactivity = Vec::with_capacity(entry.reactivity.len());

            match offset {
                0 => {
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
                }
                _ => {
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
                }
            }

            Entry {
                id: entry.id.clone(),
                sequence,
                reactivity,
            }
        })
        .collect()
}

fn resize_indices(indices: &mut Vec<usize>, new_size: usize) {
    let old_len = indices.len();

    let mut index = old_len;
    indices.resize_with(new_size, move || {
        let new_index = index + 1;
        mem::replace(&mut index, new_index)
    });

    let mut index = 0;
    indices[..new_size.min(old_len)].fill_with(move || {
        let new_index = index + 1;
        mem::replace(&mut index, new_index)
    });
}

fn get_random_offset_and_chunks<R: Rng>(
    len: usize,
    block_size: usize,
    mut rng: R,
) -> (usize, usize) {
    let block_remainder = len % block_size;
    let offset = (block_remainder > 0)
        .then(|| rng.gen_range(0..block_remainder))
        .unwrap_or(0);

    let len_without_offset = len - offset;
    let aux_chunks = match (offset, len_without_offset % block_size) {
        (0, 0) => 0,
        (_, 0) | (0, _) => 1,
        (_, _) => 2,
    };
    let chunks = len_without_offset / block_size + aux_chunks;

    (offset, chunks)
}

#[derive(Debug)]
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
            .sqrt()
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

#[inline]
fn get_chunk_with_offset<T>(index: usize, offset: usize, block_size: usize, data: &[T]) -> &[T] {
    match index.checked_sub(1) {
        Some(index) => data
            .get(offset..)
            .map(|data| {
                data.chunks(block_size)
                    .nth(index)
                    .expect("chunk index out of bound")
            })
            .unwrap_or_default(),

        None => data.get(..offset).unwrap_or_default(),
    }
}

#[inline]
fn get_chunk_without_offset<T>(index: usize, block_size: usize, data: &[T]) -> &[T] {
    data.chunks(block_size)
        .nth(index)
        .expect("chunk index out of bound")
}
