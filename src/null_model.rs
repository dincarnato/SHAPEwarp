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
    pub(crate) location: f64,
    pub(crate) scale: f64,
}

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

impl ExtremeDistribution {
    pub(crate) fn from_sample<T>(sample: &[T]) -> Self
    where
        T: NumCast + Copy,
    {
        let len = sample.len();
        match len {
            0 => {
                return Self {
                    location: 0.,
                    scale: 0.,
                };
            }
            1 => {
                return Self {
                    location: cast(sample[0]).unwrap(),
                    scale: 0.,
                };
            }
            _ => {}
        }

        let len = len as f64;
        let len_inv = 1. / len;
        let mean: f64 = sample
            .iter()
            .copied()
            .map(|x| cast::<_, f64>(x).unwrap() * len_inv)
            .sum();

        let variance = sample
            .iter()
            .copied()
            .map(|x| (cast::<_, f64>(x).unwrap() - mean).powi(2))
            .sum::<f64>()
            / (len - 1.);

        Self::from_mean_and_variance(mean, variance)
    }

    fn from_mean_and_variance(mean: f64, variance: f64) -> Self {
        use std::f64::consts::PI;

        let scale = (variance * 6. / PI.powi(2)).sqrt();
        let location = mean - scale * EULER_MASCHERONI;

        Self { location, scale }
    }

    pub(crate) fn cdf<T>(&self, value: T) -> f64
    where
        T: NumCast,
    {
        let z = (cast::<_, f64>(value).unwrap() - self.location) / self.scale;
        f64::exp(-f64::exp(-z))
    }

    #[inline]
    pub(crate) fn p_value<T>(&self, value: T) -> f64
    where
        T: NumCast,
    {
        1. - self.cdf(value)
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

#[cfg(test)]
mod tests {
    use std::fs::File;

    use rand::rngs::{mock::StepRng, SmallRng};

    use crate::{db_file, SequenceEntry};

    use super::*;

    #[test]
    fn shuffled_db_with_offset() {
        const SEED: u64 = 10;
        const SEQUENCE_LEN: usize = 1553;
        const BLOCK_SIZE: usize = 13;
        const SHUFFLE_ITERATIONS: usize = 3;
        const EXPECTED_OFFSET: usize = 3;
        // 1 chunk for the initial offset, 1 chunk for the remainder
        const EXPECTED_CHUNKS: usize = (SEQUENCE_LEN - EXPECTED_OFFSET) / BLOCK_SIZE + 1 + 1;
        const EXPECTED_SHUFFLED_INDICES: [usize; EXPECTED_CHUNKS] = [
            43, 65, 15, 119, 37, 68, 48, 104, 110, 28, 70, 88, 18, 55, 50, 103, 4, 114, 49, 67, 83,
            58, 54, 115, 89, 32, 95, 42, 91, 46, 79, 99, 21, 53, 77, 86, 60, 92, 94, 98, 29, 107,
            56, 31, 90, 9, 64, 84, 25, 34, 41, 17, 3, 81, 39, 11, 118, 113, 74, 100, 20, 117, 102,
            22, 97, 62, 75, 111, 57, 72, 76, 112, 87, 96, 2, 106, 35, 36, 7, 13, 12, 51, 82, 40, 1,
            80, 30, 5, 61, 69, 38, 52, 59, 63, 71, 105, 10, 27, 45, 23, 85, 24, 73, 19, 108, 44,
            66, 33, 101, 120, 8, 6, 116, 109, 93, 26, 16, 14, 0, 78, 47,
        ];

        let mut db = db_file::Reader::new(File::open("test_data/test.db").unwrap()).unwrap();
        let entry = db.entries().next().unwrap().unwrap();

        // Check if we are on the same page
        let mut rng = SmallRng::seed_from_u64(SEED);
        assert_eq!(entry.sequence.len(), SEQUENCE_LEN);
        assert_eq!(
            entry.sequence[EXPECTED_OFFSET..].chunks(BLOCK_SIZE).count() + 1,
            EXPECTED_CHUNKS
        );
        assert_eq!(
            rng.gen_range(0..(SEQUENCE_LEN % BLOCK_SIZE)),
            EXPECTED_OFFSET
        );
        let mut indices = Vec::new();
        super::resize_indices(&mut indices, EXPECTED_CHUNKS);
        std::iter::repeat(())
            .take(SHUFFLE_ITERATIONS)
            .for_each(|()| indices.shuffle(&mut rng));
        assert_eq!(indices, EXPECTED_SHUFFLED_INDICES);
        drop(indices);

        let expected_sequence: Vec<_> = EXPECTED_SHUFFLED_INDICES
            .iter()
            .copied()
            .flat_map(|chunk_index| {
                get_chunk_with_offset(chunk_index, EXPECTED_OFFSET, BLOCK_SIZE, &entry.sequence)
            })
            .copied()
            .collect();

        let expected_reactivity: Vec<_> = EXPECTED_SHUFFLED_INDICES
            .iter()
            .copied()
            .flat_map(|chunk_index| {
                get_chunk_with_offset(chunk_index, EXPECTED_OFFSET, BLOCK_SIZE, &entry.reactivity)
            })
            .copied()
            .collect();

        let entry_id = entry.id.clone();

        let shuffled_entries = make_shuffled_db_inner(
            &[entry],
            BLOCK_SIZE,
            SHUFFLE_ITERATIONS,
            SmallRng::seed_from_u64(SEED),
        );

        assert_eq!(shuffled_entries.len(), 1);
        let shuffled_entry = shuffled_entries.into_iter().next().unwrap();
        assert_eq!(shuffled_entry.id, entry_id);
        assert_eq!(shuffled_entry.sequence, expected_sequence);
        assert_eq!(shuffled_entry.reactivity(), expected_reactivity);
    }

    #[test]
    fn shuffled_db_without_offset() {
        const SEED: u64 = 9;
        const SEQUENCE_LEN: usize = 1553;
        const BLOCK_SIZE: usize = 13;
        const SHUFFLE_ITERATIONS: usize = 3;
        // 1 chunk for the remainder
        const EXPECTED_CHUNKS: usize = SEQUENCE_LEN / BLOCK_SIZE + 1;
        const EXPECTED_SHUFFLED_INDICES: [usize; EXPECTED_CHUNKS] = [
            75, 109, 64, 94, 116, 12, 38, 30, 91, 40, 66, 35, 15, 53, 60, 48, 119, 77, 71, 23, 61,
            68, 99, 9, 16, 93, 83, 115, 84, 62, 26, 37, 107, 88, 117, 95, 6, 56, 113, 19, 20, 70,
            87, 98, 43, 112, 101, 108, 4, 106, 114, 78, 104, 76, 81, 72, 17, 13, 49, 100, 21, 67,
            28, 46, 8, 27, 90, 118, 5, 29, 85, 92, 54, 73, 44, 96, 63, 58, 89, 55, 14, 97, 18, 32,
            103, 86, 80, 31, 36, 69, 3, 59, 65, 50, 0, 110, 7, 102, 22, 42, 82, 24, 10, 111, 57,
            25, 2, 39, 1, 45, 105, 11, 74, 52, 33, 34, 79, 47, 41, 51,
        ];

        let mut db = db_file::Reader::new(File::open("test_data/test.db").unwrap()).unwrap();
        let entry = db.entries().next().unwrap().unwrap();

        // Check if we are on the same page
        let mut rng = SmallRng::seed_from_u64(SEED);
        assert_eq!(rng.gen_range(0..(SEQUENCE_LEN % BLOCK_SIZE)), 0);
        assert_eq!(entry.sequence.len(), SEQUENCE_LEN);
        assert_eq!(entry.sequence.chunks(BLOCK_SIZE).count(), EXPECTED_CHUNKS);
        let mut indices = Vec::new();
        super::resize_indices(&mut indices, EXPECTED_CHUNKS);
        std::iter::repeat(())
            .take(SHUFFLE_ITERATIONS)
            .for_each(|()| indices.shuffle(&mut rng));
        assert_eq!(indices, EXPECTED_SHUFFLED_INDICES);
        drop(indices);

        let expected_sequence: Vec<_> = EXPECTED_SHUFFLED_INDICES
            .iter()
            .copied()
            .flat_map(|chunk_index| {
                get_chunk_without_offset(chunk_index, BLOCK_SIZE, &entry.sequence)
            })
            .copied()
            .collect();

        let expected_reactivity: Vec<_> = EXPECTED_SHUFFLED_INDICES
            .iter()
            .copied()
            .flat_map(|chunk_index| {
                get_chunk_without_offset(chunk_index, BLOCK_SIZE, &entry.reactivity)
            })
            .copied()
            .collect();

        let entry_id = entry.id.clone();

        let shuffled_entries = make_shuffled_db_inner(
            &[entry],
            BLOCK_SIZE,
            SHUFFLE_ITERATIONS,
            SmallRng::seed_from_u64(SEED),
        );

        assert_eq!(shuffled_entries.len(), 1);
        let shuffled_entry = shuffled_entries.into_iter().next().unwrap();
        assert_eq!(shuffled_entry.id, entry_id);
        assert_eq!(shuffled_entry.sequence, expected_sequence);
        assert_eq!(shuffled_entry.reactivity(), expected_reactivity);
    }

    #[test]
    fn chunks_with_zero_offset_no_remainder() {
        assert_eq!(
            get_random_offset_and_chunks(30, 5, StepRng::new(0, 0)),
            (0, 6)
        );
    }

    #[test]
    fn chunks_with_zero_offset_with_remainder() {
        let rng = StepRng::new(0, 0);
        assert_eq!(rng.clone().gen_range(0..3), 0);
        assert_eq!(get_random_offset_and_chunks(33, 5, rng), (0, 7));
    }

    #[test]
    fn chunks_with_offset_with_remainder() {
        let rng = SmallRng::seed_from_u64(0);
        assert_eq!(rng.clone().gen_range(0..3), 1);
        assert_eq!(get_random_offset_and_chunks(33, 5, rng), (1, 8));
    }

    #[test]
    fn resize_indices() {
        let mut indices = Vec::new();
        super::resize_indices(&mut indices, 6);
        assert_eq!(indices.len(), 6);
        assert!(indices.iter().copied().enumerate().all(|(a, b)| a == b));

        indices.fill(9999);
        super::resize_indices(&mut indices, 24);
        assert_eq!(indices.len(), 24);
        assert!(indices.iter().copied().enumerate().all(|(a, b)| a == b));

        indices.fill(9999);
        super::resize_indices(&mut indices, 8);
        assert_eq!(indices.len(), 8);
        assert!(indices.iter().copied().enumerate().all(|(a, b)| a == b));
    }

    #[test]
    fn chunk_with_offset() {
        let data: [u32; 13] = std::array::from_fn(|index| index as u32);
        assert_eq!(get_chunk_with_offset(0, 3, 5, &data), [0, 1, 2]);
        assert_eq!(get_chunk_with_offset(1, 3, 5, &data), [3, 4, 5, 6, 7]);
        assert_eq!(get_chunk_with_offset(2, 3, 5, &data), [8, 9, 10, 11, 12]);

        assert_eq!(get_chunk_with_offset(0, 15, 5, &data), [] as [u32; 0]);
        assert_eq!(get_chunk_with_offset(1, 15, 5, &data), [] as [u32; 0]);
    }

    #[test]
    fn chunk_without_offset() {
        let data: [u32; 9] = std::array::from_fn(|index| index as u32);
        assert_eq!(get_chunk_without_offset(0, 3, &data), [0, 1, 2]);
        assert_eq!(get_chunk_without_offset(1, 3, &data), [3, 4, 5]);
        assert_eq!(get_chunk_without_offset(2, 3, &data), [6, 7, 8]);
    }

    #[test]
    fn extreme_distribution_from_mean_and_variance() {
        let dist =
            ExtremeDistribution::from_mean_and_variance(1.508101930862146, 3.224070771022524);
        assert!((dist.location - 0.7).abs() < 0.00001);
        assert!((dist.scale - 1.4).abs() < 0.00001);
    }

    #[test]
    fn extreme_distribution_from_sample() {
        const DATA: [f64; 12] = [1., 2., 3., 4., 4.5, 5., 5.5, 6., 7., 8., 9., 10.];
        const MEAN: f64 = 5.416666666666667;
        const VARIANCE: f64 = 7.583333333333334;

        let dist = ExtremeDistribution::from_sample(&DATA);
        let expected_dist = ExtremeDistribution::from_mean_and_variance(MEAN, VARIANCE);
        assert!((dist.location - expected_dist.location).abs() < 0.000001);
        assert!((dist.scale - expected_dist.scale).abs() < 0.000001);
    }

    #[test]
    fn extreme_distribution_cdf() {
        let dist = ExtremeDistribution {
            location: 0.7,
            scale: 1.4,
        };

        assert!((dist.cdf(4.5f64) - 0.9358947464960762).abs() < 0.00000001);
    }

    #[test]
    fn extreme_distribution_p_value() {
        let dist = ExtremeDistribution {
            location: 0.7,
            scale: 1.4,
        };

        assert!((dist.p_value(8f64) - 0.0054235557278387025).abs() < 0.00000001);
    }
}
