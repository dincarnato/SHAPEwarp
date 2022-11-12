use std::{iter, mem};

use num_traits::cast;
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

    let sequences = db.len() * shuffle_iterations;
    iter::from_fn(move || {
        let entry = db.choose(&mut rng)?;
        let (offset, chunks) =
            get_random_offset_and_chunks(entry.sequence.len(), block_size, &mut rng);

        resize_indices(&mut chunk_indices, chunks);
        chunk_indices.shuffle(&mut rng);

        get_shuffled_entry(&chunk_indices, entry, offset, block_size)
    })
    .take(sequences)
    .collect()
}

#[inline]
fn get_shuffled_entry(
    chunk_indices: &[usize],
    entry: &Entry,
    offset: usize,
    block_size: usize,
) -> Option<Entry> {
    let mut sequence = Vec::with_capacity(entry.sequence.len());
    let mut reactivity = Vec::with_capacity(entry.reactivity.len());

    match offset {
        0 => {
            for &chunk_index in chunk_indices {
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
            for &chunk_index in chunk_indices {
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

    Some(Entry {
        id: entry.id.clone(),
        sequence,
        reactivity,
    })
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
        T: num_traits::NumCast + Copy,
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
        T: num_traits::NumCast,
    {
        let z = (cast::<_, f64>(value).unwrap() - self.location) / self.scale;
        f64::exp(-f64::exp(-z))
    }

    #[inline]
    pub(crate) fn p_value<T>(&self, value: T) -> f64
    where
        T: num_traits::NumCast,
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
    fn shuffled_entry_with_offset() {
        const SEQUENCE_LEN: usize = 1553;
        const BLOCK_SIZE: usize = 13;
        const OFFSET: usize = 3;
        // 1 chunk for the remainder
        const EXPECTED_CHUNKS: usize =
            1 + (1553 - OFFSET) / BLOCK_SIZE + ((SEQUENCE_LEN - OFFSET) % BLOCK_SIZE != 0) as usize;
        const SHUFFLED_INDICES: [usize; EXPECTED_CHUNKS] = [
            72, 38, 19, 42, 26, 13, 69, 51, 5, 97, 79, 62, 102, 28, 3, 120, 44, 103, 32, 81, 85,
            27, 93, 113, 106, 15, 65, 36, 59, 98, 99, 77, 84, 29, 39, 61, 16, 30, 80, 1, 68, 14,
            78, 90, 95, 118, 41, 10, 116, 91, 37, 70, 0, 92, 46, 18, 9, 2, 63, 57, 110, 40, 66, 49,
            108, 56, 119, 7, 50, 107, 55, 54, 4, 104, 115, 23, 53, 111, 82, 24, 35, 71, 12, 43, 76,
            48, 52, 25, 87, 100, 17, 74, 20, 94, 114, 109, 86, 34, 58, 21, 105, 64, 88, 60, 117,
            22, 31, 89, 73, 11, 101, 75, 112, 96, 83, 47, 67, 8, 45, 33, 6,
        ];

        let mut db = db_file::Reader::new(File::open("test_data/test.db").unwrap()).unwrap();
        let entry = db.entries().next().unwrap().unwrap();

        assert_eq!(entry.sequence.len(), SEQUENCE_LEN);

        let split_sequence: Vec<_> = iter::once(&entry.sequence[..OFFSET])
            .chain(entry.sequence[OFFSET..].chunks(BLOCK_SIZE))
            .collect();
        let expected_sequence: Vec<_> = SHUFFLED_INDICES
            .into_iter()
            .flat_map(|index| split_sequence[index])
            .copied()
            .collect();

        let split_reactivities: Vec<_> = iter::once(&entry.reactivity[..OFFSET])
            .chain(entry.reactivity[OFFSET..].chunks(BLOCK_SIZE))
            .collect();
        let expected_reactivity: Vec<_> = SHUFFLED_INDICES
            .into_iter()
            .flat_map(|index| split_reactivities[index])
            .copied()
            .collect();

        let shuffled_entry =
            get_shuffled_entry(&SHUFFLED_INDICES, &entry, OFFSET, BLOCK_SIZE).unwrap();

        assert_eq!(shuffled_entry.id, entry.id);
        assert_eq!(shuffled_entry.sequence, expected_sequence);
        assert!(shuffled_entry
            .reactivity()
            .iter()
            .copied()
            .zip(expected_reactivity)
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b));
    }

    #[test]
    fn shuffled_entry_without_offset() {
        const SEQUENCE_LEN: usize = 1553;
        const BLOCK_SIZE: usize = 13;
        const EXPECTED_CHUNKS: usize =
            SEQUENCE_LEN / BLOCK_SIZE + (SEQUENCE_LEN % BLOCK_SIZE != 0) as usize;
        const SHUFFLED_INDICES: [usize; EXPECTED_CHUNKS] = [
            72, 38, 19, 42, 26, 13, 69, 51, 5, 97, 79, 62, 102, 28, 3, 44, 103, 32, 81, 85, 27, 93,
            113, 106, 15, 65, 36, 59, 98, 99, 77, 84, 29, 39, 61, 16, 30, 80, 1, 68, 14, 78, 90,
            95, 118, 41, 10, 116, 91, 37, 70, 0, 92, 46, 18, 9, 2, 63, 57, 110, 40, 66, 49, 108,
            56, 119, 7, 50, 107, 55, 54, 4, 104, 115, 23, 53, 111, 82, 24, 35, 71, 12, 43, 76, 48,
            52, 25, 87, 100, 17, 74, 20, 94, 114, 109, 86, 34, 58, 21, 105, 64, 88, 60, 117, 22,
            31, 89, 73, 11, 101, 75, 112, 96, 83, 47, 67, 8, 45, 33, 6,
        ];

        let mut db = db_file::Reader::new(File::open("test_data/test.db").unwrap()).unwrap();
        let entry = db.entries().next().unwrap().unwrap();

        assert_eq!(entry.sequence.len(), SEQUENCE_LEN);

        let split_sequence: Vec<_> = entry.sequence.chunks(BLOCK_SIZE).collect();
        let expected_sequence: Vec<_> = SHUFFLED_INDICES
            .into_iter()
            .flat_map(|index| split_sequence[index])
            .copied()
            .collect();

        let split_reactivities: Vec<_> = entry.reactivity.chunks(BLOCK_SIZE).collect();
        let expected_reactivity: Vec<_> = SHUFFLED_INDICES
            .into_iter()
            .flat_map(|index| split_reactivities[index])
            .copied()
            .collect();

        let shuffled_entry = get_shuffled_entry(&SHUFFLED_INDICES, &entry, 0, BLOCK_SIZE).unwrap();

        assert_eq!(shuffled_entry.id, entry.id);
        assert_eq!(shuffled_entry.sequence, expected_sequence);
        assert!(shuffled_entry
            .reactivity()
            .iter()
            .copied()
            .zip(expected_reactivity)
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b));
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
