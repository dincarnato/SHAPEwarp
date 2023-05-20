use std::{iter, ops::Range, slice};

use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, Rng};

use crate::{
    aligner::{AlignedSequenceRef, BaseOrGap},
    gapped_reactivity::{GappedReactivity, GappedReactivityLike},
    gapped_sequence::{sequence_cstring_from_iter, GappedSequence, GappedSequenceLike},
    Base, Sequence,
};

#[derive(Debug, Clone)]
pub(crate) struct GappedData<'a, T> {
    query_sequence: Sequence<'a>,
    query_reactivity: &'a [T],
    query_alignment: AlignedSequenceRef<'a>,
    target_sequence: Sequence<'a>,
    target_reactivity: &'a [T],
    target_alignment: AlignedSequenceRef<'a>,
}

impl<'a, T> GappedData<'a, T> {
    pub(crate) fn new_unchecked(
        query_sequence: GappedSequence<'a>,
        query_reactivity: GappedReactivity<'a, T>,
        target_sequence: GappedSequence<'a>,
        target_reactivity: GappedReactivity<'a, T>,
    ) -> Self {
        Self {
            query_sequence: query_sequence.sequence,
            query_reactivity: query_reactivity.reactivity,
            query_alignment: query_sequence.alignment,
            target_sequence: target_sequence.sequence,
            target_reactivity: target_reactivity.reactivity,
            target_alignment: target_sequence.alignment,
        }
    }

    pub(crate) fn shuffled<'b, R>(
        self,
        block_size: u16,
        indices_buffer: &'b mut Vec<usize>,
        rng: &mut R,
    ) -> Shuffled<'a, 'b, T>
    where
        R: Rng + ?Sized,
    {
        indices_buffer.clear();
        let Self {
            query_sequence,
            query_reactivity,
            query_alignment,
            target_sequence,
            target_reactivity,
            target_alignment,
        } = self;

        let block_size_usize = usize::from(block_size);
        // Just one block size plus leading and trailing base
        let len = query_alignment.0.len();
        let different_block = if len <= block_size_usize + 2 {
            indices_buffer.push(0);
            None
        } else {
            let mut blocks = len / block_size_usize;
            let different_block_len = u16::try_from(len % block_size_usize).unwrap();
            let different_block = (different_block_len != 0).then(|| {
                blocks += 1;
                let index = Uniform::new(0, blocks).sample(rng);
                DifferentBlock {
                    index,
                    len: different_block_len,
                }
            });

            indices_buffer.extend(0..blocks);
            indices_buffer.shuffle(rng);
            different_block
        };

        Shuffled {
            query_sequence,
            query_reactivity,
            query_alignment,
            target_sequence,
            target_reactivity,
            target_alignment,
            indices: indices_buffer,
            different_block,
            block_size,
        }
    }
}

pub(crate) struct Shuffled<'a, 'b, R> {
    query_sequence: Sequence<'a>,
    query_reactivity: &'a [R],
    query_alignment: AlignedSequenceRef<'a>,
    target_sequence: Sequence<'a>,
    target_reactivity: &'a [R],
    target_alignment: AlignedSequenceRef<'a>,
    indices: &'b [usize],
    different_block: Option<DifferentBlock>,
    block_size: u16,
}

impl<'a, 'b, R> Shuffled<'a, 'b, R> {
    #[inline]
    pub(crate) fn query(&self) -> ShuffledSequence<'_, R> {
        ShuffledSequence {
            sequence: &self.query_sequence,
            reactivity: self.query_reactivity,
            alignment: self.query_alignment,
            indices: self.indices,
            block_size: self.block_size,
            different_block: &self.different_block,
        }
    }

    #[inline]
    pub(crate) fn target(&self) -> ShuffledSequence<'_, R> {
        ShuffledSequence {
            sequence: &self.target_sequence,
            reactivity: self.target_reactivity,
            alignment: self.target_alignment,
            indices: self.indices,
            block_size: self.block_size,
            different_block: &self.different_block,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DifferentBlock {
    index: usize,
    len: u16,
}

#[derive(Debug)]
pub(crate) struct ShuffledSequence<'a, R> {
    sequence: &'a Sequence<'a>,
    reactivity: &'a [R],
    alignment: AlignedSequenceRef<'a>,
    pub(crate) indices: &'a [usize],
    block_size: u16,
    different_block: &'a Option<DifferentBlock>,
}

impl<'a, R> ShuffledSequence<'a, R> {
    #[inline]
    pub(crate) fn sequence(&self) -> ShuffledSequenceData<'a, Base> {
        let &Self {
            sequence,
            indices,
            alignment,
            different_block,
            block_size,
            ..
        } = self;

        ShuffledSequenceData {
            data: sequence.bases,
            alignment,
            indices,
            different_block,
            block_size,
        }
    }

    #[inline]
    pub(crate) fn reactivity(&self) -> ShuffledSequenceData<'a, R> {
        let &Self {
            reactivity,
            indices,
            alignment,
            different_block,
            block_size,
            ..
        } = self;

        ShuffledSequenceData {
            data: reactivity,
            alignment,
            indices,
            different_block,
            block_size,
        }
    }

    pub(crate) fn alignment(&self) -> ShuffledAlignment<'a> {
        let &Self {
            indices,
            alignment,
            different_block,
            block_size,
            ..
        } = self;
        let indices = indices.iter();

        ShuffledAlignment {
            alignment,
            indices,
            different_block,
            block_size,
            block: None,
        }
    }
}

fn get_block_range(
    block_index: usize,
    block_size: usize,
    different_block: Option<&DifferentBlock>,
) -> Range<usize> {
    match different_block {
        Some(different_block) if block_index == different_block.index => {
            let start = block_size * block_index;
            let end = start + usize::from(different_block.len);
            start..end
        }
        Some(different_block) if block_index > different_block.index => {
            debug_assert!(block_index > 0);
            let start = block_size * (block_index - 1) + usize::from(different_block.len);
            let end = start + block_size;
            start..end
        }
        Some(_) | None => {
            let start = block_size * block_index;
            let end = start + block_size;
            start..end
        }
    }
}

#[derive(Debug)]
pub(crate) struct ShuffledSequenceData<'a, T> {
    data: &'a [T],
    alignment: AlignedSequenceRef<'a>,
    indices: &'a [usize],
    different_block: &'a Option<DifferentBlock>,
    block_size: u16,
}

impl<'a, T> ShuffledSequenceData<'a, T>
where
    T: Copy,
{
    pub fn into_iter(self) -> ShuffledSequenceDataIter<'a, T> {
        IntoIterator::into_iter(self)
    }
}

impl<'a, T> IntoIterator for ShuffledSequenceData<'a, T>
where
    T: Copy,
{
    type Item = T;
    type IntoIter = ShuffledSequenceDataIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let Self {
            data,
            alignment,
            indices,
            different_block,
            block_size,
        } = self;
        let indices = indices.iter();

        ShuffledSequenceDataIter {
            data,
            alignment,
            indices,
            different_block,
            block_size,
            block: None,
        }
    }
}

impl<'a, T> IntoIterator for &'a ShuffledSequenceData<'a, T>
where
    T: Copy,
{
    type Item = T;
    type IntoIter = ShuffledSequenceDataIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let &ShuffledSequenceData {
            data,
            alignment,
            indices,
            different_block,
            block_size,
        } = self;
        let indices = indices.iter();

        ShuffledSequenceDataIter {
            data,
            alignment,
            indices,
            different_block,
            block_size,
            block: None,
        }
    }
}

pub(crate) struct ShuffledSequenceDataIter<'a, T> {
    data: &'a [T],
    alignment: AlignedSequenceRef<'a>,
    indices: slice::Iter<'a, usize>,
    different_block: &'a Option<DifferentBlock>,
    block_size: u16,
    block: Option<SequenceDataBlock<'a, T>>,
}

type SequenceDataBlockAlignment<'a> =
    iter::Filter<iter::Copied<slice::Iter<'a, BaseOrGap>>, fn(&BaseOrGap) -> bool>;

struct SequenceDataBlock<'a, T> {
    data: slice::Iter<'a, T>,
    alignment: SequenceDataBlockAlignment<'a>,
}

impl<T> Iterator for ShuffledSequenceDataIter<'_, T>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.block
            .as_mut()
            .and_then(|block| block.alignment.next().map(|_| block.data.next().unwrap()))
            .copied()
            .or_else(|| loop {
                match self.indices.next() {
                    Some(&index) => {
                        let block_range = get_block_range(
                            index,
                            self.block_size.into(),
                            self.different_block.as_ref(),
                        );

                        let sequence_offset = self.alignment.0[..block_range.start]
                            .iter()
                            .filter(|base_or_gap| base_or_gap.is_base())
                            .count();

                        let mut data_block = self.data[sequence_offset..].iter();
                        let mut alignment = self.alignment.0[block_range].iter().copied().filter(
                            (|base_or_gap| base_or_gap.is_base()) as fn(&BaseOrGap) -> bool,
                        );

                        match alignment.next() {
                            Some(_) => {
                                let base = *data_block.next().unwrap();

                                self.block = Some(SequenceDataBlock {
                                    data: data_block,
                                    alignment,
                                });

                                break Some(base);
                            }
                            None => continue,
                        }
                    }
                    None => break None,
                }
            })
    }
}

pub(crate) struct ShuffledAlignment<'a> {
    alignment: AlignedSequenceRef<'a>,
    indices: slice::Iter<'a, usize>,
    different_block: &'a Option<DifferentBlock>,
    block: Option<slice::Iter<'a, BaseOrGap>>,
    block_size: u16,
}

impl Iterator for ShuffledAlignment<'_> {
    type Item = BaseOrGap;

    fn next(&mut self) -> Option<Self::Item> {
        self.block
            .as_mut()
            .and_then(|block| block.next())
            .copied()
            .or_else(|| {
                self.indices.next().map(|&index| {
                    let block_range = get_block_range(
                        index,
                        self.block_size.into(),
                        self.different_block.as_ref(),
                    );

                    let mut block = self.alignment.0[block_range].iter();
                    // A block cannot be empty
                    let base_or_gap = *block.next().unwrap();
                    self.block = Some(block);
                    base_or_gap
                })
            })
    }
}

impl<R> GappedSequenceLike for ShuffledSequence<'_, R> {
    fn to_cstring(&self, molecule: Option<crate::Molecule>) -> std::ffi::CString {
        // Rough estimation
        let estimated_len = self.alignment.0.len().max(self.sequence.bases.len()) + 1;
        let molecule = molecule.unwrap_or(self.sequence.molecule);

        sequence_cstring_from_iter(
            self.sequence().into_iter(),
            self.alignment(),
            molecule,
            estimated_len,
        )
    }
}

impl<R: Copy> GappedReactivityLike<R> for ShuffledSequence<'_, R> {
    type AlignmentIter<'a> = ShuffledAlignment<'a>
    where
        Self: 'a;

    type ReactivityIter<'a> = ShuffledSequenceDataIter<'a, R>
    where
        Self: 'a;

    #[inline]
    fn alignment(&self) -> Self::AlignmentIter<'_> {
        self.alignment()
    }

    #[inline]
    fn reactivity(&self) -> Self::ReactivityIter<'_> {
        self.reactivity().into_iter()
    }
}
