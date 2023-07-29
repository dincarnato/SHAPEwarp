use std::{ops::Range, slice};

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
        query_reactivity: &GappedReactivity<'a, T>,
        target_sequence: GappedSequence<'a>,
        target_reactivity: &GappedReactivity<'a, T>,
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
    ) -> Shuffled<'a, 'b, T, ()>
    where
        R: Rng + ?Sized,
    {
        self.shuffled_inner(block_size, indices_buffer, |_, _, _| (), rng)
    }

    pub(crate) fn shuffled_in_blocks<'b, R>(
        self,
        block_size: u16,
        indices_buffer: &'b mut Vec<usize>,
        block_indices_buffer: &'b mut Vec<usize>,
        rng: &mut R,
    ) -> Shuffled<'a, 'b, T, &'b [usize]>
    where
        R: Rng + ?Sized,
    {
        self.shuffled_inner(
            block_size,
            indices_buffer,
            move |len, different_block, rng| {
                shuffle_inner_indices_blocks(
                    usize::from(block_size),
                    block_indices_buffer,
                    len,
                    different_block,
                    rng,
                )
            },
            rng,
        )
    }

    fn shuffled_inner<'b, F, B, R>(
        self,
        block_size: u16,
        indices_buffer: &'b mut Vec<usize>,
        blocks_handler: F,
        rng: &mut R,
    ) -> Shuffled<'a, 'b, T, B>
    where
        F: FnOnce(usize, &Option<DifferentBlock>, &mut R) -> B,
        R: Rng + ?Sized,
    {
        let Self {
            query_sequence,
            query_reactivity,
            query_alignment,
            target_sequence,
            target_reactivity,
            target_alignment,
        } = self;

        let len = query_alignment.0.len();
        let different_block = shuffle_indices(usize::from(block_size), len, indices_buffer, rng);
        let block_indices = blocks_handler(len, &different_block, rng);

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
            block_indices,
        }
    }
}

fn shuffle_indices<R>(
    block_size: usize,
    data_len: usize,
    indices_buffer: &mut Vec<usize>,
    rng: &mut R,
) -> Option<DifferentBlock>
where
    R: Rng + ?Sized,
{
    indices_buffer.clear();
    // Just one block size plus leading and trailing base
    if data_len <= block_size + 2 {
        indices_buffer.push(0);
        None
    } else {
        let mut blocks = data_len / block_size;
        let different_block_len = u16::try_from(data_len % block_size).unwrap();
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
    }
}

fn shuffle_inner_indices_blocks<'a, R>(
    block_size: usize,
    block_indices_buffer: &'a mut Vec<usize>,
    len: usize,
    different_block: &Option<DifferentBlock>,
    rng: &mut R,
) -> &'a [usize]
where
    R: Rng + ?Sized,
{
    block_indices_buffer.resize(len, 0);

    match different_block {
        Some(different_block) => {
            let before_different_block = different_block.index * block_size;
            let (first_part, rest) = block_indices_buffer.split_at_mut(before_different_block);
            let (different_block, second_part) =
                rest.split_at_mut(usize::from(different_block.len));

            different_block
                .iter_mut()
                .enumerate()
                .for_each(|(index, v)| *v = index);
            different_block.shuffle(rng);

            for part in [first_part, second_part] {
                part.chunks_mut(block_size).for_each(|block| {
                    block
                        .iter_mut()
                        .enumerate()
                        .for_each(|(index, v)| *v = index);
                    block.shuffle(rng);
                });
            }
        }
        None => block_indices_buffer
            .chunks_mut(block_size)
            .for_each(|block| {
                block
                    .iter_mut()
                    .enumerate()
                    .for_each(|(index, v)| *v = index);
                block.shuffle(rng);
            }),
    }

    block_indices_buffer.as_slice()
}

pub(crate) struct Shuffled<'a, 'b, R, B> {
    query_sequence: Sequence<'a>,
    query_reactivity: &'a [R],
    query_alignment: AlignedSequenceRef<'a>,
    target_sequence: Sequence<'a>,
    target_reactivity: &'a [R],
    target_alignment: AlignedSequenceRef<'a>,
    indices: &'b [usize],
    different_block: Option<DifferentBlock>,
    block_size: u16,
    block_indices: B,
}

impl<'a, 'b, R, B> Shuffled<'a, 'b, R, B>
where
    B: Copy,
{
    #[inline]
    pub(crate) fn query(&self) -> ShuffledSequence<'_, 'b, R, B> {
        ShuffledSequence {
            sequence: &self.query_sequence,
            reactivity: self.query_reactivity,
            alignment: self.query_alignment,
            indices: self.indices,
            block_size: self.block_size,
            different_block: &self.different_block,
            block_indices: self.block_indices,
        }
    }

    #[inline]
    pub(crate) fn target(&self) -> ShuffledSequence<'_, 'b, R, B> {
        ShuffledSequence {
            sequence: &self.target_sequence,
            reactivity: self.target_reactivity,
            alignment: self.target_alignment,
            indices: self.indices,
            block_size: self.block_size,
            different_block: &self.different_block,
            block_indices: self.block_indices,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DifferentBlock {
    index: usize,
    len: u16,
}

#[derive(Debug)]
pub(crate) struct ShuffledSequence<'a, 'b, R, B> {
    sequence: &'a Sequence<'a>,
    reactivity: &'a [R],
    alignment: AlignedSequenceRef<'a>,
    pub(crate) indices: &'b [usize],
    block_size: u16,
    different_block: &'a Option<DifferentBlock>,
    block_indices: B,
}

impl<'a, 'b, R, B> ShuffledSequence<'a, 'b, R, B>
where
    B: Copy,
{
    #[inline]
    pub(crate) fn sequence(&self) -> ShuffledSequenceData<'a, 'b, Base, B> {
        let &Self {
            sequence,
            indices,
            alignment,
            different_block,
            block_size,
            block_indices,
            ..
        } = self;

        ShuffledSequenceData {
            data: sequence.bases,
            alignment,
            indices,
            different_block,
            block_size,
            block_indices,
        }
    }

    #[inline]
    pub(crate) fn reactivity(&self) -> ShuffledSequenceData<'a, 'b, R, B> {
        let &Self {
            reactivity,
            indices,
            alignment,
            different_block,
            block_size,
            block_indices,
            ..
        } = self;

        ShuffledSequenceData {
            data: reactivity,
            alignment,
            indices,
            different_block,
            block_size,
            block_indices,
        }
    }

    pub(crate) fn alignment(&self) -> ShuffledAlignment<'a, 'b> {
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
pub(crate) struct ShuffledSequenceData<'a, 'b, T, B> {
    data: &'a [T],
    alignment: AlignedSequenceRef<'a>,
    indices: &'b [usize],
    different_block: &'a Option<DifferentBlock>,
    block_size: u16,
    block_indices: B,
}

impl<'a, 'b, T, B> ShuffledSequenceData<'a, 'b, T, B>
where
    T: Copy,
    ShuffledSequenceDataIter<'a, 'b, T, B>: Iterator<Item = T>,
    B: InternalBlockIter<'a, 'b, T>,
{
    pub fn into_iter(self) -> ShuffledSequenceDataIter<'a, 'b, T, B> {
        IntoIterator::into_iter(self)
    }
}

impl<'a, 'b, T, B> IntoIterator for ShuffledSequenceData<'a, 'b, T, B>
where
    T: Copy,
    ShuffledSequenceDataIter<'a, 'b, T, B>: Iterator<Item = T>,
    B: InternalBlockIter<'a, 'b, T>,
{
    type Item = T;
    type IntoIter = ShuffledSequenceDataIter<'a, 'b, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        let Self {
            data,
            alignment,
            indices,
            different_block,
            block_size,
            block_indices,
        } = self;
        let indices = indices.iter();

        ShuffledSequenceDataIter {
            data,
            alignment,
            indices,
            different_block,
            block_size,
            block: None,
            block_indices,
        }
    }
}

impl<'a, 'b, T, B> IntoIterator for &'a ShuffledSequenceData<'a, 'b, T, B>
where
    T: Copy,
    ShuffledSequenceDataIter<'a, 'b, T, B>: Iterator<Item = T>,
    B: InternalBlockIter<'a, 'b, T> + Copy,
{
    type Item = T;
    type IntoIter = ShuffledSequenceDataIter<'a, 'b, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        let &ShuffledSequenceData {
            data,
            alignment,
            indices,
            different_block,
            block_size,
            block_indices,
        } = self;
        let indices = indices.iter();

        ShuffledSequenceDataIter {
            data,
            alignment,
            indices,
            different_block,
            block_size,
            block: None,
            block_indices,
        }
    }
}

pub(crate) struct ShuffledSequenceDataIter<'a, 'b, T, B: InternalBlockIter<'a, 'b, T>> {
    data: &'a [T],
    alignment: AlignedSequenceRef<'a>,
    indices: slice::Iter<'b, usize>,
    different_block: &'a Option<DifferentBlock>,
    block_size: u16,
    block: Option<B::Iterator>,
    block_indices: B,
}

pub(crate) trait InternalBlockIter<'a, 'b, T: 'a>: Sized {
    type Iterator: Sized + Iterator<Item = T>;
}

impl<'a, 'b, T: 'a + Copy> InternalBlockIter<'a, 'b, T> for () {
    type Iterator = SimpleBlockIterator<'a, T>;
}

impl<'a, 'b, T: 'a + Copy> InternalBlockIter<'a, 'b, T> for &'b [usize] {
    type Iterator = RandomAccessBlockIterator<'a, 'b, T>;
}

pub(crate) struct SimpleBlockIterator<'a, T> {
    data: slice::Iter<'a, T>,
    align: slice::Iter<'a, BaseOrGap>,
}

pub(crate) struct RandomAccessBlockIterator<'a, 'b, T> {
    data: &'a [T],
    align: &'a [BaseOrGap],
    indices: slice::Iter<'b, usize>,
}

impl<'a, T: 'a + Copy> Iterator for SimpleBlockIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        for align in self.align.by_ref() {
            if align.is_base() {
                return Some(*self.data.next().unwrap());
            }
        }

        None
    }
}

impl<'a, 'b, T: 'a + Copy> Iterator for RandomAccessBlockIterator<'a, 'b, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        for &index in self.indices.by_ref() {
            let (base_or_gap, base_or_gaps_before) = self.align[..=index].split_last().unwrap();

            if base_or_gap.is_base() {
                let gaps_before = base_or_gaps_before.iter().filter(|b| b.is_gap()).count();
                return Some(self.data[index - gaps_before]);
            }
        }

        None
    }
}

impl<'a, 'b, T, B> ShuffledSequenceDataIter<'a, 'b, T, B>
where
    T: Copy,
    B: InternalBlockIter<'a, 'b, T>,
{
    fn generic_next<F>(&mut self, get_block_iterator: F) -> Option<T>
    where
        F: Fn(&Self, &'a [T], &'a [BaseOrGap], Range<usize>) -> B::Iterator,
    {
        self.block
            .as_mut()
            .and_then(Iterator::next)
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

                        let mut block_iterator = get_block_iterator(
                            self,
                            &self.data[sequence_offset..],
                            &self.alignment.0[block_range.clone()],
                            block_range,
                        );

                        let Some(base) = block_iterator.next() else {
                            continue;
                        };

                        self.block = Some(block_iterator);
                        break Some(base);
                    }
                    None => break None,
                }
            })
    }
}

impl<T> Iterator for ShuffledSequenceDataIter<'_, '_, T, ()>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.generic_next(|_this, data_block, alignment_block, _block_range| {
            let data = data_block.iter();
            let align = alignment_block.iter();
            SimpleBlockIterator { data, align }
        })
    }
}

impl<'b, T> Iterator for ShuffledSequenceDataIter<'_, 'b, T, &'b [usize]>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.generic_next(|this, data, align, block_range| RandomAccessBlockIterator {
            data,
            align,
            indices: this.block_indices[block_range].iter(),
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IndexAccessIter<'a, T, I> {
    items: &'a [T],
    indices: I,
}

impl<'a, T, I> Iterator for IndexAccessIter<'a, T, I>
where
    I: Iterator<Item = usize>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|index| &self.items[index])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

pub(crate) struct ShuffledAlignment<'a, 'b> {
    alignment: AlignedSequenceRef<'a>,
    indices: slice::Iter<'b, usize>,
    different_block: &'a Option<DifferentBlock>,
    block: Option<slice::Iter<'a, BaseOrGap>>,
    block_size: u16,
}

impl Iterator for ShuffledAlignment<'_, '_> {
    type Item = BaseOrGap;

    fn next(&mut self) -> Option<Self::Item> {
        self.block
            .as_mut()
            .and_then(Iterator::next)
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

impl<'a, 'b, R, B> GappedSequenceLike for ShuffledSequence<'a, 'b, R, B>
where
    ShuffledSequenceData<'a, 'b, Base, B>:
        IntoIterator<IntoIter = ShuffledSequenceDataIter<'a, 'b, Base, B>, Item = Base>,
    ShuffledSequenceDataIter<'a, 'b, Base, B>: Iterator<Item = Base>,
    B: InternalBlockIter<'a, 'b, Base> + Copy,
{
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

impl<'a, 'b, R, B> GappedReactivityLike<R> for ShuffledSequence<'a, 'b, R, B>
where
    R: Copy,
    ShuffledSequenceDataIter<'a, 'b, R, B>: Iterator<Item = R>,
    B: InternalBlockIter<'a, 'b, R> + Copy,
{
    type AlignmentIter<'c> = ShuffledAlignment<'a, 'b>
    where
        Self: 'c;

    type ReactivityIter<'c> = ShuffledSequenceDataIter<'a, 'b, R, B>
    where
        Self: 'c;

    #[inline]
    fn alignment(&self) -> Self::AlignmentIter<'_> {
        ShuffledSequence::alignment(self)
    }

    #[inline]
    fn reactivity(&self) -> Self::ReactivityIter<'_> {
        ShuffledSequence::reactivity(self).into_iter()
    }
}

#[cfg(test)]
mod tests {
    use std::{array, iter, ops::Range};

    use rand::{seq::SliceRandom, thread_rng};

    use crate::{
        aligner::{AlignedSequence, AlignedSequenceRef, BaseOrGap},
        gapped_data::RandomAccessBlockIterator,
    };

    use super::{
        shuffle_indices, shuffle_inner_indices_blocks, DifferentBlock, ShuffledSequenceData,
        SimpleBlockIterator,
    };

    #[test]
    fn iter_simple_block_iterator() {
        let data = [0, 1, 2, 3];
        let align = [
            BaseOrGap::Gap,
            BaseOrGap::Base,
            BaseOrGap::Base,
            BaseOrGap::Gap,
            BaseOrGap::Gap,
            BaseOrGap::Base,
            BaseOrGap::Base,
            BaseOrGap::Gap,
        ];

        let result = SimpleBlockIterator {
            data: data.iter(),
            align: align.iter(),
        }
        .collect::<Vec<_>>();

        assert_eq!(result, data);
    }

    #[test]
    fn iter_random_access_block_iterator() {
        let data = [0, 1, 2, 3];
        let align = [
            BaseOrGap::Gap,
            BaseOrGap::Base,
            BaseOrGap::Base,
            BaseOrGap::Gap,
            BaseOrGap::Gap,
            BaseOrGap::Base,
            BaseOrGap::Base,
            BaseOrGap::Gap,
        ];
        let indices = [2, 4, 6, 5, 0, 7, 0, 3, 1];

        let result = RandomAccessBlockIterator {
            data: &data,
            align: &align,
            indices: indices.iter(),
        }
        .collect::<Vec<_>>();

        assert_eq!(result, [1, 3, 2, 0]);
    }

    fn shuffled_inner_impl(block_size: u16, data_len: u16) {
        let data = (0..data_len).collect::<Vec<_>>();
        let mut indices = Vec::new();
        let mut block_indices = Vec::new();
        let mut rng = thread_rng();
        let alignment = AlignedSequence(vec![BaseOrGap::Base; data.len()]);
        let alignment = alignment.to_ref();

        let different_block =
            shuffle_indices(usize::from(block_size), data.len(), &mut indices, &mut rng);
        assert_eq!(
            different_block.is_some(),
            data.len() % usize::from(block_size) != 0,
        );

        let block_indices = shuffle_inner_indices_blocks(
            usize::from(block_size),
            &mut block_indices,
            data.len(),
            &different_block,
            &mut rng,
        );

        let shuffled_sequence_data = ShuffledSequenceData {
            data: &data,
            alignment,
            indices: &indices,
            different_block: &different_block,
            block_size,
            block_indices,
        }
        .into_iter()
        .collect::<Vec<_>>();

        let block_size = usize::from(block_size);
        let is_standard_chunk = |chunk: &[_]| {
            assert_eq!(chunk.len(), block_size);
            let min = chunk.iter().min().unwrap();
            let max = chunk.iter().max().unwrap();
            usize::from(max - min) == block_size - 1
        };
        match different_block {
            Some(different_block) => {
                let different_block_index = indices
                    .iter()
                    .copied()
                    .position(|index| index == different_block.index)
                    .unwrap();

                let before_different_block = different_block_index * block_size;
                let after_different_block =
                    before_different_block + usize::from(different_block.len);
                assert!(shuffled_sequence_data[..before_different_block]
                    .chunks(block_size)
                    .all(is_standard_chunk));
                assert!(shuffled_sequence_data[after_different_block..]
                    .chunks(block_size)
                    .all(is_standard_chunk));

                let different_block_data =
                    &shuffled_sequence_data[before_different_block..after_different_block];
                let min = different_block_data.iter().min().unwrap();
                let max = different_block_data.iter().max().unwrap();
                assert_eq!(max - min, different_block.len - 1);
            }
            None => {
                assert!(shuffled_sequence_data
                    .chunks(block_size)
                    .all(is_standard_chunk));
            }
        }
        assert_eq!(
            shuffled_sequence_data
                .iter()
                .copied()
                .map(u16::from)
                .sum::<u16>(),
            u16::try_from(data.len()).unwrap() * (u16::try_from(data.len()).unwrap() - 1) / 2
        );
    }

    #[test]
    fn shuffled_inner_block_no_different_block() {
        shuffled_inner_impl(5, 100);
    }

    #[test]
    fn shuffled_inner_block_with_different_block() {
        shuffled_inner_impl(7, 100);
    }

    #[test]
    fn get_block_range() {
        assert_eq!(super::get_block_range(4, 7, None), 28..35);
        assert_eq!(
            super::get_block_range(3, 7, Some(&DifferentBlock { index: 5, len: 11 })),
            21..28,
        );
        assert_eq!(
            super::get_block_range(5, 7, Some(&DifferentBlock { index: 5, len: 11 })),
            35..46,
        );
        assert_eq!(
            super::get_block_range(6, 7, Some(&DifferentBlock { index: 5, len: 11 })),
            46..53,
        );
        assert_eq!(
            super::get_block_range(7, 7, Some(&DifferentBlock { index: 5, len: 11 })),
            53..60,
        );
    }

    #[test]
    fn index_access_iter() {
        let data = array::from_fn::<_, 100, _>(|index| index);
        let mut indices = data;
        indices.shuffle(&mut thread_rng());
        assert!(super::IndexAccessIter {
            items: &data,
            indices: indices.iter().copied(),
        }
        .eq(indices.iter().copied().map(|index| &data[index])));
    }

    fn shuffled_gapped_inner_impl(block_size: u16, data_len: u16, gaps: &[Range<u16>]) {
        assert!(gaps.windows(2).all(|win| win[0].end < win[1].start));
        assert!(gaps.last().map_or(true, |gap| gap.end <= data_len));

        let data = (0..data_len).collect::<Vec<_>>();
        let mut indices = Vec::new();
        let mut block_indices = Vec::new();
        let mut rng = thread_rng();
        let (mut alignment, used_bases) =
            gaps.iter()
                .fold((Vec::new(), 0), |(mut alignment, used_bases), gap| {
                    let new_bases = gap.start - used_bases;
                    alignment.extend(
                        iter::repeat(BaseOrGap::Base)
                            .take(new_bases.into())
                            .chain(iter::repeat(BaseOrGap::Gap).take((gap.end - gap.start).into())),
                    );

                    (alignment, used_bases + new_bases)
                });
        assert!(usize::from(used_bases) <= data.len());
        alignment.resize(
            alignment.len() - usize::from(used_bases) + data.len(),
            BaseOrGap::Base,
        );
        let alignment = AlignedSequence(alignment);
        let alignment = alignment.to_ref();
        let alignment_len = alignment.0.len();

        let different_block = shuffle_indices(
            usize::from(block_size),
            alignment_len,
            &mut indices,
            &mut rng,
        );
        assert_eq!(
            different_block.is_some(),
            alignment_len % usize::from(block_size) != 0,
        );

        let block_indices = shuffle_inner_indices_blocks(
            usize::from(block_size),
            &mut block_indices,
            alignment_len,
            &different_block,
            &mut rng,
        );
        assert_eq!(block_indices.len(), alignment_len);

        let shuffled_sequence_data = ShuffledSequenceData {
            data: &data,
            alignment,
            indices: &indices,
            different_block: &different_block,
            block_size,
            block_indices,
        }
        .into_iter()
        .collect::<Vec<_>>();

        check_shuffled_data(
            usize::from(block_size),
            different_block,
            &indices,
            alignment,
            &shuffled_sequence_data,
            data_len,
        );
    }

    fn check_shuffled_data(
        block_size: usize,
        different_block: Option<DifferentBlock>,
        indices: &[usize],
        alignment: AlignedSequenceRef<'_>,
        shuffled_sequence_data: &[u16],
        data_len: u16,
    ) {
        let is_standard_chunk = |chunk: &[_]| {
            assert!(chunk.len() <= block_size);
            let min = chunk.iter().min().unwrap();
            let max = chunk.iter().max().unwrap();
            usize::from(max - min) == chunk.len() - 1
        };
        let chunked_alignment: Vec<_> = match different_block {
            Some(different_block) => {
                let different_block_index = indices
                    .iter()
                    .copied()
                    .position(|index| index == different_block.index)
                    .unwrap();

                let before_different_block = different_block_index * block_size;
                let after_different_block =
                    before_different_block + usize::from(different_block.len);

                alignment
                    .0
                    .chunks_exact(block_size)
                    .take(different_block.index)
                    .chain(iter::once(
                        &alignment.0[before_different_block..after_different_block],
                    ))
                    .chain(alignment.0[after_different_block..].chunks_exact(block_size))
                    .collect::<Vec<_>>()
            }
            None => alignment.0.chunks_exact(block_size).collect(),
        };

        let mut shuffled_sequence_data_iter = shuffled_sequence_data.iter();
        assert!(indices.iter().copied().all(|block_index| {
            let bases = chunked_alignment[block_index]
                .iter()
                .filter(|b| b.is_base())
                .count();

            let shuffled_data_block: Vec<_> = shuffled_sequence_data_iter
                .by_ref()
                .take(bases)
                .copied()
                .collect();

            shuffled_data_block.len() == bases && is_standard_chunk(&shuffled_data_block)
        }));

        assert_eq!(
            shuffled_sequence_data
                .iter()
                .copied()
                .map(u16::from)
                .sum::<u16>(),
            data_len * (data_len - 1) / 2
        );
    }

    #[test]
    fn shuffled_gapped_inner_block_no_different_block_no_gaps() {
        const DATA_LEN: u16 = 20;
        const BLOCK_SIZE: u16 = 5;

        shuffled_gapped_inner_impl(BLOCK_SIZE, DATA_LEN, &[]);
    }

    #[test]
    fn shuffled_gapped_inner_block_no_different_block() {
        const GAPS: [Range<u16>; 2] = [3..5, 11..12];
        const DATA_LEN: u16 = 17;
        const BLOCK_SIZE: u16 = 5;

        assert_eq!(
            (GAPS.into_iter().map(|gap| gap.end - gap.start).sum::<u16>() + DATA_LEN) % BLOCK_SIZE,
            0,
        );
        shuffled_gapped_inner_impl(BLOCK_SIZE, DATA_LEN, &GAPS);
    }
}
