use std::{
    ffi::CString,
    fmt::{self, Display},
    ops::{Not, Range},
    slice,
};

use crate::{
    aligner::{AlignedSequence, AlignedSequenceRef, BaseOrGap},
    Base, Molecule, Sequence,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GappedSequence<'a> {
    pub(crate) sequence: Sequence<'a>,
    pub(crate) alignment: AlignedSequenceRef<'a>,
}

pub(crate) trait GappedSequenceLike {
    fn to_cstring(&self, molecule: Option<Molecule>) -> CString;
}

impl<'a> GappedSequence<'a> {
    pub(crate) fn new(sequence: Sequence<'a>, alignment: &'a AlignedSequence) -> Self {
        let alignment = alignment.to_ref();
        Self {
            sequence,
            alignment,
        }
    }

    #[inline]
    pub(crate) fn iter(&self) -> GappedSequenceIter<'_> {
        IntoIterator::into_iter(self)
    }

    pub(crate) fn get(&self, index: Range<usize>) -> Option<GappedSequence<'a>> {
        let start = index.start;
        self.alignment.0.get(index).map(|alignment| {
            let bases_before = self.alignment.0[..start]
                .iter()
                .filter(|base_or_gap| base_or_gap.is_base())
                .count();
            let bases = alignment
                .iter()
                .filter(|base_or_gap| base_or_gap.is_base())
                .count();

            let bases = &self.sequence.bases[bases_before..(bases_before + bases)];
            let sequence = Sequence {
                bases,
                molecule: self.sequence.molecule,
            };
            let alignment = AlignedSequenceRef(alignment);

            GappedSequence {
                sequence,
                alignment,
            }
        })
    }
}

impl GappedSequenceLike for GappedSequence<'_> {
    #[inline]
    fn to_cstring(&self, molecule: Option<Molecule>) -> CString {
        // Rough estimation
        let estimated_len = self.alignment.0.len().max(self.sequence.bases.len()) + 1;
        let molecule = molecule.unwrap_or(self.sequence.molecule);
        sequence_cstring_from_iter(
            self.sequence.bases.iter().copied(),
            self.alignment.0.iter().copied(),
            molecule,
            estimated_len,
        )
    }
}

pub(crate) fn sequence_cstring_from_iter<S, B>(
    sequence: S,
    base_or_gap: B,
    molecule: Molecule,
    estimated_len: usize,
) -> CString
where
    S: Iterator<Item = Base>,
    B: Iterator<Item = BaseOrGap>,
{
    let mut chars = Vec::with_capacity(estimated_len);
    let mut sequence = sequence.map(|base| base.to_byte(molecule));
    let iter = base_or_gap.filter_map(|alignment| match alignment {
        BaseOrGap::Base => sequence.next(),
        BaseOrGap::Gap => Some(b'-'),
    });
    chars.extend(iter);
    chars.extend(sequence);
    chars.push(b'\0');

    CString::from_vec_with_nul(chars).unwrap()
}

impl<'a> IntoIterator for &'a GappedSequence<'a> {
    type Item = StatefulBaseOrGap;
    type IntoIter = GappedSequenceIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let GappedSequence {
            sequence,
            alignment,
        } = self;

        let bases = sequence.bases.iter();
        let alignment = alignment.0.iter();
        GappedSequenceIter { bases, alignment }
    }
}

impl<'a> IntoIterator for GappedSequence<'a> {
    type Item = StatefulBaseOrGap;
    type IntoIter = GappedSequenceIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let GappedSequence {
            sequence,
            alignment,
        } = self;

        let bases = sequence.bases.iter();
        let alignment = alignment.0.iter();
        GappedSequenceIter { bases, alignment }
    }
}

impl Display for GappedSequence<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use fmt::Write;

        self.iter().try_for_each(|b| match b {
            StatefulBaseOrGap::Base(b) => write!(f, "{}", b.display(self.sequence.molecule)),
            StatefulBaseOrGap::Gap => f.write_char('-'),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum StatefulBaseOrGap {
    Base(Base),
    Gap,
}

impl StatefulBaseOrGap {
    #[inline]
    pub(crate) fn to_base(self) -> Option<Base> {
        match self {
            Self::Base(base) => Some(base),
            Self::Gap => None,
        }
    }
}

pub(crate) struct GappedSequenceIter<'a> {
    bases: slice::Iter<'a, Base>,
    alignment: slice::Iter<'a, BaseOrGap>,
}

impl Iterator for GappedSequenceIter<'_> {
    type Item = StatefulBaseOrGap;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.alignment.next()? {
            BaseOrGap::Base => self.bases.next().copied().map(StatefulBaseOrGap::Base),
            BaseOrGap::Gap => self
                .bases
                .as_slice()
                .is_empty()
                .not()
                .then_some(StatefulBaseOrGap::Gap),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let bases = self.bases.size_hint();
        let alignment = self.alignment.size_hint();

        (
            bases.0.min(alignment.0),
            bases
                .1
                .map(|bases| alignment.1.map_or(bases, |alignment| bases.max(alignment)))
                .or(alignment.1),
        )
    }
}

impl DoubleEndedIterator for GappedSequenceIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.alignment.next_back()? {
            BaseOrGap::Base => self.bases.next_back().copied().map(StatefulBaseOrGap::Base),
            BaseOrGap::Gap => self
                .bases
                .as_slice()
                .is_empty()
                .not()
                .then_some(StatefulBaseOrGap::Gap),
        }
    }
}
