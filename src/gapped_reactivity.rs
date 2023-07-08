use std::{iter, ops::Not, slice};

use serde::{ser::SerializeSeq, Serialize, Serializer};

use crate::{
    aligner::{AlignedSequenceRef, BaseOrGap},
    db_file::ReactivityLike,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GappedReactivity<'a, T> {
    pub(crate) reactivity: &'a [T],
    pub(crate) alignment: AlignedSequenceRef<'a>,
}

pub(crate) trait GappedReactivityLike<R> {
    type AlignmentIter<'a>: Iterator<Item = BaseOrGap> + 'a
    where
        Self: 'a;

    type ReactivityIter<'a>: Iterator<Item = R> + 'a
    where
        Self: 'a;

    fn alignment(&self) -> Self::AlignmentIter<'_>;
    fn reactivity(&self) -> Self::ReactivityIter<'_>;
}

impl<'a, T> Serialize for GappedReactivity<'a, T>
where
    T: ReactivityLike + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut usable_alignment = 0;
        let gaps = self
            .alignment
            .0
            .iter()
            .scan(0, |bases, base| {
                if *bases == self.reactivity.len() {
                    None
                } else {
                    usable_alignment += 1;
                    if let BaseOrGap::Base = base {
                        *bases += 1;
                    }
                    Some(base)
                }
            })
            .filter(|base| matches!(base, BaseOrGap::Gap))
            .count();
        let usable_alignment = usable_alignment;
        let len = self.reactivity.len() + gaps;
        let mut seq = serializer.serialize_seq(Some(len))?;

        let mut reactivities = self.reactivity.iter();
        for base_or_gap in &self.alignment.0[..usable_alignment] {
            match base_or_gap {
                BaseOrGap::Base => match reactivities.next() {
                    Some(reactivity) if reactivity.is_nan() => seq.serialize_element("NaN")?,
                    Some(reactivity) => seq.serialize_element(reactivity)?,
                    None => break,
                },
                BaseOrGap::Gap => seq.serialize_element(&f32::NAN)?,
            }
        }

        reactivities.try_for_each(|reactivity| seq.serialize_element(reactivity))?;
        seq.end()
    }
}

impl<T> GappedReactivityLike<T> for GappedReactivity<'_, T>
where
    T: Copy,
{
    type AlignmentIter<'a> = iter::Copied<slice::Iter<'a, BaseOrGap>>
    where
        Self: 'a;

    type ReactivityIter<'a> = iter::Copied<slice::Iter<'a, T>>
    where
        Self: 'a;

    #[inline]
    fn alignment(&self) -> Self::AlignmentIter<'_> {
        self.alignment.0.iter().copied()
    }

    #[inline]
    fn reactivity(&self) -> Self::ReactivityIter<'_> {
        self.reactivity.iter().copied()
    }
}

impl<'a, T> IntoIterator for &'a GappedReactivity<'a, T>
where
    T: Copy,
{
    type Item = GappedReactivityValue<T>;
    type IntoIter = GappedReactivityIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let reactivity = self.reactivity.iter();
        let alignment = self.alignment.0.iter();
        GappedReactivityIter {
            reactivity,
            alignment,
        }
    }
}

impl<'a, T> IntoIterator for GappedReactivity<'a, T>
where
    T: Copy,
{
    type Item = GappedReactivityValue<T>;
    type IntoIter = GappedReactivityIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let reactivity = self.reactivity.iter();
        let alignment = self.alignment.0.iter();
        GappedReactivityIter {
            reactivity,
            alignment,
        }
    }
}

pub(crate) struct GappedReactivityIter<'a, T> {
    reactivity: slice::Iter<'a, T>,
    alignment: slice::Iter<'a, BaseOrGap>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GappedReactivityValue<T> {
    Reactivity(T),
    Gap,
}

impl<'a, T> Iterator for GappedReactivityIter<'a, T>
where
    T: Copy,
{
    type Item = GappedReactivityValue<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.alignment.next()? {
            BaseOrGap::Base => self
                .reactivity
                .next()
                .copied()
                .map(GappedReactivityValue::Reactivity),

            BaseOrGap::Gap => self
                .reactivity
                .as_slice()
                .is_empty()
                .not()
                .then_some(GappedReactivityValue::Gap),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let reactivity = self.reactivity.size_hint();
        let alignment = self.alignment.size_hint();

        (
            reactivity.0.min(alignment.0),
            reactivity
                .1
                .map(|bases| alignment.1.map_or(bases, |alignment| bases.max(alignment)))
                .or(alignment.1),
        )
    }
}
