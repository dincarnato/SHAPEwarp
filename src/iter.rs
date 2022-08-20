use std::iter::FusedIterator;

pub(crate) trait IterWithRestExt<T> {
    fn iter_with_rest(&self) -> IterWithRest<'_, T>;
}

pub(crate) struct IterWithRest<'a, T>(&'a [T]);

impl<'a, T> Iterator for IterWithRest<'a, T> {
    type Item = (&'a T, &'a [T]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (first, rest) = self.0.split_first()?;
        self.0 = rest;
        Some((first, rest))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

impl<T> ExactSizeIterator for IterWithRest<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> FusedIterator for IterWithRest<'_, T> {}

impl<T> DoubleEndedIterator for IterWithRest<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (last, rest) = self.0.split_last()?;
        self.0 = rest;
        Some((last, rest))
    }
}

impl<T> IterWithRestExt<T> for [T] {
    #[inline]
    fn iter_with_rest(&self) -> IterWithRest<'_, T> {
        IterWithRest(self)
    }
}
