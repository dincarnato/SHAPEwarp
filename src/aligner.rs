use std::{
    cmp, iter,
    num::NonZeroUsize,
    ops::{ControlFlow, Not, Range, RangeInclusive},
    slice::{self, SliceIndex},
};

use ndarray::{s, Array2};

use crate::{
    calc_base_alignment_score,
    cli::{AlignmentArgs, Cli},
    db_file::{self, ReactivityLike, ReactivityWithPlaceholder},
    get_sequence_base_alignment_score, query_file, Base, Reactivity, SequenceEntry,
};

const MIN_BAND_SIZE: usize = 10;

pub(crate) struct Aligner<'a> {
    cli: &'a Cli,
    matrix: Array2<Cell>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Cell {
    score: Reactivity,
    traceback: Traceback,
    dropoff: u16,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            score: Reactivity::NAN,
            traceback: Default::default(),
            dropoff: Default::default(),
        }
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        (self.score == other.score || (self.score.is_nan() && other.score.is_nan()))
            && self.traceback == other.traceback
            && self.dropoff == other.dropoff
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlignResult<Alignment> {
    pub(crate) query_index: usize,
    pub(crate) query_alignment: Alignment,
    pub(crate) target_index: usize,
    pub(crate) target_alignment: Alignment,
    pub(crate) score: Reactivity,
}

pub(crate) struct AlignParams<'a> {
    pub(crate) query: &'a query_file::Entry,
    pub(crate) target: &'a db_file::Entry,
    pub(crate) query_range: RangeInclusive<usize>,
    pub(crate) target_range: RangeInclusive<usize>,
    pub(crate) seed_score: Reactivity,
    pub(crate) align_tolerance: &'a AlignTolerance,
    pub(crate) direction: Direction,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct AlignmentResult<Alignment> {
    pub(crate) query: Alignment,
    pub(crate) target: Alignment,
}

pub(crate) trait AlignBehavior {
    type Alignment: Default;

    fn get_alignment<const FORWARD: bool>(
        matrix: &Array2<Cell>,
        best_cell: [usize; 2],
    ) -> AlignmentResult<Self::Alignment>;

    fn merge_upstream_downstream(
        upstream: Self::Alignment,
        downstream: Self::Alignment,
        seed_size: NonZeroUsize,
    ) -> Self::Alignment;
}

pub(crate) struct NoOpBehavior;
pub(crate) struct BacktrackBehavior;

impl AlignBehavior for NoOpBehavior {
    type Alignment = ();

    #[inline]
    fn get_alignment<const FORWARD: bool>(
        _matrix: &Array2<Cell>,
        _best_cell: [usize; 2],
    ) -> AlignmentResult<Self::Alignment> {
        AlignmentResult {
            query: (),
            target: (),
        }
    }

    #[inline(always)]
    fn merge_upstream_downstream(
        _upstream: Self::Alignment,
        _downstream: Self::Alignment,
        _seed_size: NonZeroUsize,
    ) -> Self::Alignment {
    }
}

impl AlignBehavior for BacktrackBehavior {
    type Alignment = AlignedSequence;

    fn get_alignment<const FORWARD: bool>(
        matrix: &Array2<Cell>,
        best_cell: [usize; 2],
    ) -> AlignmentResult<Self::Alignment> {
        let inner = || {
            assert!(best_cell[0] < matrix.nrows());
            assert!(best_cell[1] < matrix.ncols());

            let mut target = vec![];
            let mut query = vec![];

            let mut cell = best_cell;
            while cell != [0, 0] {
                let cell_data = &matrix[cell];
                let t;
                let q;
                ([t, q], cell) = match cell_data.traceback {
                    Traceback::Diagonal => ([BaseOrGap::Base; 2], [cell[0] - 1, cell[1] - 1]),
                    Traceback::Left => ([BaseOrGap::Gap, BaseOrGap::Base], [cell[0], cell[1] - 1]),
                    Traceback::Up => ([BaseOrGap::Base, BaseOrGap::Gap], [cell[0] - 1, cell[1]]),
                };

                target.push(t);
                query.push(q);
            }

            target.push(BaseOrGap::Base);
            query.push(BaseOrGap::Base);

            [target, query]
        };

        let [mut target, mut query] = inner();

        if FORWARD {
            target.reverse();
            query.reverse();
        }

        let target = AlignedSequence(target);
        let query = AlignedSequence(query);
        AlignmentResult { target, query }
    }

    fn merge_upstream_downstream(
        mut upstream: Self::Alignment,
        downstream: Self::Alignment,
        seed_size: NonZeroUsize,
    ) -> Self::Alignment {
        // Last base of upstream is the first base of the seed, sequence is reversed
        // First base of downstream is the last base of the seed

        upstream.0.reverse();
        upstream.0.extend(
            iter::repeat(BaseOrGap::Base)
                .take(seed_size.get() - 1)
                .chain(downstream.0.into_iter().skip(1)),
        );
        upstream
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BaseOrGap {
    Base,
    Gap,
}

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct AlignedSequence(pub(crate) Vec<BaseOrGap>);

impl<'a> Aligner<'a> {
    pub(crate) fn new(cli: &'a Cli) -> Self {
        Self {
            cli,
            matrix: Default::default(),
        }
    }

    pub(crate) fn align<Behavior: AlignBehavior>(
        &mut self,
        params: AlignParams,
    ) -> AlignResult<Behavior::Alignment> {
        let AlignParams {
            query,
            target,
            query_range,
            target_range,
            seed_score,
            align_tolerance,
            direction,
        } = params;

        let trimmed_query_range = trimmed_range(query.reactivity());
        let trimmed_target_range = trimmed_range(target.reactivity());

        match direction {
            Direction::Downstream => {
                let query_start = (*query_range.end() + 1).min(trimmed_query_range.end);
                let target_start = (*target_range.end() + 1).min(trimmed_target_range.end);
                let tolerance = align_tolerance.downstream;
                let query = EntrySlice::forward(
                    query,
                    query_start
                        ..trimmed_query_range
                            .end
                            .min(query_start.saturating_add(tolerance)),
                );
                let target = EntrySlice::forward(
                    target,
                    target_start
                        ..trimmed_target_range
                            .end
                            .min(target_start.saturating_add(tolerance)),
                );

                if query.is_empty() || target.is_empty() {
                    return AlignResult {
                        query_index: query_start.saturating_sub(1),
                        query_alignment: Default::default(),
                        target_index: target_start.saturating_sub(1),
                        target_alignment: Default::default(),
                        score: seed_score,
                    };
                }

                let best_cell = self.handle_slices(&query, &target, seed_score);
                let target_index = target_start + best_cell[0] - 1;
                let query_index = query_start + best_cell[1] - 1;
                let score = self.matrix[best_cell].score;
                let AlignmentResult {
                    query: query_alignment,
                    target: target_alignment,
                } = Behavior::get_alignment::<true>(&self.matrix, best_cell);

                AlignResult {
                    query_index,
                    query_alignment,
                    target_index,
                    target_alignment,
                    score,
                }
            }

            Direction::Upstream => {
                let tolerance = align_tolerance.upstream;
                let query_end = (*query_range.start()).max(trimmed_query_range.start);
                let target_end = (*target_range.start()).max(trimmed_target_range.start);
                let query_start = query_end
                    .saturating_sub(tolerance)
                    .min(trimmed_query_range.end);
                let target_start = target_end
                    .saturating_sub(tolerance)
                    .min(trimmed_target_range.end);
                let query = EntrySlice::backward(query, query_start..query_end);
                let target = EntrySlice::backward(target, target_start..target_end);

                if query.is_empty() || target.is_empty() {
                    return AlignResult {
                        query_index: query_end.saturating_sub(1),
                        query_alignment: Default::default(),
                        target_index: target_end.saturating_sub(1),
                        target_alignment: Default::default(),
                        score: seed_score,
                    };
                }

                let best_cell = self.handle_slices(&query, &target, seed_score);
                let target_index = target_end.checked_sub(best_cell[0]).unwrap();
                let query_index = query_end.checked_sub(best_cell[1]).unwrap();
                let score = self.matrix[best_cell].score;
                let AlignmentResult {
                    query: query_alignment,
                    target: target_alignment,
                } = Behavior::get_alignment::<true>(&self.matrix, best_cell);

                AlignResult {
                    query_index,
                    query_alignment,
                    target_index,
                    target_alignment,
                    score,
                }
            }
        }
    }

    fn handle_slices<'b, const FW: bool>(
        &mut self,
        query: &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>,
        target: &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>,
        seed_score: Reactivity,
    ) -> [usize; 2]
    where
        &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>:
            IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>,
        EntrySlice<'b, ReactivityWithPlaceholder, FW>: IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>
            + EntrySliceExt<ReactivityWithPlaceholder>,
        &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>:
            IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>,
        EntrySlice<'b, ReactivityWithPlaceholder, FW>: IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>
            + EntrySliceExt<ReactivityWithPlaceholder>,
    {
        // TODO: this is unconvenient, we cannot arbitrarly reshape ndarrays, maybe we
        // should avoid reallocations
        self.matrix = self.create_matrix(query, target);
        self.fill_borders(seed_score);
        self.align_diagonal_band(query, target)
    }

    #[inline(always)]
    fn create_matrix<'b, const FW: bool>(
        &mut self,
        query: &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>,
        target: &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>,
    ) -> Array2<Cell>
    where
        &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>:
            IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>,
        EntrySlice<'b, ReactivityWithPlaceholder, FW>: IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>
            + EntrySliceExt<ReactivityWithPlaceholder>,
        &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>:
            IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>,
        EntrySlice<'b, ReactivityWithPlaceholder, FW>: IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>
            + EntrySliceExt<ReactivityWithPlaceholder>,
    {
        // We need to add 1 row and column for the starting base of the seed.
        let rows = target.len() + 1;
        let cols = query.len() + 1;

        debug_assert!(rows > 0);
        debug_assert!(cols > 0);
        let size = cmp::min(rows, cols) + self.calc_band_size_from_rows_cols(rows, cols);
        Array2::default([size.min(rows), size.min(cols)])
    }

    fn fill_borders(&mut self, seed_score: Reactivity) {
        let AlignmentArgs {
            align_gap_open_penalty,
            align_gap_ext_penalty,
            align_max_drop_off_rate,
            ..
        } = self.cli.alignment_args;

        self.matrix[(0, 0)].score = seed_score;
        let dropoff_score = seed_score * align_max_drop_off_rate;
        let first_open_score = seed_score + align_gap_open_penalty;

        let make_cell_handler = |mut score, traceback| {
            move |mut dropoff: u16, cell: &mut Cell| -> u16 {
                score += align_gap_ext_penalty;
                if score < dropoff_score {
                    dropoff += 1;
                }
                *cell = Cell {
                    score,
                    dropoff,
                    traceback,
                };
                dropoff
            }
        };

        let useful_cells = self.calc_band_size() + 2;

        self.matrix
            .row_mut(0)
            .iter_mut()
            .take(useful_cells)
            .skip(1)
            .fold(0, make_cell_handler(first_open_score, Traceback::Left));

        self.matrix
            .column_mut(0)
            .iter_mut()
            .take(useful_cells)
            .skip(1)
            .fold(0, make_cell_handler(first_open_score, Traceback::Up));
    }

    fn align_diagonal_band<'b, const FW: bool>(
        &mut self,
        query: &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>,
        target: &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>,
    ) -> [usize; 2]
    where
        &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>:
            IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>,
        EntrySlice<'b, ReactivityWithPlaceholder, FW>: IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>
            + EntrySliceExt<ReactivityWithPlaceholder>,
        &'b EntrySlice<'b, ReactivityWithPlaceholder, FW>:
            IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>,
        EntrySlice<'b, ReactivityWithPlaceholder, FW>: IntoIterator<Item = EntryElement<ReactivityWithPlaceholder>>
            + EntrySliceExt<ReactivityWithPlaceholder>,
    {
        #[derive(Debug)]
        struct State {
            best_score: Reactivity,
            dropoff_score: Reactivity,
            best_cell: [usize; 2],
        }

        impl State {
            fn new(
                best_score: Reactivity,
                best_cell: [usize; 2],
                alignment_args: &AlignmentArgs,
            ) -> Self {
                Self {
                    best_score,
                    dropoff_score: best_score * alignment_args.align_max_drop_off_rate,
                    best_cell,
                }
            }
        }

        let rows = target.len() + 1;
        let cols = query.len() + 1;
        let band_size = self.calc_band_size();
        let diag_len = cmp::min(rows, cols);
        let alignment_args = &self.cli.alignment_args;
        let initial_score = self.matrix[(0, 0)].score;
        let state = State::new(initial_score, [0, 0], alignment_args);

        let calc_alignment_score = if alignment_args.align_score_seq {
            |query: &EntryElement<ReactivityWithPlaceholder>,
             target: &EntryElement<ReactivityWithPlaceholder>,
             cli: &Cli| {
                calc_base_alignment_score(query.reactivity, target.reactivity, cli)
                    + get_sequence_base_alignment_score(
                        query.base,
                        target.base,
                        &cli.alignment_args,
                    )
            }
        } else {
            |query: &EntryElement<ReactivityWithPlaceholder>,
             target: &EntryElement<ReactivityWithPlaceholder>,
             cli: &Cli| {
                calc_base_alignment_score(query.reactivity, target.reactivity, cli)
            }
        };

        let handle_new_score = |cell: &mut Cell,
                                state: &mut State,
                                parent_cell: &Cell,
                                row_index: usize,
                                col_index: usize| {
            debug_assert!(cell.score.is_nan().not());

            if cell.score >= state.best_score {
                *state = State::new(cell.score, [row_index, col_index], alignment_args);
            } else if cell.score < state.dropoff_score && cell.score > 0. {
                let dropoff = parent_cell.dropoff.saturating_add(1);
                if dropoff <= self.cli.alignment_args.align_max_drop_off_bases {
                    cell.dropoff = dropoff;
                } else {
                    cell.score = -Reactivity::INFINITY;
                    cell.dropoff = 0;
                }
            }
        };

        let align_result = (1..(diag_len + band_size)).zip(target).try_fold(
            state,
            |mut state, (row_index, target)| {
                let band_range_start = row_index.saturating_sub(band_size).max(1);
                let band_range_end = (row_index + band_size).min(cols);
                let (above, mut left, mut band) = self.matrix.multi_slice_mut((
                    s![row_index - 1, (band_range_start - 1)..band_range_end],
                    s![row_index, band_range_start - 1],
                    s![row_index, band_range_start..band_range_end],
                ));

                let left = left
                    .first_mut()
                    .expect("left cell should be always available");
                if band_range_start > 1 {
                    left.score = score_from_gap(&above[0], alignment_args);
                    // left.score = 0.;
                    left.traceback = Traceback::Up;
                    handle_new_score(left, &mut state, &above[0], row_index, band_range_start - 1)
                };

                let left = &*left;
                let query_band = query.slice((band_range_start - 1)..(band_range_end - 1));

                let (_, mut state, _final_col_index, has_valid_score) = band
                    .iter_mut()
                    .zip(query_band)
                    .zip(above.windows(2).into_iter())
                    .fold(
                        (left, state, band_range_start, false),
                        |(left, mut state, col_index, has_valid_score), ((cell, query), above)| {
                            let [diag, above] = [&above[0], &above[1]];

                            let alignment_score = calc_alignment_score(&query, &target, self.cli);
                            let left_score = score_from_gap(left, alignment_args);
                            let above_score = score_from_gap(above, alignment_args);
                            let diag_score = diag.score + alignment_score;

                            debug_assert!(left_score.is_nan().not());
                            debug_assert!(above_score.is_nan().not());
                            debug_assert!(diag_score.is_nan().not());
                            let (traceback, score, parent_cell) =
                                if diag_score >= left_score && diag_score >= above_score {
                                    (Traceback::Diagonal, diag_score, diag)
                                } else if left_score >= above_score {
                                    (Traceback::Left, left_score, left)
                                } else {
                                    (Traceback::Up, above_score, above)
                                };

                            cell.score = score;
                            cell.traceback = traceback;
                            handle_new_score(cell, &mut state, parent_cell, row_index, col_index);

                            (cell, state, col_index + 1, has_valid_score | (score > 0.))
                        },
                    );
                debug_assert_eq!(_final_col_index, band_range_end);

                if has_valid_score.not() {
                    ControlFlow::Break(state.best_cell)
                } else {
                    if band_range_end < cols {
                        let query_base = query.get(band_range_end - 1);

                        let (diag, left, mut cell) = self.matrix.multi_slice_mut((
                            s![row_index - 1, band_range_end - 1],
                            s![row_index, band_range_end - 1],
                            s![row_index, band_range_end],
                        ));
                        let diag = diag.first().expect("diag cell should be always available");
                        let left = left.first().expect("left cell should be always available");
                        let cell = cell.first_mut().expect("cell should be always available");

                        let alignment_score = calc_alignment_score(&query_base, &target, self.cli);
                        let left_score = score_from_gap(left, alignment_args);
                        let diag_score = diag.score + alignment_score;

                        debug_assert!(left_score.is_nan().not());
                        debug_assert!(diag_score.is_nan().not());
                        let (traceback, score, parent_cell) = if diag_score >= left_score {
                            (Traceback::Diagonal, diag_score, diag)
                        } else {
                            (Traceback::Left, left_score, left)
                        };

                        cell.score = score;
                        cell.traceback = traceback;
                        handle_new_score(cell, &mut state, parent_cell, row_index, band_range_end);
                    }

                    ControlFlow::Continue(state)
                }
            },
        );

        match align_result {
            ControlFlow::Break(best_cell) => best_cell,
            ControlFlow::Continue(state) => state.best_cell,
        }
    }

    #[inline]
    fn calc_band_size(&self) -> usize {
        self.calc_band_size_from_rows_cols(self.matrix.nrows(), self.matrix.ncols())
    }

    fn calc_band_size_from_rows_cols(&self, rows: usize, cols: usize) -> usize {
        let diag_len = cmp::min(rows, cols);
        ((diag_len as f32 * self.cli.alignment_args.align_len_tolerance).round() as usize)
            .max(MIN_BAND_SIZE)
    }
}

fn score_from_gap(cell: &Cell, alignment_args: &AlignmentArgs) -> Reactivity {
    use Traceback::*;

    let open_gap_score = match cell.traceback {
        Diagonal => alignment_args.align_gap_open_penalty,
        Left | Up => 0.,
    };

    cell.score + alignment_args.align_gap_ext_penalty + open_gap_score
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    Upstream,
    Downstream,
}

#[derive(Debug)]
struct EntrySlice<'a, T, const FW: bool> {
    sequence: &'a [Base],
    reactivity: &'a [T],
}

impl<'a, T, const FW: bool> EntrySlice<'a, T, FW> {
    fn new<E, R>(entry: &'a E, range: R) -> Self
    where
        E: SequenceEntry<Reactivity = T>,
        R: SliceIndex<[Base], Output = [Base]> + SliceIndex<[T], Output = [T]> + Clone,
    {
        let sequence = &entry.sequence()[range.clone()];
        let reactivity = &entry.reactivity()[range];

        Self {
            sequence,
            reactivity,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.reactivity.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.reactivity.is_empty()
    }

    fn slice(&self, range: Range<usize>) -> Self {
        let range = if FW {
            range
        } else {
            let len = self.len();
            (len - range.end)..(len - range.start)
        };

        let sequence = &self.sequence[range.clone()];
        let reactivity = &self.reactivity[range];

        Self {
            sequence,
            reactivity,
        }
    }
}

impl<'a, T> EntrySlice<'a, T, true> {
    #[inline]
    fn forward<E, R>(entry: &'a E, range: R) -> Self
    where
        E: SequenceEntry<Reactivity = T>,
        R: SliceIndex<[Base], Output = [Base]> + SliceIndex<[T], Output = [T]> + Clone,
    {
        Self::new(entry, range)
    }
}

impl<'a, T> EntrySlice<'a, T, false> {
    #[inline]
    fn backward<E, R>(entry: &'a E, range: R) -> Self
    where
        E: SequenceEntry<Reactivity = T>,
        R: SliceIndex<[Base], Output = [Base]> + SliceIndex<[T], Output = [T]> + Clone,
    {
        Self::new(entry, range)
    }
}

trait EntrySliceExt<T> {
    fn get(&self, index: usize) -> EntryElement<T>;
}

impl<T: Copy> EntrySliceExt<T> for EntrySlice<'_, T, true> {
    fn get(&self, index: usize) -> EntryElement<T> {
        let base = self.sequence[index];
        let reactivity = self.reactivity[index];

        EntryElement { base, reactivity }
    }
}

impl<T: Copy> EntrySliceExt<T> for EntrySlice<'_, T, false> {
    fn get(&self, index: usize) -> EntryElement<T> {
        let len = self.len();
        let index = len - 1 - index;
        let base = self.sequence[index];
        let reactivity = self.reactivity[index];

        EntryElement { base, reactivity }
    }
}

impl<'a, T: Copy> IntoIterator for &'a EntrySlice<'a, T, true> {
    type Item = EntryElement<T>;
    type IntoIter = EntryIterFw<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter();
        let reactivity = self.reactivity.iter();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

impl<'a, T: Copy> IntoIterator for &'a EntrySlice<'a, T, false> {
    type Item = EntryElement<T>;
    type IntoIter = EntryIterBw<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter().rev();
        let reactivity = self.reactivity.iter().rev();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

impl<'a, T: Copy> IntoIterator for EntrySlice<'a, T, true> {
    type Item = EntryElement<T>;
    type IntoIter = EntryIterFw<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter();
        let reactivity = self.reactivity.iter();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

impl<'a, T: Copy> IntoIterator for EntrySlice<'a, T, false> {
    type Item = EntryElement<T>;
    type IntoIter = EntryIterBw<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter().rev();
        let reactivity = self.reactivity.iter().rev();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

struct EntryIterFw<'a, T> {
    sequence: slice::Iter<'a, Base>,
    reactivity: slice::Iter<'a, T>,
}

struct EntryIterBw<'a, T> {
    sequence: iter::Rev<slice::Iter<'a, Base>>,
    reactivity: iter::Rev<slice::Iter<'a, T>>,
}

#[derive(Debug)]
struct EntryElement<T> {
    base: Base,
    reactivity: T,
}

impl<'a, T: Copy> Iterator for EntryIterFw<'a, T> {
    type Item = EntryElement<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let base = *self.sequence.next()?;
        let reactivity = *self.reactivity.next()?;

        Some(Self::Item { base, reactivity })
    }
}

impl<'a, T: Copy> Iterator for EntryIterBw<'a, T> {
    type Item = EntryElement<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let base = *self.sequence.next()?;
        let reactivity = *self.reactivity.next()?;

        Some(Self::Item { base, reactivity })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Traceback {
    Diagonal,
    Left,
    Up,
}

impl Default for Traceback {
    fn default() -> Self {
        Self::Diagonal
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AlignTolerance {
    upstream: usize,
    downstream: usize,
}

pub(crate) fn calc_seed_align_tolerance(
    query_seed_range: RangeInclusive<usize>,
    target_seed_range: RangeInclusive<usize>,
    query_range: Range<usize>,
    target_range: Range<usize>,
    align_len_tolerance: f32,
) -> AlignTolerance {
    let upstream_len = (query_seed_range.start() - query_range.start)
        .min(target_seed_range.start() - target_range.start);
    let downstream_len = (query_range.end - query_seed_range.end() - 1)
        .min(target_range.end - target_seed_range.end() - 1);
    let upstream = upstream_len + (upstream_len as f32 * align_len_tolerance).round() as usize;
    let downstream =
        downstream_len + (downstream_len as f32 * align_len_tolerance).round() as usize;

    AlignTolerance {
        upstream,
        downstream,
    }
}

pub fn trimmed_range<T: ReactivityLike>(reactivities: &[T]) -> Range<usize> {
    match reactivities.iter().copied().position(|x| x.is_nan().not()) {
        Some(start) => {
            let new_range = &reactivities[start..];
            let end_offset = new_range
                .iter()
                .copied()
                .rposition(|x| x.is_nan().not())
                .map(|x| x + 1)
                .unwrap_or(new_range.len());
            start..(start + end_offset)
        }
        None => 0..(reactivities.len()),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ndarray::array;

    use super::*;

    #[test]
    fn fill_borders_small() {
        let cli = Cli::dummy();
        const SEED_SCORE: Reactivity = 4.5;
        let AlignmentArgs {
            align_gap_open_penalty,
            align_gap_ext_penalty,
            ..
        } = &cli.alignment_args;

        let mut aligner = Aligner::new(&cli);
        aligner.matrix = Array2::default((4, 5));
        aligner.fill_borders(SEED_SCORE);

        macro_rules! b {
            ($score:expr) => {
                Cell {
                    score: $score,
                    ..Default::default()
                }
            };

            ($score:expr, $dropoff:literal, up) => {
                Cell {
                    score: $score,
                    dropoff: $dropoff,
                    traceback: Traceback::Up,
                }
            };

            ($score:expr, $dropoff:literal, left) => {
                Cell {
                    score: $score,
                    dropoff: $dropoff,
                    traceback: Traceback::Left,
                }
            };
        }

        macro_rules! i {
            () => {
                Cell::default()
            };
        }

        assert_eq!(
            aligner.matrix,
            array![
                [
                    b!(SEED_SCORE),
                    b!(
                        SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty,
                        1,
                        left
                    ),
                    b!(
                        SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * 2.,
                        2,
                        left
                    ),
                    b!(
                        SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * 3.,
                        3,
                        left
                    ),
                    b!(
                        SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * 4.,
                        4,
                        left
                    ),
                ],
                [
                    b!(
                        SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty,
                        1,
                        up
                    ),
                    i!(),
                    i!(),
                    i!(),
                    i!(),
                ],
                [
                    b!(
                        SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * 2.,
                        2,
                        up
                    ),
                    i!(),
                    i!(),
                    i!(),
                    i!(),
                ],
                [
                    b!(
                        SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * 3.,
                        3,
                        up
                    ),
                    i!(),
                    i!(),
                    i!(),
                    i!(),
                ],
            ],
        )
    }

    #[test]
    fn band_size() {
        let mut cli = Cli::dummy();
        cli.alignment_args.align_len_tolerance = 0.25;
        let mut aligner = Aligner::new(&cli);
        aligner.matrix = Array2::default((100, 200));
        assert_eq!(aligner.calc_band_size(), 25);

        aligner.matrix = Array2::default((200, 100));
        assert_eq!(aligner.calc_band_size(), 25);

        aligner.matrix = Array2::default((36, 200));
        assert_eq!(aligner.calc_band_size(), MIN_BAND_SIZE);
    }

    #[test]
    fn fill_borders_large() {
        let cli = Cli::dummy();
        const SEED_SCORE: Reactivity = 4.5;
        let AlignmentArgs {
            align_gap_open_penalty,
            align_gap_ext_penalty,
            align_max_drop_off_rate,
            ..
        } = cli.alignment_args;

        let mut aligner = Aligner::new(&cli);
        aligner.matrix = Array2::default((17, 19));
        aligner.fill_borders(SEED_SCORE);

        // Let's say we have a band size equal to 2. Then the following matrix:
        //
        // X x x c - - - -
        // x X x x c - - -
        // x x X x x c - -
        // c x x X x x c -
        // - c x x X x x -
        //
        // Where
        // X: diagonal
        // x: band around diagonal
        // c: cell referenced
        //
        // Therefore, the number of cells on the borders are BAND_SIZE + 2

        assert_eq!(
            aligner.matrix[[0, 0]],
            Cell {
                score: SEED_SCORE,
                traceback: Traceback::Diagonal,
                dropoff: 0
            }
        );
        let first_row = aligner.matrix.row(0);
        first_row
            .iter()
            .enumerate()
            .skip(1)
            .take(MIN_BAND_SIZE + 1)
            .fold(0, |mut dropoff, (col, cell)| {
                let expected_score =
                    SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * col as Reactivity;
                assert_eq!(cell.score, expected_score,);

                assert_eq!(cell.traceback, Traceback::Left);
                if expected_score < SEED_SCORE * align_max_drop_off_rate {
                    dropoff += 1;
                }
                assert_eq!(cell.dropoff, dropoff);

                dropoff
            });

        first_row.iter().skip(MIN_BAND_SIZE + 2).for_each(|cell| {
            assert_eq!(
                cell,
                &Cell {
                    score: Reactivity::NAN,
                    traceback: Traceback::Diagonal,
                    dropoff: 0,
                }
            );
        });

        let first_col = aligner.matrix.column(0);
        first_col
            .iter()
            .enumerate()
            .skip(1)
            .take(MIN_BAND_SIZE + 1)
            .fold(0, |mut dropoff, (row, cell)| {
                let expected_score =
                    SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * row as Reactivity;
                assert_eq!(cell.score, expected_score,);
                assert_eq!(cell.traceback, Traceback::Up);

                if expected_score < SEED_SCORE * align_max_drop_off_rate {
                    dropoff += 1;
                }
                assert_eq!(cell.dropoff, dropoff);

                dropoff
            });

        first_col.iter().skip(MIN_BAND_SIZE + 2).for_each(|cell| {
            assert_eq!(
                cell,
                &Cell {
                    score: Reactivity::NAN,
                    traceback: Traceback::Diagonal,
                    dropoff: 0,
                }
            );
        });
    }

    macro_rules! read_raw_matrix_results {
        ($($start:tt . $end:tt)+) => {
            vec![$(parse_raw_matrix_result(concat!(stringify!($start), ".", stringify!($end)))),+]
        };
    }

    fn parse_raw_matrix_result(raw_value: &'static str) -> Cell {
        let (raw_traceback, raw_value) = raw_value.split_at(1);
        let (raw_score, raw_distance) = raw_value.split_once('_').unwrap();

        let traceback = match raw_traceback {
            "n" | "d" => Traceback::Diagonal,
            "l" => Traceback::Left,
            "u" => Traceback::Up,
            _ => panic!("invalid traceback"),
        };

        let score = raw_score.parse().unwrap();
        let dist_from_max_score = raw_distance[1..].parse().unwrap();

        Cell {
            score,
            traceback,
            dropoff: dist_from_max_score,
        }
    }

    #[test]
    fn align_diagonal_band_downstream() {
        let cli = Cli::dummy();
        const SEED_SCORE: Reactivity = 93.77;

        let query_slice = 80..;
        let query = query_file::read_file(Path::new("./test_data/query.txt")).unwrap();
        let mut query = query.into_iter().next().unwrap();
        query.cap_reactivities(cli.max_reactivity);
        let query_sequence = &query.sequence()[query_slice.clone()];
        let query_reactivity = &query.reactivity()[query_slice];
        let query = EntrySlice::<_, true> {
            sequence: query_sequence,
            reactivity: query_reactivity,
        };

        let target = db_file::read_file(Path::new("./test_data/test.db")).unwrap();
        let mut target = target.into_iter().next().unwrap();
        target.cap_reactivities(cli.max_reactivity);
        let target_slice = 838..;
        let target_sequence = &target.sequence()[target_slice.clone()];
        let target_reactivity = &target.reactivity()[target_slice];
        let target = EntrySlice {
            sequence: target_sequence,
            reactivity: target_reactivity,
        };

        let mut aligner = Aligner::new(&cli);
        aligner.matrix = aligner.create_matrix(&query, &target);
        aligner.fill_borders(SEED_SCORE);

        let cell = aligner.align_diagonal_band(&query, &target);

        let expected_matrix = read_raw_matrix_results!(
            n93.77_r0   l74.77_r1  l69.77_r2  l64.77_r3  l59.77_r4  l54.77_r5  l49.77_r6  l44.77_r7  l39.77_r8  l34.77_r9  l29.77_r10  l24.77_r11 l19.77_r12  l14.77_r13  l9.77_r14  l4.77_r15   l0.00_r16   l0.00_r17   l0.00_r18   l0.00_r19   l0.00_r20
            u74.77_r1   d93.55_r0  d75.79_r0  d70.74_r3  d66.39_r4  d60.97_r5  d55.76_r6  d49.94_r7  d41.23_r8  n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u69.77_r2   d76.17_r0  d93.72_r0  d76.01_r0  d70.31_r4  d66.37_r5  d61.17_r6  d56.78_r7  d50.87_r8  n0.00_r0   n0.00_r1    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u64.77_r3   d71.23_r3  d77.47_r0  d95.07_r0  d76.71_r0  d71.42_r5  d67.70_r6  d63.02_r7  d56.58_r8  n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r1    n0.00_r1    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u59.77_r4   d66.49_r4  d71.72_r4  d78.01_r0  d94.96_r0  d77.01_r0  d71.94_r6  d69.04_r7  d63.63_r8  n0.00_r0   n0.00_r1    n0.00_r0   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u54.77_r5   d60.73_r5  d68.29_r5  d73.57_r5  d79.21_r0  d96.58_r0  d78.84_r0  d73.29_r7  d68.10_r8  n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r1    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u49.77_r6   d56.67_r6  d61.60_r6  d69.21_r6  d73.84_r6  d79.89_r0  d97.47_r0  d80.55_r0  d73.52_r8  n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r1    n0.00_r1    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u44.77_r7   d50.34_r7  d58.48_r7  d63.36_r7  d70.80_r7  d75.83_r7  d81.68_r0  d98.43_r0  l79.43_r0  n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r1    n0.00_r1    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u39.77_r8   d45.31_r8  d52.12_r8  d60.21_r8  d64.98_r8  d72.76_r8  d77.59_r8  d82.61_r0  d96.56_r0  d80.78_r0  l61.78_r1   l56.78_r2  l51.78_r3   l46.78_r4   l41.78_r5  l36.78_r6   l31.78_r7   l26.78_r8   n0.00_r0    n0.00_r0    n0.00_r0
            u34.77_r9   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   d79.23_r0  d82.20_r0  d97.21_r0  d81.76_r0   d62.84_r2  d54.79_r3   d49.79_r4   d44.79_r5  d39.79_r6   d34.79_r7   d32.24_r8   n0.00_r0    n0.00_r0    n0.00_r0
            u29.77_r10  n0.00_r0   n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1   u60.23_r1  d80.46_r0  u78.21_r1  d92.97_r0   d81.52_r0  d64.84_r3   d56.79_r4   d51.79_r5  d46.79_r6   d41.79_r7   d29.39_r8   n0.00_r0    n0.00_r1    n0.00_r1
            u24.77_r11  n0.00_r0   n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1   u55.23_r2  d61.71_r2  d76.04_r1  d74.53_r2   d92.98_r0  d83.27_r0   d66.59_r4   d58.53_r5  d53.53_r6   d48.53_r7   d36.95_r8   n0.00_r0    n0.00_r1    n0.00_r1
            u19.77_r12  n0.00_r0   n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1   u50.23_r3  d56.46_r3  u57.04_r2  d71.80_r2   d74.29_r3  d94.98_r0   d85.27_r0   d68.59_r5  d60.53_r6   d55.53_r7   d43.14_r8   n0.00_r0    n0.00_r1    n0.00_r1
            u14.77_r13  n0.00_r0   n0.00_r0   n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1   u45.23_r4  d51.46_r4  u52.04_r3  u52.80_r3   d71.56_r3  d76.29_r4   d96.98_r0   d87.27_r0  d70.59_r6   d62.53_r7   d50.14_r8   n0.00_r0    n0.00_r1    n0.00_r1
            u9.77_r14   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r1   n0.00_r1   n0.00_r1   u40.23_r5  d46.46_r5  u47.04_r4  u47.80_r4   u52.56_r4  d73.56_r4   d78.29_r5   d98.98_r0  d89.27_r0   d72.59_r7   d57.14_r8   n0.00_r0    n0.00_r1    n0.00_r1
            u4.77_r15   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r1   n0.00_r1   u35.23_r6  d41.46_r6  u42.04_r5  u42.80_r5   u47.56_r5  d54.56_r5   d75.56_r5   d80.29_r0  d100.98_r0  d91.27_r0   l72.27_r1   l67.27_r2   l62.27_r3   l57.27_r4
            u0.00_r16   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r1   u30.23_r7  d36.46_r7  u37.04_r6  u37.80_r6   u42.56_r6  d49.56_r6   d56.56_r6   d77.56_r6  d82.29_r0   d102.98_r0  d85.87_r0   d67.30_r2   d62.07_r3   d56.94_r4
            u0.00_r17   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   u25.23_r8  d28.15_r8  d37.91_r8  d38.83_r7   d38.06_r7  d38.79_r7   d45.79_r7   u58.56_r7  d73.79_r7   u83.98_r0   d104.24_r0  d87.33_r0   d68.65_r3   d63.36_r4
            u0.00_r18   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    d37.75_r8  d32.06_r8   d32.79_r8   u53.56_r8  u54.79_r8   u78.98_r1   d85.71_r0   d105.77_r0  d88.96_r0   d70.35_r4
            u0.00_r19   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    u73.98_r2   d80.71_r2   d87.24_r0   d107.41_r0  d90.66_r0
            u0.00_r20   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r1   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r1    u68.98_r3   d75.71_r3   d82.24_r3   d88.87_r0   d109.10_r0
            u0.00_r21   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r1   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r1    u63.98_r4   d70.73_r4   d77.26_r4   d83.89_r4   d90.59_r0
            u0.00_r22   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r1    u58.98_r5   d65.66_r5   d72.60_r5   d79.02_r5   d85.60_r5
            u0.00_r23   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r1    n0.00_r1   n0.00_r1    u53.98_r6   d60.62_r6   d67.49_r6   d74.32_r6   d80.69_r6
            u0.00_r24   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r1   n0.00_r1    u48.98_r7   d55.47_r7   d62.30_r7   d69.07_r7   d75.84_r7
            u0.00_r25   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r1    u43.98_r8   d49.60_r8   d56.29_r8   d63.01_r8   d69.72_r8
            u0.00_r26   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u0.00_r27   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u0.00_r28   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u0.00_r29   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
            u0.00_r30   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0    n0.00_r0
        );

        assert!(aligner
            .matrix
            .iter()
            .zip(expected_matrix)
            .filter(|(a, _)| a.score.is_nan().not())
            .all(|(a, b)| (b.score == 0. && a.score <= 0.)
                || ((a.score - b.score).abs() < 0.01
                    && a.dropoff == b.dropoff
                    && a.traceback == b.traceback)));

        assert_eq!(cell, [20, 20]);
    }

    #[test]
    fn align_diagonal_band_upstream() {
        let cli = Cli::dummy();
        const SEED_SCORE: Reactivity = 81.54;

        let query_slice = 0..15;
        let query = query_file::read_file(Path::new("./test_data/query.txt")).unwrap();
        let mut query = query.into_iter().next().unwrap();
        query.cap_reactivities(cli.max_reactivity);
        let query_sequence = &query.sequence()[query_slice.clone()];
        let query_reactivity = &query.reactivity()[query_slice];
        let query = EntrySlice::<_, false> {
            sequence: query_sequence,
            reactivity: query_reactivity,
        };

        let target = db_file::read_file(Path::new("./test_data/test.db")).unwrap();
        let mut target = target.into_iter().next().unwrap();
        target.cap_reactivities(cli.max_reactivity);
        let target_slice = ..773;
        let target_sequence = &target.sequence()[target_slice];
        let target_reactivity = &target.reactivity()[target_slice];
        let target = EntrySlice {
            sequence: target_sequence,
            reactivity: target_reactivity,
        };

        let mut aligner = Aligner::new(&cli);
        aligner.matrix = aligner.create_matrix(&query, &target);
        aligner.fill_borders(SEED_SCORE);

        let cell = aligner.align_diagonal_band(&query, &target);

        let expected_matrix = read_raw_matrix_results!(
            n81.54_r0   l62.54_r1  l57.54_r2  l52.54_r3  l47.54_r4  l42.54_r5  l37.54_r6  l32.54_r7  l27.54_r8  l22.54_r9  l17.54_r10  l12.54_r11  l7.54_r12  l2.54_r13  l0.00_r14  l0.00_r15
            u62.54_r1   d82.22_r0  l63.22_r1  l58.22_r2  l53.22_r3  l48.22_r4  l43.22_r5  d38.24_r7  d34.26_r8  n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0
            u57.54_r2   d63.84_r2  d83.23_r0  l64.23_r1  l59.23_r2  l54.23_r3  d50.13_r5  d44.50_r6  d38.50_r8  n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0
            u52.54_r3   d57.76_r3  u64.23_r1  d80.12_r0  d61.38_r2  d60.13_r3  d53.57_r4  d50.37_r6  d45.76_r7  n0.00_r0   n0.00_r1    n0.00_r1    n0.00_r1   n0.00_r1   n0.00_r0   n0.00_r0
            u47.54_r4   d52.89_r4  u59.23_r2  d61.40_r2  d77.55_r0  d62.15_r3  d59.69_r4  d53.93_r5  d51.75_r7  d46.85_r8  n0.00_r0    n0.00_r1    n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r0
            u42.54_r5   d46.21_r5  u54.23_r3  d53.70_r3  u58.55_r1  d79.55_r0  l60.55_r1  d58.39_r5  d54.09_r6  d51.62_r8  n0.00_r0    n0.00_r1    n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1
            u37.54_r6   d43.20_r6  u49.23_r4  d52.08_r4  u53.55_r2  u60.55_r1  d79.41_r0  d61.22_r2  d60.08_r6  d55.49_r7  n0.00_r0    n0.00_r1    n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1
            u32.54_r7   d37.88_r7  u44.23_r5  d46.38_r5  d49.49_r5  u55.55_r2  u60.41_r1  d79.77_r0  d62.60_r3  d61.17_r7  d52.85_r8   n0.00_r0    n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1
            u27.54_r8   d33.15_r8  u39.23_r6  d41.96_r6  d44.37_r6  u50.55_r3  u55.41_r2  d61.03_r2  d81.41_r0  d63.95_r4  d59.10_r8   n0.00_r0    n0.00_r1   n0.00_r1   n0.00_r1   n0.00_r1
            u22.54_r9   n0.00_r0   u34.23_r7  d39.28_r7  d42.13_r7  u45.55_r4  d51.71_r4  d57.38_r3  u62.41_r1  d82.71_r0  d64.09_r5   n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0
            u17.54_r10  n0.00_r0   u29.23_r8  d36.21_r8  d41.18_r8  u40.55_r5  d46.46_r5  d51.81_r5  u57.41_r2  u63.71_r1  d84.64_r0   d65.95_r6   l46.95_r7  l41.95_r8  n0.00_r0   n0.00_r0
            u12.54_r11  n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   u35.55_r6  d41.22_r6  d46.32_r6  u52.41_r3  u58.71_r2  u65.64_r1   d86.26_r0   d67.75_r7  l48.75_r8  n0.00_r0   n0.00_r0
            u7.54_r12   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   u30.55_r7  d36.47_r7  d41.34_r7  u47.41_r4  u53.71_r3  d60.65_r3   d67.51_r2   d88.21_r0  d69.67_r8  n0.00_r0   n0.00_r0
            u2.54_r13   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   u25.55_r8  d31.50_r8  d36.62_r8  u42.41_r5  u48.71_r4  d55.68_r4   d62.56_r4   d69.43_r3  d90.16_r0  n0.00_r0   n0.00_r0
            u0.00_r14   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   u37.41_r6  u43.71_r5  d50.50_r5   d57.41_r5   d64.46_r5  d71.20_r4  d91.85_r0  l72.85_r1
            u0.00_r15   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r1   n0.00_r0   n0.00_r0   u32.41_r7  u38.71_r6  d45.58_r6   d52.45_r6   d59.18_r6  d66.36_r6  d73.19_r5  d93.77_r0
            u0.00_r16   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r1   u27.41_r8  u33.71_r7  d40.46_r7   d47.27_r7   d54.31_r7  d60.91_r7  d68.00_r7  u74.77_r1
            u0.00_r17   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r1   n0.00_r0   u28.71_r8  d28.38_r8   d35.29_r8   d41.71_r8  d49.03_r8  d55.82_r8  u69.77_r2
            u0.00_r18   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   u64.77_r3
            u0.00_r19   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   u59.77_r4
            u0.00_r20   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   u54.77_r5
            u0.00_r21   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   u49.77_r6
            u0.00_r22   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   u44.77_r7
            u0.00_r23   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   u39.77_r8
            u0.00_r24   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0
            u0.00_r25   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0    n0.00_r0    n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0
        );

        assert_eq!(aligner.matrix.len(), expected_matrix.len());
        assert!(aligner
            .matrix
            .iter()
            .zip(expected_matrix)
            .filter(|(a, _)| a.score.is_nan().not())
            .all(|(a, b)| (b.score == 0. && a.score <= 0.)
                || ((a.score - b.score).abs() < 0.01
                    && a.dropoff == b.dropoff
                    && a.traceback == b.traceback)));

        assert_eq!(cell, [15, 15]);
    }

    #[test]
    fn align_empty_range() {
        let cli = Cli::dummy();
        let query = query_file::read_file(Path::new("./test_data/query.txt"))
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let target = db_file::read_file(Path::new("./test_data/test.db"))
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let query = &query;
        let target = &target;
        let align_tolerance = AlignTolerance {
            downstream: 10000,
            upstream: 10000,
        };
        let align_tolerance = &align_tolerance;

        let mut aligner = Aligner::new(&cli);
        // None of these must panic
        aligner.align::<NoOpBehavior>(AlignParams {
            query,
            target,
            query_range: (0..=query.reactivity().len()),
            target_range: 0..=10,
            seed_score: 10.,
            align_tolerance,
            direction: Direction::Downstream,
        });

        aligner.align::<NoOpBehavior>(AlignParams {
            query,
            target,
            query_range: 0..=10,
            target_range: 0..=target.reactivity().len(),
            seed_score: 10.,
            align_tolerance,
            direction: Direction::Downstream,
        });

        aligner.align::<NoOpBehavior>(AlignParams {
            query,
            target,
            query_range: 0..=20,
            target_range: 10..=20,
            seed_score: 10.,
            align_tolerance,
            direction: Direction::Upstream,
        });
        aligner.align::<NoOpBehavior>(AlignParams {
            query,
            target,
            query_range: 10..=20,
            target_range: 0..=20,
            seed_score: 10.,
            align_tolerance,
            direction: Direction::Upstream,
        });
    }

    #[test]
    fn seed_align_tolerance() {
        let tolerance = calc_seed_align_tolerance(36..=39, 21..=24, 20..51, 15..101, 0.5);
        assert_eq!(
            tolerance,
            AlignTolerance {
                upstream: 9,
                downstream: 17,
            }
        );

        let tolerance = calc_seed_align_tolerance(36..=39, 21..=24, 0..40, 15..101, 0.5);
        assert_eq!(
            tolerance,
            AlignTolerance {
                upstream: 9,
                downstream: 0,
            }
        );
    }

    macro_rules! cell {
        (d) => {
            Cell {
                traceback: Traceback::Diagonal,
                ..Default::default()
            }
        };

        (u) => {
            Cell {
                traceback: Traceback::Up,
                ..Default::default()
            }
        };

        (l) => {
            Cell {
                traceback: Traceback::Left,
                ..Default::default()
            }
        };
    }

    macro_rules! cells {
        ($($cell:tt)*) => {
            vec![$(cell!($cell)),*]
        };
    }

    macro_rules! base_or_gap {
        (-) => {
            BaseOrGap::Gap
        };

        (x) => {
            BaseOrGap::Base
        };
    }

    macro_rules! aln_seq {
        ($($base_or_gap:tt)*) => {
            AlignedSequence(vec![$(base_or_gap!($base_or_gap)),*])
        }
    }

    #[test]
    fn backtracking_alignment_forward() {
        let cells = cells![
            d l l l l l l
            u d l l l l l
            u u l l l l l
            u u l l l l l
            u u d l l l l
            u u u d l l l
            u u u u u u d
            u u u u u u u
        ];
        let matrix = Array2::<Cell>::from_shape_vec((8, 7), cells).unwrap();
        let alignment = BacktrackBehavior::get_alignment::<true>(&matrix, [6, 6]);
        assert_eq!(alignment.target, aln_seq!(x x x x x x - - x ));
        assert_eq!(alignment.query, aln_seq!( x x - - x x x x x ));
    }

    #[test]
    fn backtracking_alignment_backward() {
        let cells = cells![
            d l l l l l l l l
            u d l l l l l l l
            u u l l l l l l l
            u u l l l l l l l
            u u d l l l l l l
            u u u d l l l l l
            u u u u u u d l l
            u u u u u u u u l
        ];
        let matrix = Array2::<Cell>::from_shape_vec((8, 9), cells).unwrap();
        let alignment = BacktrackBehavior::get_alignment::<false>(&matrix, [6, 6]);
        assert_eq!(alignment.target, aln_seq!(x - - x x x x x x));
        assert_eq!(alignment.query, aln_seq!( x x x x x - - x x));
    }

    #[test]
    fn merge_alignment() {
        let upstream = aln_seq!(x x - x - x x - - - x - x x - - x);
        let downstream = aln_seq!(x x - x x - - x - x x x x);

        let merge = BacktrackBehavior::merge_upstream_downstream(
            upstream,
            downstream,
            NonZeroUsize::new(4).unwrap(),
        );
        assert_eq!(
            merge,
            aln_seq!(x - - x x - x - - - x x - x - x x x x x x - x x - - x - x x x x)
        );
    }

    #[test]
    fn trimmed_range_no_trimming() {
        let data = [1., 2., 3., 4., 5.];
        assert_eq!(trimmed_range(&data), 0..5);
    }

    #[test]
    fn trimmed_range_start() {
        use std::f32::NAN;
        let data = [NAN, NAN, 1., NAN, 3., 4., 5.];
        assert_eq!(trimmed_range(&data), 2..7);
    }

    #[test]
    fn trimmed_range_end() {
        use std::f32::NAN;
        let data = [1., NAN, 3., 4., 5., NAN, NAN];
        assert_eq!(trimmed_range(&data), 0..5);
    }

    #[test]
    fn trimmed_range_both() {
        use std::f32::NAN;
        let data = [NAN, NAN, NAN, 1., NAN, 3., 4., 5., NAN, NAN];
        assert_eq!(trimmed_range(&data), 3..8);
    }
}
