use std::{
    cmp, iter,
    num::NonZeroU16,
    ops::{ControlFlow, Index, Not, RangeInclusive},
    slice::{self, SliceIndex},
};

use ndarray::{s, Array2};

use crate::{
    calc_base_alignment_score,
    cli::{AlignmentArgs, Cli},
    db_file, get_sequence_base_alignment_score, query_file, Base, Reactivity, SequenceEntry,
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
    dist_from_max_score: u16,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            score: Reactivity::NAN,
            traceback: Default::default(),
            dist_from_max_score: Default::default(),
        }
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        (self.score == other.score || (self.score.is_nan() && other.score.is_nan()))
            && self.traceback == other.traceback
            && self.dist_from_max_score == other.dist_from_max_score
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlignResult {
    pub(crate) query_index: usize,
    pub(crate) _target_index: usize,
    pub(crate) score: Reactivity,
}

impl<'a> Aligner<'a> {
    pub(crate) fn new(cli: &'a Cli) -> Self {
        Self {
            cli,
            matrix: Default::default(),
        }
    }

    pub(crate) fn align(
        &mut self,
        query: &query_file::Entry,
        target: &db_file::Entry,
        query_range: RangeInclusive<usize>,
        target_range: RangeInclusive<usize>,
        seed_score: Reactivity,
        direction: Direction,
    ) -> AlignResult {
        match direction {
            Direction::Downstream => {
                let query =
                    EntrySlice::forward(query, (*query_range.end())..query.reactivity().len());
                let target =
                    EntrySlice::forward(target, (*target_range.end())..target.reactivity().len());

                let best_cell = self.handle_slices(&query, &target, seed_score);
                let _target_index = target_range.end() + best_cell[0];
                let query_index = query_range.end() + best_cell[1];
                let score = self.matrix[best_cell].score;

                AlignResult {
                    query_index,
                    _target_index,
                    score,
                }
            }

            Direction::Upstream => {
                let query = EntrySlice::backward(query, 0..(*query_range.start()));
                let target = EntrySlice::backward(target, 0..(*target_range.start()));

                let best_cell = self.handle_slices(&query, &target, seed_score);
                let _target_index = target_range.start().checked_sub(best_cell[0] + 1).unwrap();
                let query_index = query_range.end().checked_sub(best_cell[1] + 1).unwrap();
                let score = self.matrix[best_cell].score;

                AlignResult {
                    query_index,
                    _target_index,
                    score,
                }
            }
        }
    }

    fn handle_slices<'b, const FW: bool>(
        &mut self,
        query: &'b EntrySlice<'b, FW>,
        target: &'b EntrySlice<'b, FW>,
        seed_score: Reactivity,
    ) -> [usize; 2]
    where
        &'b EntrySlice<'b, FW>: IntoIterator<Item = EntryElement>,
        EntrySlice<'b, FW>: IntoIterator<Item = EntryElement> + EntrySliceExt,
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
        query: &'b EntrySlice<'b, FW>,
        target: &'b EntrySlice<'b, FW>,
    ) -> Array2<Cell>
    where
        &'b EntrySlice<'b, FW>: IntoIterator<Item = EntryElement>,
        EntrySlice<'b, FW>: IntoIterator<Item = EntryElement> + EntrySliceExt,
    {
        let rows = target.len();
        let cols = query.len();

        let size = cmp::min(rows, cols) + self.calc_band_size_from_rows_cols(rows, cols) + 1;
        Array2::default([size.min(rows), size.min(cols)])
    }

    fn fill_borders(&mut self, seed_score: Reactivity) {
        let AlignmentArgs {
            align_gap_open_penalty,
            align_gap_ext_penalty,
            ..
        } = self.cli.alignment_args;

        self.matrix[(0, 0)].score = seed_score;
        let first_open_score = seed_score + align_gap_open_penalty;

        let make_cell_handler = |mut score, traceback| {
            move |(index, cell): (usize, &mut Cell)| {
                score += align_gap_ext_penalty;
                *cell = Cell {
                    score,
                    dist_from_max_score: index.try_into().unwrap_or(u16::MAX),
                    traceback,
                };
            }
        };

        let useful_cells = self.calc_band_size() + 2;

        self.matrix
            .row_mut(0)
            .iter_mut()
            .enumerate()
            .take(useful_cells)
            .skip(1)
            .for_each(make_cell_handler(first_open_score, Traceback::Left));

        self.matrix
            .column_mut(0)
            .iter_mut()
            .enumerate()
            .take(useful_cells)
            .skip(1)
            .for_each(make_cell_handler(first_open_score, Traceback::Up));
    }

    fn align_diagonal_band<'b, const FW: bool>(
        &mut self,
        query: &'b EntrySlice<'b, FW>,
        target: &'b EntrySlice<'b, FW>,
    ) -> [usize; 2]
    where
        &'b EntrySlice<'b, FW>: IntoIterator<Item = EntryElement>,
        EntrySlice<'b, FW>: IntoIterator<Item = EntryElement> + EntrySliceExt,
    {
        #[derive(Debug)]
        struct State {
            best_score: Reactivity,
            dropoff_score: Reactivity,
            dist_from_max_score: Option<NonZeroU16>,
            best_bad_score: Reactivity,
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
                    dist_from_max_score: None,
                    best_bad_score: Reactivity::NAN,
                }
            }
        }

        let rows = target.len();
        let cols = query.len();
        let band_size = self.calc_band_size();
        let diag_len = cmp::min(rows, cols);
        let alignment_args = &self.cli.alignment_args;
        let initial_score = self.matrix[(0, 0)].score;
        let state = State::new(initial_score, [0, 0], alignment_args);

        let calc_alignment_score = if alignment_args.align_score_seq {
            |query: &EntryElement, target: &EntryElement, cli: &Cli| {
                calc_base_alignment_score(query.reactivity, target.reactivity, cli)
                    + get_sequence_base_alignment_score(
                        query.base,
                        target.base,
                        &cli.alignment_args,
                    )
            }
        } else {
            |query: &EntryElement, target: &EntryElement, cli: &Cli| {
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
            } else {
                cell.dist_from_max_score = parent_cell.dist_from_max_score + 1;
                let cell_max_score = cell
                    .dist_from_max_score
                    .try_into()
                    .expect("dist from max score cannot be 0");

                state.best_bad_score = state
                    .best_bad_score
                    .is_nan()
                    .then(|| cell.score)
                    .unwrap_or_else(|| {
                        cmp::max_by(state.best_bad_score, cell.score, |a, b| {
                            a.partial_cmp(b).unwrap()
                        })
                    });

                state.dist_from_max_score = Some(
                    state
                        .dist_from_max_score
                        .map(|max_score| max_score.min(cell_max_score))
                        .unwrap_or(cell_max_score),
                );
            }
        };

        let align_result = (0..(diag_len + band_size)).zip(target).skip(1).try_fold(
            state,
            |mut state, (row_index, target)| {
                let band_range_start = row_index.saturating_sub(band_size).max(1);
                let band_range_end = (row_index + band_size).min(cols);
                dbg!(band_range_start, band_range_end);
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
                let query_band = query.slice(band_range_start..band_range_end);

                let (_, mut state, _final_col_index) = band
                    .iter_mut()
                    .zip(query_band)
                    .zip(above.windows(2).into_iter())
                    .fold(
                        (left, state, band_range_start),
                        |(left, mut state, col_index), ((cell, query), above)| {
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

                            (cell, state, col_index + 1)
                        },
                    );
                debug_assert_eq!(_final_col_index, band_range_end);

                dbg!(&state);
                if state.best_bad_score <= 0.
                    || (state.best_bad_score < state.dropoff_score
                        && state.dist_from_max_score.map(NonZeroU16::get).unwrap_or(0)
                            > alignment_args.align_max_drop_off_bases)
                {
                    ControlFlow::Break(state.best_cell)
                } else {
                    if band_range_end < cols {
                        let query_base = query.get(band_range_end);

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
struct EntrySlice<'a, const FW: bool> {
    sequence: &'a [Base],
    reactivity: &'a [Reactivity],
}

impl<'a, const FW: bool> EntrySlice<'a, FW> {
    fn new<E, R>(entry: &'a E, range: R) -> Self
    where
        E: SequenceEntry,
        R: SliceIndex<[Base], Output = [Base]>
            + SliceIndex<[Reactivity], Output = [Reactivity]>
            + Clone,
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

    fn slice<I>(&self, index: I) -> Self
    where
        I: Clone,
        [Base]: Index<I, Output = [Base]>,
        [Reactivity]: Index<I, Output = [Reactivity]>,
    {
        let sequence = &self.sequence[index.clone()];
        let reactivity = &self.reactivity[index];

        Self {
            sequence,
            reactivity,
        }
    }
}

impl<'a> EntrySlice<'a, true> {
    #[inline]
    fn forward<E, R>(entry: &'a E, range: R) -> Self
    where
        E: SequenceEntry,
        R: SliceIndex<[Base], Output = [Base]>
            + SliceIndex<[Reactivity], Output = [Reactivity]>
            + Clone,
    {
        Self::new(entry, range)
    }
}

impl<'a> EntrySlice<'a, false> {
    #[inline]
    fn backward<E, R>(entry: &'a E, range: R) -> Self
    where
        E: SequenceEntry,
        R: SliceIndex<[Base], Output = [Base]>
            + SliceIndex<[Reactivity], Output = [Reactivity]>
            + Clone,
    {
        Self::new(entry, range)
    }
}

trait EntrySliceExt {
    fn get(&self, index: usize) -> EntryElement;
}

impl EntrySliceExt for EntrySlice<'_, true> {
    fn get(&self, index: usize) -> EntryElement {
        let base = self.sequence[index];
        let reactivity = self.reactivity[index];

        EntryElement { base, reactivity }
    }
}

impl EntrySliceExt for EntrySlice<'_, false> {
    fn get(&self, index: usize) -> EntryElement {
        let len = self.len();
        let index = len - index;
        let base = self.sequence[index];
        let reactivity = self.reactivity[index];

        EntryElement { base, reactivity }
    }
}

impl<'a> IntoIterator for &'a EntrySlice<'a, true> {
    type Item = EntryElement;
    type IntoIter = EntryIterFw<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter();
        let reactivity = self.reactivity.iter();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

impl<'a> IntoIterator for &'a EntrySlice<'a, false> {
    type Item = EntryElement;
    type IntoIter = EntryIterBw<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter().rev();
        let reactivity = self.reactivity.iter().rev();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

impl<'a> IntoIterator for EntrySlice<'a, true> {
    type Item = EntryElement;
    type IntoIter = EntryIterFw<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter();
        let reactivity = self.reactivity.iter();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

impl<'a> IntoIterator for EntrySlice<'a, false> {
    type Item = EntryElement;
    type IntoIter = EntryIterBw<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let sequence = self.sequence.iter().rev();
        let reactivity = self.reactivity.iter().rev();

        Self::IntoIter {
            sequence,
            reactivity,
        }
    }
}

struct EntryIterFw<'a> {
    sequence: slice::Iter<'a, Base>,
    reactivity: slice::Iter<'a, Reactivity>,
}

struct EntryIterBw<'a> {
    sequence: iter::Rev<slice::Iter<'a, Base>>,
    reactivity: iter::Rev<slice::Iter<'a, Reactivity>>,
}

#[derive(Debug)]
struct EntryElement {
    base: Base,
    reactivity: Reactivity,
}

impl<'a> Iterator for EntryIterFw<'a> {
    type Item = EntryElement;

    fn next(&mut self) -> Option<Self::Item> {
        let base = *self.sequence.next()?;
        let reactivity = *self.reactivity.next()?;

        Some(Self::Item { base, reactivity })
    }
}

impl<'a> Iterator for EntryIterBw<'a> {
    type Item = EntryElement;

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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::StructOpt;
    use ndarray::array;

    use super::*;

    fn dummy_cli() -> Cli {
        Cli::parse_from(["test", "--database", "test", "--query", "test"])
    }

    #[test]
    fn fill_borders_small() {
        let cli = dummy_cli();
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

            ($score:expr, $distance:literal, up) => {
                Cell {
                    score: $score,
                    dist_from_max_score: $distance,
                    traceback: Traceback::Up,
                }
            };

            ($score:expr, $distance:literal, left) => {
                Cell {
                    score: $score,
                    dist_from_max_score: $distance,
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
        let mut cli = dummy_cli();
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
        let cli = dummy_cli();
        const SEED_SCORE: Reactivity = 4.5;
        let AlignmentArgs {
            align_gap_open_penalty,
            align_gap_ext_penalty,
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
                dist_from_max_score: 0
            }
        );
        let first_row = aligner.matrix.row(0);
        first_row
            .iter()
            .enumerate()
            .skip(1)
            .take(MIN_BAND_SIZE + 1)
            .for_each(|(col, cell)| {
                assert_eq!(
                    cell.score,
                    SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * col as Reactivity
                );
                assert_eq!(cell.traceback, Traceback::Left);
                assert_eq!(usize::from(cell.dist_from_max_score), col);
            });

        first_row.iter().skip(MIN_BAND_SIZE + 2).for_each(|cell| {
            assert_eq!(
                cell,
                &Cell {
                    score: Reactivity::NAN,
                    traceback: Traceback::Diagonal,
                    dist_from_max_score: 0,
                }
            );
        });

        let first_col = aligner.matrix.column(0);
        first_col
            .iter()
            .enumerate()
            .skip(1)
            .take(MIN_BAND_SIZE + 1)
            .for_each(|(row, cell)| {
                assert_eq!(
                    cell.score,
                    SEED_SCORE + align_gap_open_penalty + align_gap_ext_penalty * row as Reactivity
                );
                assert_eq!(cell.traceback, Traceback::Up);
                assert_eq!(usize::from(cell.dist_from_max_score), row);
            });

        first_col.iter().skip(MIN_BAND_SIZE + 2).for_each(|cell| {
            assert_eq!(
                cell,
                &Cell {
                    score: Reactivity::NAN,
                    traceback: Traceback::Diagonal,
                    dist_from_max_score: 0,
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
            dist_from_max_score,
        }
    }

    const QUERY_SEQUENCE: &str = "TGACGCTCAGGTGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGTCGACTTGGAGGTTGTGCCCTTGAGGCGT";
    #[allow(clippy::approx_constant)]
    const QUERY_REACTIVITY: [Reactivity; 100] = [
        0.102, 0.083, 0.066, 0.040, 0.075, 0.061, 0.573, 0.631, 0.427, 0.265, 1.190, 0.066, 0.042,
        0.085, 0.424, 0.413, 0.375, 0.447, 0.035, 0.045, 0.037, 0.242, 0.221, 0.157, 0.170, 0.370,
        1.238, 0.743, 0.571, 0.138, 0.837, 0.859, 0.042, 0.021, 0.080, 0.318, 0.195, 0.792, 1.581,
        1.058, 2.004, 1.512, 2.273, 1.256, 0.036, 0.005, 0.094, 0.091, 0.464, 0.741, 0.667, 0.367,
        0.428, 0.162, 0.020, 0.000, 0.046, 0.044, 0.114, 0.054, 0.101, 1.192, 1.264, 0.104, 0.623,
        0.937, 1.593, 1.279, 0.599, 1.695, 0.072, 0.030, 0.002, 0.030, 0.094, 0.120, 0.332, 1.424,
        0.173, 0.100, 0.513, 0.266, 0.276, 0.146, 0.229, 0.271, 0.436, 0.846, 0.093, 0.160, 0.552,
        1.456, 5.895, 1.110, 2.465, 1.198, 0.055, 0.094, 0.073, 0.061,
    ];

    #[test]
    #[ignore = "we need the matrix produced with the same algorithm"]
    fn align_diagonal_band_downstream() {
        let cli = dummy_cli();
        const SEED_SCORE: Reactivity = 93.77;

        let query_slice = 79..;
        let query_sequence: Vec<_> = QUERY_SEQUENCE[query_slice.clone()]
            .bytes()
            .map(|b| Base::try_from(b).unwrap())
            .collect();

        let query_reactivity: Vec<_> = QUERY_REACTIVITY
            .into_iter()
            .map(|x| x.min(cli.max_reactivity))
            .collect();
        let query = EntrySlice::<true> {
            sequence: &query_sequence,
            reactivity: &query_reactivity[query_slice],
        };

        let target =
            db_file::read_file(Path::new("./test_data/test.db"), cli.max_reactivity).unwrap();
        let target = target.into_iter().next().unwrap();
        let target_slice = 837..;
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
            n93.77_r0 n74.77_r0 n69.77_r0 n64.77_r0 n59.77_r0 n54.77_r0 n49.77_r0 n44.77_r0 n39.77_r0 n34.77_r0 n29.77_r0 n24.77_r0 n19.77_r0 n14.77_r0 n9.77_r0  n4.77_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n74.77_r0 d93.55_r0 d75.79_r0 d70.74_r1 d66.39_r1 d60.97_r1 d55.76_r1 d49.94_r1 d41.23_r1 d41.65_r1 d36.32_r1 d29.36_r1 n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n69.77_r0 d76.17_r0 d93.72_r0 d76.01_r0 d70.31_r2 d66.37_r2 d61.17_r2 d56.78_r2 d50.87_r2 d40.30_r2 d41.29_r2 d37.92_r2 d29.52_r2 n0.00_r0  n0.00_r0  n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n64.77_r0 d71.23_r1 d77.47_r0 d95.07_r0 d76.71_r0 d71.42_r3 d67.70_r3 d63.02_r3 d56.58_r3 d51.30_r3 d41.07_r3 d42.56_r3 d36.38_r3 d27.98_r3 n0.00_r0  n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n59.77_r0 d66.49_r1 d71.72_r2 d78.01_r0 d94.96_r0 d77.01_r0 d71.94_r4 d69.04_r4 d63.63_r4 d56.21_r4 d51.26_r4 d42.99_r4 d42.40_r4 d36.22_r4 d27.82_r4 n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n54.77_r0 d60.73_r1 d68.29_r2 d73.57_r3 d79.21_r0 d96.58_r0 d78.84_r0 d73.29_r5 d68.10_r5 d64.56_r5 d57.48_r5 d52.03_r5 d40.36_r5 d39.77_r5 d33.59_r5 d25.19_r5  n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n49.77_r0 d56.67_r1 d61.60_r2 d69.21_r3 d73.84_r4 d79.89_r0 d97.47_r0 d80.55_r0 d73.52_r6 d68.10_r6 d64.90_r6 d59.18_r6 d51.46_r6 d39.78_r6 d39.19_r6 d33.01_r6  d24.61_r6  n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n44.77_r0 d50.34_r1 d58.48_r2 d63.36_r3 d70.80_r4 d75.83_r5 d81.68_r0 d98.43_r0 l79.43_r0 d74.85_r7 d69.76_r7 d65.28_r7 d55.69_r7 d47.96_r7 d36.29_r7 d35.70_r7  d29.52_r7  d25.75_r7  n0.00_r0   n0.00_r0  n0.00_r0
            n39.77_r0 d45.31_r1 d52.12_r2 d60.21_r3 d64.98_r4 d72.76_r5 d77.59_r6 d82.61_r0 d96.56_r0 d80.78_r0 d76.54_r8 d70.11_r8 d61.72_r8 d52.13_r8 d44.41_r8 d32.73_r8  d32.14_r8  d30.69_r8  d27.11_r8  n0.00_r0  n0.00_r0
            n34.77_r0 d41.02_r1 d46.82_r2 d53.68_r3 d61.12_r4 d66.30_r5 d74.30_r6 d79.23_r0 d82.20_r0 d97.21_r0 d81.76_r0 n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0   n0.00_r0  n0.00_r0
            n29.77_r0 d34.33_r1 d37.95_r2 d43.86_r3 d49.29_r4 d57.64_r5 d63.28_r6 d73.09_r7 d80.46_r0 u78.21_r1 d92.97_r0 d81.52_r0 l62.52_r1 l57.52_r2 l52.52_r3 l47.52_r4  l42.52_r5  l37.52_r6  l32.52_r7  l27.52_r8 n0.00_r0
            n24.77_r0 d29.59_r1 d31.82_r2 d35.55_r3 d40.03_r4 d46.37_r5 d55.18_r6 d62.64_r7 d74.58_r8 d76.04_r1 d74.53_r2 d92.98_r0 d83.27_r0 d64.27_r2 d59.27_r3 d54.27_r4  d49.27_r5  d37.69_r6  d33.12_r7  d27.89_r8 n0.00_r0
            n19.77_r0 n0.00_r0  d26.51_r2 d28.86_r3 d31.15_r4 d36.54_r5 d43.35_r6 d53.97_r7 d63.87_r8 n0.00_r0  d71.80_r2 d74.29_r3 d94.98_r0 d85.27_r0 d66.27_r3 d61.27_r4  d56.27_r5  d43.87_r6  d32.72_r7  d27.92_r8 n0.00_r0
            n14.77_r0 n0.00_r0  n0.00_r0  d23.55_r3 d24.46_r4 d27.67_r5 d33.53_r6 d42.15_r7 d55.20_r8 n0.00_r0  u52.80_r3 d71.56_r3 d76.29_r4 d96.98_r0 d87.27_r0 d68.27_r4  d63.27_r5  d50.87_r6  d38.91_r7  d27.53_r8 n0.00_r0
            n9.77_r0  n0.00_r0  n0.00_r0  n0.00_r0  d19.16_r4 d20.98_r5 d24.65_r6 d32.32_r7 d43.38_r8 n0.00_r0  u47.80_r4 u52.56_r4 d73.56_r4 d78.29_r5 d98.98_r0 d89.27_r0  d70.27_r5  d57.87_r6  d45.91_r7  d33.71_r8 n0.00_r0
            n4.77_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d15.67_r5 d17.96_r6 d23.45_r7 d33.55_r8 n0.00_r0  u42.80_r5 u47.56_r5 d54.56_r5 d75.56_r5 d80.29_r0 d100.98_r0 d91.27_r0  l72.27_r1  l67.27_r2  l62.27_r3 l57.27_r4
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d12.66_r6 d16.76_r7 d24.68_r8 n0.00_r0  u37.80_r6 u42.56_r6 d49.56_r6 d56.56_r6 d77.56_r6 d82.29_r0  d102.98_r0 d85.87_r0  d67.30_r2  d62.07_r3 d56.94_r4
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d13.49_r7 d14.68_r8 n0.00_r0  u32.80_r7 d38.06_r7 d38.79_r7 d45.79_r7 u58.56_r7 d73.79_r7  u83.98_r0  d104.24_r0 d87.33_r0  d68.65_r3 d63.36_r4
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d9.18_r8  n0.00_r0  u27.80_r8 d31.73_r8 d32.06_r8 d32.79_r8 u53.56_r8 u54.79_r8  u78.98_r1  d85.71_r0  d105.77_r0 d88.96_r0 d70.35_r4
        );

        for row in aligner.matrix.rows() {
            for col in row {
                print!("{:>6.3} ", col.score);
            }
            println!();
        }

        assert!(aligner
            .matrix
            .iter()
            .zip(expected_matrix)
            // .inspect(|(a, b)| {
            //     dbg!(a, b);
            // })
            .filter(|(a, _)| a.score.is_nan().not())
            .all(|(a, b)| (a.score - b.score).abs() < 0.01
                /* && a.traceback == b.traceback */
                /* && a.dist_from_max_score == b.dist_from_max_score */));

        assert_eq!(cell, [20, 20]);
    }

    #[test]
    #[ignore = "we need the matrix produced with the same algorithm"]
    fn align_diagonal_band_upstream() {
        let cli = dummy_cli();
        const SEED_SCORE: Reactivity = 81.54;

        let query_slice = 0..15;
        let query_sequence: Vec<_> = QUERY_SEQUENCE[query_slice.clone()]
            .bytes()
            .map(|b| Base::try_from(b).unwrap())
            .collect();

        let query_reactivity: Vec<_> = QUERY_REACTIVITY
            .into_iter()
            .map(|x| x.min(cli.max_reactivity))
            .collect();
        let query = EntrySlice::<false> {
            sequence: &query_sequence,
            reactivity: &query_reactivity[query_slice],
        };

        let target =
            db_file::read_file(Path::new("./test_data/test.db"), cli.max_reactivity).unwrap();
        let target = target.into_iter().next().unwrap();
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
            n81.54_r0 n62.54_r0 n57.54_r0 n52.54_r0 n47.54_r0 n42.54_r0 n37.54_r0 n32.54_r0 n27.54_r0 n22.54_r0 n17.54_r0 n12.54_r0 n7.54_r0  n2.54_r0  n0.00_r0  n0.00_r0
            n62.54_r0 d82.22_r0 l63.22_r1 l58.22_r2 l53.22_r3 l48.22_r4 l43.22_r5 d38.24_r1 d34.26_r1 d28.97_r1 d20.65_r1 d15.80_r1 n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0
            n57.54_r0 d63.84_r1 d83.23_r0 l64.23_r1 l59.23_r2 l54.23_r3 d50.13_r5 d44.50_r6 d38.50_r2 d34.81_r2 d29.86_r2 d21.61_r2 d16.59_r2 n0.00_r0  n0.00_r0  n0.00_r0
            n52.54_r0 d57.76_r1 u64.23_r1 d80.12_r0 d61.38_r2 d60.13_r3 d53.57_r4 d50.37_r6 d45.76_r7 d39.46_r3 d31.90_r3 d27.10_r3 d18.47_r3 d13.73_r3 n0.00_r0  n0.00_r0
            n47.54_r0 d52.89_r1 u59.23_r2 d61.40_r2 d77.55_r0 d62.15_r3 d59.69_r4 d53.93_r5 d51.75_r7 d46.85_r8 d36.84_r4 d29.43_r4 d24.25_r4 d15.90_r4 d11.35_r4 n0.00_r0
            n42.54_r0 d46.21_r1 u54.23_r3 d53.70_r3 u58.55_r1 d79.55_r0 l60.55_r1 d58.39_r5 d54.09_r6 d51.62_r8 n0.00_r0  d31.67_r5 d23.87_r5 d18.98_r5 d10.81_r5 d6.47_r5
            n37.54_r0 d43.20_r1 u49.23_r4 d52.08_r4 u53.55_r2 u60.55_r1 d79.41_r0 d61.22_r2 d60.08_r6 d55.49_r7 n0.00_r0  u12.67_r6 d29.49_r6 d21.98_r6 d17.28_r6 d9.32_r6
            n32.54_r0 d37.88_r1 u44.23_r5 d46.38_r5 d49.49_r5 u55.55_r2 u60.41_r1 d79.77_r0 d62.60_r3 d61.17_r7 d52.85_r8 n0.00_r0  u10.49_r7 d26.90_r7 d19.58_r7 d15.08_r7
            n27.54_r0 d33.15_r1 u39.23_r6 d41.96_r6 d44.37_r6 u50.55_r3 u55.41_r2 d61.03_r2 d81.41_r0 d63.95_r4 d59.10_r8 n0.00_r0  u5.49_r8  d8.49_r8  d25.08_r8 d17.97_r8
            n22.54_r0 d29.50_r1 u34.23_r7 d39.28_r7 d42.13_r7 u45.55_r4 d51.71_r4 d57.38_r3 u62.41_r1 d82.71_r0 d64.09_r5 n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0
            n17.54_r0 d22.66_r1 d31.31_r2 d36.21_r8 d41.18_r8 u40.55_r5 d46.46_r5 d51.81_r5 u57.41_r2 u63.71_r1 d84.64_r0 d65.95_r6 l46.95_r7 l41.95_r8 n0.00_r0  n0.00_r0
            n12.54_r0 d17.42_r1 d24.23_r2 d33.10_r3 n0.00_r0  u35.55_r6 d41.22_r6 d46.32_r6 u52.41_r3 u58.71_r2 u65.64_r1 d86.26_r0 d67.75_r7 l48.75_r8 n0.00_r0  n0.00_r0
            n7.54_r0  n0.00_r0  d19.25_r2 d26.19_r3 d35.02_r4 u30.55_r7 d36.47_r7 d41.34_r7 u47.41_r4 u53.71_r3 d60.65_r3 d67.51_r2 d88.21_r0 d69.67_r8 n0.00_r0  n0.00_r0
            n2.54_r0  n0.00_r0  n0.00_r0  d21.18_r3 d28.14_r4 d29.63_r5 d31.50_r8 d36.62_r8 u42.41_r5 u48.71_r4 d55.68_r4 d62.56_r4 d69.43_r3 d90.16_r0 n0.00_r0  n0.00_r0
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d22.95_r4 d22.36_r5 d30.41_r6 n0.00_r0  u37.41_r6 u43.71_r5 d50.50_r5 d57.41_r5 d64.46_r5 d71.20_r4 d91.85_r0 l72.85_r1
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d17.89_r5 d23.47_r6 d30.70_r7 u32.41_r7 u38.71_r6 d45.58_r6 d52.45_r6 d59.18_r6 d66.36_r6 d73.19_r5 d93.77_r0
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d18.63_r6 d23.39_r7 d28.90_r8 u33.71_r7 d40.46_r7 d47.27_r7 d54.31_r7 d60.91_r7 d68.00_r7 u74.77_r1
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d17.33_r7 d23.55_r8 n0.00_r0  d28.38_r8 d35.29_r8 d41.71_r8 d49.03_r8 d55.82_r8 u69.77_r2
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  d19.06_r8 n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  u64.77_r3
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  u59.77_r4
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  u54.77_r5
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  u49.77_r6
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  u44.77_r7
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  u39.77_r8
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0
            n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0  n0.00_r0
        );

        for row in aligner.matrix.rows() {
            for col in row {
                print!("{:>6.3} ", col.score);
            }
            println!();
        }

        assert!(aligner
            .matrix
            .iter()
            .zip(expected_matrix)
            // .inspect(|(a, b)| {
            //     dbg!(a, b);
            // })
            .filter(|(a, _)| a.score.is_nan().not())
            .all(|(a, b)| (a.score - b.score).abs() < 0.01
                /* && a.traceback == b.traceback */
                /* && a.dist_from_max_score == b.dist_from_max_score */));

        assert_eq!(cell, [15, 15]);
    }
}
