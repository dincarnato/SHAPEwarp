use core::slice;
use std::{
    marker::PhantomData,
    mem,
    num::NonZeroUsize,
    ops::{self, Not, Range, RangeInclusive},
    sync::Arc,
};

use crate::{
    aligner::{
        calc_seed_align_tolerance, trimmed_range, AlignBehavior, AlignParams, Aligner,
        AlignmentResult, Direction,
    },
    calc_seed_alignment_score,
    cli::Cli,
    db_file, get_matching_kmers, group_matching_kmers, query_file, DbData, DbEntryMatches,
    MatchRanges, Reactivity, SequenceEntry,
};

pub(crate) fn align_query_to_target_db<'a, 'cli, Behavior>(
    query_entry: &'a query_file::Entry,
    db_entries: &'a [db_file::Entry],
    query_results: &mut Vec<DbEntryMatches<'a>>,
    cli: &'cli Cli,
) -> anyhow::Result<QueryAligner<'a, 'cli, Behavior>> {
    query_results.clear();
    for db_entry in db_entries {
        let db_file::Entry {
            sequence,
            reactivity,
            ..
        } = db_entry;

        let db_data = DbData::new(sequence, reactivity)?;
        let matching_kmers = get_matching_kmers(
            query_entry.reactivity(),
            query_entry.sequence(),
            &db_data,
            cli,
        )?;
        let grouped = group_matching_kmers(&matching_kmers, cli);

        if grouped.is_empty().not() {
            query_results.push(DbEntryMatches {
                db_entry,
                matches: grouped,
            });
        }
    }

    Ok(QueryAligner {
        query_entry,
        cli,
        _marker: PhantomData,
    })
}

pub(crate) struct QueryAligner<'a, 'cli, Behavior> {
    query_entry: &'a query_file::Entry,
    cli: &'cli Cli,
    _marker: PhantomData<Behavior>,
}

impl<'a, 'cli, Behavior> QueryAligner<'a, 'cli, Behavior> {
    pub(crate) fn into_iter<'aln>(
        self,
        query_results: &'a [DbEntryMatches<'a>],
        aligner: &'aln mut Aligner<'cli>,
    ) -> QueryAlignIterator<'a, 'cli, 'aln, Behavior> {
        let Self {
            query_entry,
            cli,
            _marker,
        } = self;

        let query_results = query_results.iter();
        QueryAlignIterator(QueryAlignIteratorEnum::Empty {
            query_results,
            query_entry,
            cli,
            aligner,
        })
    }
}

pub(crate) struct QueryAlignIterator<'a, 'cli, 'aln, Behavior>(
    QueryAlignIteratorEnum<'a, 'cli, 'aln, Behavior>,
);

enum QueryAlignIteratorEnum<'a, 'cli, 'aln, Behavior> {
    Empty {
        query_results: slice::Iter<'a, DbEntryMatches<'a>>,
        query_entry: &'a query_file::Entry,
        cli: &'cli Cli,
        aligner: &'aln mut Aligner<'cli>,
    },
    Full {
        query_results: slice::Iter<'a, DbEntryMatches<'a>>,
        iter: QueryAlignIteratorInner<'a, 'cli, 'aln, Behavior>,
        query_entry: &'a query_file::Entry,
        cli: &'cli Cli,
    },
    Finished,
}

impl<'a, 'cli, 'aln, Behavior> QueryAlignIterator<'a, 'cli, 'aln, Behavior> {
    fn make_new_iter(&mut self) -> Option<&mut QueryAlignIteratorInner<'a, 'cli, 'aln, Behavior>> {
        match mem::replace(&mut self.0, QueryAlignIteratorEnum::Finished) {
            QueryAlignIteratorEnum::Empty {
                query_results,
                query_entry,
                cli,
                aligner,
            } => self.create_new_state(query_results, query_entry, cli, aligner),
            QueryAlignIteratorEnum::Full {
                query_results,
                iter,
                query_entry,
                cli,
            } => {
                let aligner = iter.aligner;
                self.create_new_state(query_results, query_entry, cli, aligner)
            }
            QueryAlignIteratorEnum::Finished => None,
        }
    }

    fn create_new_state(
        &mut self,
        mut query_results: slice::Iter<'a, DbEntryMatches<'a>>,
        query_entry: &'a query_file::Entry,
        cli: &'cli Cli,
        aligner: &'aln mut Aligner<'cli>,
    ) -> Option<&mut QueryAlignIteratorInner<'a, 'cli, 'aln, Behavior>> {
        query_results.next().map(|query_result| {
            let &DbEntryMatches {
                db_entry,
                matches: ref db,
            } = query_result;

            let iter = QueryAlignIteratorInner {
                aligner,
                db_iter: db.iter(),
                query_entry,
                db_entry,
                cli,
                _marker: PhantomData,
            };

            self.0 = QueryAlignIteratorEnum::Full {
                query_results,
                iter,
                query_entry,
                cli,
            };

            match &mut self.0 {
                QueryAlignIteratorEnum::Full { iter, .. } => iter,
                _ => unreachable!(),
            }
        })
    }

    #[inline]
    fn get_next_from_new_iter(&mut self) -> Option<QueryAlignResult<'a, Behavior::Alignment>>
    where
        Behavior: AlignBehavior,
        <Behavior as AlignBehavior>::Alignment: std::fmt::Debug,
    {
        loop {
            let next = self.make_new_iter()?.next();
            if next.is_some() {
                break next;
            }
        }
    }
}

impl<'a, 'cli, 'aln, Behavior> Iterator for QueryAlignIterator<'a, 'cli, 'aln, Behavior>
where
    Behavior: AlignBehavior,
    <Behavior as AlignBehavior>::Alignment: std::fmt::Debug,
{
    type Item = QueryAlignResult<'a, Behavior::Alignment>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            QueryAlignIteratorEnum::Empty { .. } => self.get_next_from_new_iter(),
            QueryAlignIteratorEnum::Full { iter, .. } => match iter.next() {
                Some(item) => Some(item),
                None => self.get_next_from_new_iter(),
            },
            QueryAlignIteratorEnum::Finished => None,
        }
    }
}

struct QueryAlignIteratorInner<'a, 'cli, 'aln, Behavior> {
    aligner: &'aln mut Aligner<'cli>,
    db_iter: slice::Iter<'a, MatchRanges>,
    query_entry: &'a query_file::Entry,
    db_entry: &'a db_file::Entry,
    cli: &'cli Cli,
    _marker: PhantomData<Behavior>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QueryAlignResult<'a, Alignment> {
    pub(crate) db_entry: &'a db_file::Entry,
    pub(crate) db_match: MatchRanges,
    pub(crate) score: Reactivity,
    pub(crate) db: ops::RangeInclusive<usize>,
    pub(crate) query: ops::RangeInclusive<usize>,
    pub(crate) alignment: Arc<AlignmentResult<Alignment>>,
}

impl<'a, 'cli, 'aln, Behavior> Iterator for QueryAlignIteratorInner<'a, 'cli, 'aln, Behavior>
where
    Behavior: AlignBehavior,
    <Behavior as AlignBehavior>::Alignment: std::fmt::Debug,
{
    type Item = QueryAlignResult<'a, Behavior::Alignment>;

    fn next(&mut self) -> Option<Self::Item> {
        let &mut Self {
            ref mut aligner,
            ref mut db_iter,
            query_entry,
            db_entry,
            cli,
            _marker,
        } = self;

        loop {
            let db_match = db_iter.next()?;
            let seed_score = calc_seed_alignment_score(
                query_entry,
                db_entry,
                db_match.query.clone(),
                db_match.db.clone(),
                cli,
            );

            if seed_score <= 0. {
                continue;
            }

            let result = handle_match::<Behavior>(
                db_match.clone(),
                query_entry,
                db_entry,
                seed_score,
                aligner,
                cli,
            );
            break Some(result);
        }
    }
}

fn handle_match<'db, 'cli, Behavior>(
    db_match: MatchRanges,
    query_entry: &query_file::Entry,
    db_entry: &'db db_file::Entry,
    seed_score: f32,
    aligner: &mut Aligner<'cli>,
    cli: &'cli Cli,
) -> QueryAlignResult<'db, Behavior::Alignment>
where
    Behavior: AlignBehavior,
    <Behavior as AlignBehavior>::Alignment: std::fmt::Debug,
{
    let MatchRanges { db, query } = db_match.clone();
    let seed_length = db.end() - db.start() + 1;
    debug_assert_eq!(seed_length, query.end() - query.start() + 1);
    let seed_length = NonZeroUsize::new(seed_length)
        .expect("seed must have a length greater than zero (and more)");

    let trimmed_query_range = trimmed_range(query_entry.reactivity());
    let trimmed_db_range = trimmed_range(db_entry.reactivity());
    let query_len = trimmed_query_range.len();
    let query = intersect_range(query, trimmed_query_range.clone());
    let db = intersect_range(db, trimmed_db_range.clone());

    let align_tolerance = calc_seed_align_tolerance(
        query.clone(),
        db.clone(),
        trimmed_query_range,
        trimmed_db_range,
        cli.alignment_args.align_len_tolerance,
    );
    let align_tolerance = &align_tolerance;

    let upstream_result = aligner.align::<Behavior>(AlignParams {
        query: query_entry,
        target: db_entry,
        query_range: query.clone(),
        target_range: db.clone(),
        seed_score,
        align_tolerance,
        direction: Direction::Upstream,
    });

    let downstream_result = aligner.align::<Behavior>(AlignParams {
        query: query_entry,
        target: db_entry,
        query_range: query,
        target_range: db,
        seed_score: upstream_result.score,
        align_tolerance,
        direction: Direction::Downstream,
    });

    let query = upstream_result.query_index..=downstream_result.query_index;
    let db = upstream_result.target_index..=downstream_result.target_index;
    let aligned_query_len = downstream_result.query_index + 1 - upstream_result.query_index;
    let score = downstream_result.score as f64
        * ((aligned_query_len as f64).ln() / (query_len as f64).ln());
    let score = score as Reactivity;

    let query_alignment = Behavior::merge_upstream_downstream(
        upstream_result.query_alignment,
        downstream_result.query_alignment,
        seed_length,
    );
    let target_alignment = Behavior::merge_upstream_downstream(
        upstream_result.target_alignment,
        downstream_result.target_alignment,
        seed_length,
    );
    let alignment = Arc::new(AlignmentResult {
        query: query_alignment,
        target: target_alignment,
    });

    QueryAlignResult {
        db_entry,
        db_match,
        score,
        db,
        query,
        alignment,
    }
}

#[inline]
fn intersect_range(a: RangeInclusive<usize>, b: Range<usize>) -> RangeInclusive<usize> {
    let start = *a.start().max(&b.start);
    let end = *a.end().min(&b.end.saturating_sub(1));
    start..=end
}
