use core::slice;
use std::{
    mem,
    ops::{self, Not},
};

use crate::{
    aligner::{calc_seed_align_tolerance, AlignParams, Aligner, Direction},
    calc_seed_alignment_score,
    cli::Cli,
    db_file, get_matching_kmers, group_matching_kmers, query_file, DbData, DbEntryMatches,
    MatchRanges, Reactivity, SequenceEntry,
};

pub(crate) fn align_query_to_target_db<'q, 'db, 'cli>(
    query_entry: &'q query_file::Entry,
    db_entries: &'db [db_file::Entry],
    query_results: &mut Vec<DbEntryMatches<'db>>,
    cli: &'cli Cli,
) -> anyhow::Result<QueryAligner<'q, 'cli>> {
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

    Ok(QueryAligner { query_entry, cli })
}

pub(crate) struct QueryAligner<'q, 'cli> {
    query_entry: &'q query_file::Entry,
    cli: &'cli Cli,
}

impl<'q, 'cli> QueryAligner<'q, 'cli> {
    pub(crate) fn into_iter<'res, 'db, 'aln>(
        self,
        query_results: &'res [DbEntryMatches<'db>],
        aligner: &'aln mut Aligner<'cli>,
    ) -> QueryAlignIterator<'q, 'db, 'res, 'cli, 'aln> {
        let Self { query_entry, cli } = self;

        let query_results = query_results.iter();
        QueryAlignIterator(QueryAlignIteratorEnum::Empty {
            query_results,
            query_entry,
            cli,
            aligner,
        })
    }
}

pub(crate) struct QueryAlignIterator<'q, 'db, 'res, 'cli, 'aln>(
    QueryAlignIteratorEnum<'q, 'db, 'res, 'cli, 'aln>,
);

enum QueryAlignIteratorEnum<'q, 'db, 'res, 'cli, 'aln> {
    Empty {
        query_results: slice::Iter<'res, DbEntryMatches<'db>>,
        query_entry: &'q query_file::Entry,
        cli: &'cli Cli,
        aligner: &'aln mut Aligner<'cli>,
    },
    Full {
        query_results: slice::Iter<'res, DbEntryMatches<'db>>,
        iter: QueryAlignIteratorInner<'q, 'db, 'res, 'cli, 'aln>,
        query_entry: &'q query_file::Entry,
        cli: &'cli Cli,
    },
    Finished,
}

impl<'q, 'db, 'res, 'cli, 'aln> QueryAlignIterator<'q, 'db, 'res, 'cli, 'aln> {
    fn make_new_iter(&mut self) -> Option<&mut QueryAlignIteratorInner<'q, 'db, 'res, 'cli, 'aln>> {
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
        mut query_results: slice::Iter<'res, DbEntryMatches<'db>>,
        query_entry: &'q query_file::Entry,
        cli: &'cli Cli,
        aligner: &'aln mut Aligner<'cli>,
    ) -> Option<&mut QueryAlignIteratorInner<'q, 'db, 'res, 'cli, 'aln>> {
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
    fn get_next_from_new_iter(&mut self) -> Option<QueryAlignResult<'res>> {
        loop {
            let next = self.make_new_iter()?.next();
            if next.is_some() {
                break next;
            }
        }
    }
}

impl<'q, 'db, 'res, 'cli, 'aln> Iterator for QueryAlignIterator<'q, 'db, 'res, 'cli, 'aln> {
    type Item = QueryAlignResult<'res>;

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

struct QueryAlignIteratorInner<'q, 'db, 'res, 'cli, 'aln> {
    aligner: &'aln mut Aligner<'cli>,
    db_iter: slice::Iter<'res, MatchRanges>,
    query_entry: &'q query_file::Entry,
    db_entry: &'db db_file::Entry,
    cli: &'cli Cli,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QueryAlignResult<'res> {
    pub(crate) db_entry: &'res db_file::Entry,
    pub(crate) db_match: MatchRanges,
    pub(crate) score: Reactivity,
    pub(crate) db: ops::RangeInclusive<usize>,
    pub(crate) query: ops::RangeInclusive<usize>,
}

impl<'q, 'db: 'res, 'res, 'cli, 'aln> Iterator
    for QueryAlignIteratorInner<'q, 'db, 'res, 'cli, 'aln>
{
    type Item = QueryAlignResult<'res>;

    fn next(&mut self) -> Option<Self::Item> {
        let &mut Self {
            ref mut aligner,
            ref mut db_iter,
            query_entry,
            db_entry,
            cli,
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

            let MatchRanges { db, query } = db_match.clone();
            let query_len = query_entry.sequence().len();
            let db_len = db_entry.sequence().len();
            let align_tolerance = calc_seed_align_tolerance(
                query.clone(),
                db.clone(),
                query_len,
                db_len,
                cli.alignment_args.align_len_tolerance,
            );
            let align_tolerance = &align_tolerance;

            let upstream_result = aligner.align(AlignParams {
                query: query_entry,
                target: db_entry,
                query_range: query.clone(),
                target_range: db.clone(),
                seed_score,
                align_tolerance,
                direction: Direction::Upstream,
            });

            let downstream_result = aligner.align(AlignParams {
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
            let db_match = db_match.clone();

            break Some(QueryAlignResult {
                db_entry,
                db_match,
                score,
                db,
                query,
            });
        }
    }
}

impl QueryAlignResult<'_> {
    #[inline]
    pub(crate) fn query_len(&self) -> usize {
        self.query.end() + 1 - self.query.start()
    }
}
