use std::{
    cmp::Reverse,
    ops::{Not, RangeInclusive},
    sync::Arc,
};

use rand::{rngs::ThreadRng, thread_rng};
use statrs::distribution::{self, ContinuousCDF};

use crate::{
    alifold_mfe, alifold_on_result,
    aligner::{AlignedSequence, BacktrackBehavior, NoOpBehavior},
    cli::Cli,
    dotbracket::{self, DotBracketOwnedSorted},
    iter::IterWithRestExt,
    norm_dist::NormDist,
    null_model::ExtremeDistribution,
    query_aligner::{align_query_to_target_db, QueryAlignResult},
    query_file, query_result, reuse_vec, AlifoldOnResult, HandlerData, MutableHandlerData,
    QueryResult, SequenceEntry, SharedHandlerData,
};

pub(super) fn handle_query_entry<'a>(
    query_entry: &'a query_file::Entry,
    query_entry_orig: &'a query_file::Entry,
    handler_data: HandlerData<'a>,
) -> anyhow::Result<MutableHandlerData<'a>> {
    let HandlerData {
        shared:
            SharedHandlerData {
                cli,
                db_entries,
                db_entries_orig,
                db_entries_shuffled,
            },
        mutable:
            MutableHandlerData {
                mut aligner,
                mut null_all_scores,
                mut null_scores,
                mut query_all_results,
                mut reusable_query_results,
                mut index_to_remove,
                mut results,
            },
    } = handler_data;

    let null_aligner = align_query_to_target_db::<NoOpBehavior>(
        query_entry,
        db_entries_shuffled,
        db_entries_shuffled,
        &mut null_all_scores,
        cli,
    )?;
    null_scores.clear();
    null_scores.extend(
        null_aligner
            .into_iter(&null_all_scores, &mut aligner)
            .take(cli.null_hsgs.try_into().unwrap_or(usize::MAX))
            .map(|query_align_result| query_align_result.score),
    );
    let null_distribution = ExtremeDistribution::from_sample(&null_scores);

    let mut query_results = reuse_vec(reusable_query_results);
    query_results.extend(
        align_query_to_target_db::<BacktrackBehavior>(
            query_entry,
            db_entries,
            db_entries_orig,
            &mut query_all_results,
            cli,
        )?
        .into_iter(&query_all_results, &mut aligner)
        .map(|result| {
            assert_eq!(
                result.alignment.query.0.len(),
                result.alignment.target.0.len()
            );

            assert_eq!(
                result.db.end() - result.db.start()
                    + result
                        .alignment
                        .target
                        .0
                        .iter()
                        .filter(|bog| bog.is_base().not())
                        .count(),
                result.query.end() - result.query.start()
                    + result
                        .alignment
                        .query
                        .0
                        .iter()
                        .filter(|bog| bog.is_base().not())
                        .count(),
            );

            let p_value = null_distribution.p_value(result.score);

            // FIXME: we need to avoid this clone
            (result.clone(), p_value, 0.)
        }),
    );

    // In case of precision loss, it is still ok to evaluate the e_value
    #[allow(clippy::cast_precision_loss)]
    let results_len = query_results.len() as f64;
    let report_evalue = cli.report_evalue.max(cli.inclusion_evalue);
    query_results.retain_mut(|(_, p_value, e_value)| {
        *e_value = *p_value * results_len;
        *e_value <= report_evalue
    });
    remove_overlapping_results(&mut query_results, &mut index_to_remove, cli);

    let mut query_results_handler = QueryResultHandler::new(
        Arc::clone(&query_entry.name),
        query_entry,
        query_entry_orig,
        cli,
    );
    results.extend(
        query_results
            .iter()
            .map(|&(ref result, pvalue, evalue)| query_results_handler.run(result, pvalue, evalue)),
    );

    reusable_query_results = reuse_vec(query_results);
    Ok(MutableHandlerData {
        aligner,
        null_all_scores,
        null_scores,
        query_all_results,
        reusable_query_results,
        index_to_remove,
        results,
    })
}

struct QueryResultHandler<'a> {
    query: Arc<str>,
    query_entry: &'a query_file::Entry,
    query_entry_orig: &'a query_file::Entry,
    cli: &'a Cli,
    dotbracket_results_buffer: Vec<dotbracket::PairedBlock>,
    dotbracket_temp_buffer: Vec<dotbracket::PartialPairedBlock>,
    null_model_energies: Vec<f32>,
    rng: ThreadRng,
}

impl<'a> QueryResultHandler<'a> {
    fn new(
        query: Arc<str>,
        query_entry: &'a query_file::Entry,
        query_entry_orig: &'a query_file::Entry,
        cli: &'a Cli,
    ) -> Self {
        Self {
            query,
            query_entry,
            query_entry_orig,
            cli,
            dotbracket_results_buffer: Vec::new(),
            dotbracket_temp_buffer: Vec::new(),
            null_model_energies: Vec::new(),
            rng: thread_rng(),
        }
    }

    fn run(
        &mut self,
        result: &QueryAlignResult<AlignedSequence>,
        pvalue: f64,
        exp_value: f64,
    ) -> QueryResult {
        let mut status = if exp_value > self.cli.report_evalue {
            query_result::Status::NotPass
        } else if exp_value > self.cli.inclusion_evalue {
            query_result::Status::PassReportEvalue
        } else {
            query_result::Status::PassInclusionEvalue
        };

        let (mfe_pvalue_dotbracket, bp_support) = self.get_mfe_data(result, &mut status);

        let &QueryAlignResult {
            db_entry,
            ref db_match,
            score,
            db: ref db_range,
            query: ref query_range,
            ref alignment,
            ..
        } = result;

        let db_entry = db_entry.name().to_owned();
        let query = Arc::clone(&self.query);
        let alignment = Arc::clone(alignment);

        let query_seed = query_result::Range(db_match.query.clone());
        let db_seed = query_result::Range(db_match.db.clone());
        let query_start = *query_range.start();
        let query_end = *query_range.end();
        let db_start = *db_range.start();
        let db_end = *db_range.end();

        let (target_bp_support, query_bp_support) = bp_support
            .map(|BpSupport { target, query }| (target, query))
            .unzip();
        let (mfe_pvalue, dotbracket) = match mfe_pvalue_dotbracket {
            MfeResult::Evaluated { pvalue, dotbracket } => {
                (Some(pvalue.unwrap_or(1.)), Some(dotbracket))
            }
            MfeResult::Unevaluated => (None, None),
        };

        QueryResult {
            query,
            db_entry,
            query_start,
            query_end,
            db_start,
            db_end,
            query_seed,
            db_seed,
            score,
            pvalue,
            evalue: exp_value,
            target_bp_support,
            query_bp_support,
            mfe_pvalue,
            status,
            alignment,
            dotbracket,
        }
    }

    fn get_mfe_data(
        &mut self,
        result: &QueryAlignResult<AlignedSequence>,
        status: &mut query_result::Status,
    ) -> (MfeResult, Option<BpSupport>) {
        let &mut Self {
            ref mut query_entry,
            query_entry_orig,
            cli,
            ref mut dotbracket_results_buffer,
            ref mut dotbracket_temp_buffer,
            ref mut null_model_energies,
            ref mut rng,
            ..
        } = self;

        if cli.alignment_folding_eval_args.eval_align_fold.not()
            || matches!(status, query_result::Status::PassInclusionEvalue).not()
        {
            return (MfeResult::Unevaluated, None);
        }

        let AlifoldOnResult {
            dotbracket,
            ignore,
            mfe,
            gapped_data,
            target_bp_support,
            query_bp_support,
        } = alifold_on_result(
            result,
            query_entry,
            query_entry_orig,
            cli,
            dotbracket_results_buffer,
            dotbracket_temp_buffer,
        );

        let bp_support = BpSupport {
            target: target_bp_support,
            query: query_bp_support,
        };

        if ignore {
            *status = query_result::Status::PassReportEvalue;
            return (MfeResult::Unevaluated, Some(bp_support));
        }

        let mut indices_buffer = Vec::new();
        let mut block_indices_buffer = cli
            .alignment_folding_eval_args
            .in_block_shuffle
            .then_some(Vec::new());

        let block_size = cli.alignment_folding_eval_args.block_size;
        null_model_energies.clear();
        null_model_energies.extend((0..cli.alignment_folding_eval_args.shufflings).map(
            move |_| match &mut block_indices_buffer {
                Some(block_indices_buffer) => {
                    let gapped_data = gapped_data.clone().shuffled_in_blocks(
                        block_size,
                        &mut indices_buffer,
                        block_indices_buffer,
                        rng,
                    );

                    let sequences = [gapped_data.target(), gapped_data.query()];
                    let (_, mfe) = alifold_mfe(&sequences, &sequences, cli);

                    mfe
                }
                None => {
                    let gapped_data = gapped_data.clone().shuffled(
                        cli.alignment_folding_eval_args.block_size,
                        &mut indices_buffer,
                        rng,
                    );

                    let sequences = [gapped_data.target(), gapped_data.query()];
                    let (_, mfe) = alifold_mfe(&sequences, &sequences, cli);

                    mfe
                }
            },
        ));
        let dist = NormDist::from_sample(null_model_energies.as_slice());
        let z_score = dist.z_score(mfe);

        let mfe_pvalue = (z_score < 0.).then(|| {
            distribution::Normal::new(dist.mean(), dist.stddev())
                .expect("stddev is expected to be greater than 0")
                .cdf(mfe.into())
        });
        let mfe_result = MfeResult::Evaluated {
            pvalue: mfe_pvalue,
            dotbracket: dotbracket.unwrap().into_sorted().to_owned(),
        };

        (mfe_result, Some(bp_support))
    }
}

#[derive(Debug)]
struct BpSupport {
    query: f32,
    target: f32,
}

#[derive(Debug)]
enum MfeResult {
    Unevaluated,
    Evaluated {
        pvalue: Option<f64>,
        dotbracket: DotBracketOwnedSorted,
    },
}

fn remove_overlapping_results(
    results: &mut Vec<(QueryAlignResult<'_, AlignedSequence>, f64, f64)>,
    indices_buffer: &mut Vec<usize>,
    cli: &Cli,
) {
    let max_align_overlap: f64 = cli.max_align_overlap.into();
    indices_buffer.clear();
    results.sort_unstable_by(|(a, _, _), (b, _, _)| {
        a.db_entry
            .id
            .cmp(&b.db_entry.id)
            .then(a.query.start().cmp(b.query.start()))
            .then(a.query.end().cmp(b.query.end()).reverse())
    });
    results
        .iter_with_rest()
        .enumerate()
        .flat_map(|(a_index, (a, rest))| {
            let same_db_index = rest.partition_point(|b| a.0.db_entry.id == b.0.db_entry.id);
            // TODO: check if pre-calculating `a_len` does change anything
            rest[..same_db_index]
                .iter()
                .enumerate()
                .take_while(|(_, b)| are_overlapping(&a.0.query, &b.0.query, max_align_overlap))
                .filter(|(_, b)| are_overlapping(&a.0.db, &b.0.db, max_align_overlap))
                .map(move |(b_offset, b)| (a_index, a_index + b_offset + 1, a.0.score, b.0.score))
        })
        .for_each(|(a_index, b_index, a_score, b_score)| {
            if a_score >= b_score {
                indices_buffer.push(b_index);
            } else {
                indices_buffer.push(a_index);
            }
        });

    indices_buffer.sort_unstable_by_key(|&index| Reverse(index));

    if let Some(&first_index) = indices_buffer.first() {
        results.swap_remove(first_index);
        indices_buffer
            .windows(2)
            .map(|win| [win[0], win[1]])
            .filter(|[a, b]| a != b)
            .for_each(|[_, index]| {
                results.swap_remove(index);
            });
    }
}

#[inline]
fn are_overlapping(
    a: &RangeInclusive<usize>,
    b: &RangeInclusive<usize>,
    max_align_overlap: f64,
) -> bool {
    b.start() < a.end() && {
        let overlap = overlapping_range(a, b);
        // If we are losing precision, ranges are so bit that we are probably going to crash elsewhere
        // anyway
        #[allow(clippy::cast_precision_loss)]
        let a_len = (a.end() + 1 - a.start()) as f64;
        #[allow(clippy::cast_precision_loss)]
        let b_len = (b.end() + 1 - b.start()) as f64;

        #[allow(clippy::cast_precision_loss)]
        let overlap = (overlap.end() + 1).saturating_sub(*overlap.start()) as f64;

        overlap > (a_len.min(b_len)) * max_align_overlap
    }
}

#[inline]
fn overlapping_range<T>(a: &RangeInclusive<T>, b: &RangeInclusive<T>) -> RangeInclusive<T>
where
    T: Ord + Clone,
{
    let start = a.start().max(b.start()).clone();
    let end = a.end().min(b.end()).clone();
    start..=end
}

#[cfg(test)]
mod tests {
    use crate::{
        aligner::{AlignmentResult, BaseOrGap},
        db_file::{self, ReactivityWithPlaceholder},
        tests::dummy_cli,
        Base, MatchRanges,
    };

    use super::*;

    #[test]
    fn remove_overlapping_results_empty() {
        remove_overlapping_results(&mut Vec::new(), &mut vec![1, 2, 3], &dummy_cli());
    }

    static EMPTY_ENTRY: db_file::Entry = db_file::Entry {
        id: String::new(),
        sequence: Vec::new(),
        reactivity: Vec::new(),
    };
    macro_rules! query_result {
        ($query:expr, $score:expr) => {
            (
                QueryAlignResult {
                    db_entry: &EMPTY_ENTRY,
                    db_entry_orig: &EMPTY_ENTRY,
                    db_match: MatchRanges {
                        db: 0..=0,
                        query: 0..=0,
                    },
                    score: $score,
                    db: 0..=10,
                    query: $query,
                    alignment: Default::default(),
                },
                0.0,
                0.0,
            )
        };

        ($query:expr) => {
            query_result!($query, 0.)
        };
    }

    #[test]
    fn remove_overlapping_results_non_overlapping() {
        let mut results = vec![
            query_result!(0..=4),
            query_result!(4..=10),
            query_result!(8..=16),
        ];
        let initial_results = results.clone();
        remove_overlapping_results(&mut results, &mut vec![0, 1, 2, 3, 4], &dummy_cli());

        assert_eq!(results, initial_results);
    }

    #[test]
    fn remove_overlapping_results_simple_overlap() {
        let mut results = vec![query_result!(0..=10, 0.5), query_result!(4..=14, 0.2)];
        remove_overlapping_results(&mut results, &mut vec![0, 1, 2, 3, 4], &dummy_cli());

        assert_eq!(results, vec![query_result!(0..=10, 0.5)]);

        results = vec![query_result!(0..=10, 0.2), query_result!(4..=14, 0.5)];
        remove_overlapping_results(&mut results, &mut vec![0, 1, 2, 3, 4], &dummy_cli());

        assert_eq!(results, vec![query_result!(4..=14, 0.5)]);
    }

    #[test]
    fn remove_overlapping_results_nested_overlap() {
        let mut results = vec![query_result!(0..=10, 0.5), query_result!(2..=6, 0.2)];
        remove_overlapping_results(&mut results, &mut vec![], &dummy_cli());

        assert_eq!(results, vec![query_result!(0..=10, 0.5)]);

        results = vec![query_result!(0..=10, 0.2), query_result!(2..=6, 0.5)];
        remove_overlapping_results(&mut results, &mut vec![], &dummy_cli());

        assert_eq!(results, vec![query_result!(2..=6, 0.5)]);
    }

    #[test]
    fn remove_overlapping_results_chained_overlap() {
        let mut results = vec![
            query_result!(0..=5, 0.5),
            query_result!(2..=8, 0.5),
            query_result!(4..=10, 0.7),
            query_result!(6..=12, 0.5),
            query_result!(8..=14, 0.5),
            query_result!(10..=16, 0.5),
        ];
        remove_overlapping_results(&mut results, &mut vec![], &dummy_cli());

        // The reason for this result is, using the current algorithm:
        // - the second is removed because this is the current behavior for overlapping sequences
        //   with the same score;
        // - the third is kept (it removes the second, again) because it has a higher score;
        // - all the others are removed "by chaining" (the first has a higher priority in case of
        //   equal score, but it's been removed by another comparison).
        assert_eq!(
            results,
            vec![query_result!(0..=5, 0.5), query_result!(4..=10, 0.7)]
        );
    }

    #[test]
    fn keep_targets_with_overlapping_results() {
        let cli = Cli::dummy();

        let targets: Vec<_> = (0..5)
            .map(|index| db_file::Entry {
                id: format!("db_{index}"),
                sequence: vec![Base::A; 100],
                reactivity: vec![ReactivityWithPlaceholder::from(0.5); 100],
            })
            .collect();

        let alignment_result = Arc::new(AlignmentResult {
            query: AlignedSequence(vec![BaseOrGap::Base; 4]),
            target: AlignedSequence(vec![BaseOrGap::Base; 4]),
        });
        let alignment_result = &alignment_result;
        let mut results = targets
            .iter()
            .enumerate()
            .flat_map(|(outer_index, db_entry)| {
                (0..3).map(move |inner_index| {
                    let result = QueryAlignResult {
                        db_entry,
                        db_entry_orig: db_entry,
                        db_match: MatchRanges {
                            db: 13..=16,
                            query: 13..=16,
                        },
                        score: 15. + f32::from(u16::try_from(outer_index).unwrap()) * 3.
                            - (f32::from(i16::try_from(inner_index).unwrap()) * 2.),
                        db: 10..=20,
                        query: 10..=20,
                        alignment: Arc::clone(alignment_result),
                    };

                    let p_value = f64::from(u32::try_from(outer_index).unwrap()) / 1000. + 0.01
                        - (f64::from(inner_index + 1) / 10000.);

                    (result, p_value, p_value)
                })
            })
            .collect();

        let mut indices = vec![];
        remove_overlapping_results(&mut results, &mut indices, &cli);

        assert_eq!(results.len(), targets.len());
        let mut scores: Vec<_> = results
            .into_iter()
            .map(|(result, _, _)| result.score)
            .collect();
        scores.sort_unstable_by(f32::total_cmp);
        assert_eq!(
            scores,
            (0..5)
                .map(|index| 15. + f32::from(i16::try_from(index).unwrap()) * 3.)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn keep_targets_with_overlapping_results_different_target() {
        let mut results = vec![
            query_result!(0..=4),
            (
                QueryAlignResult {
                    db: 15..=20,
                    ..(query_result!(0..=4).0)
                },
                0.,
                0.,
            ),
        ];
        let initial_results = results.clone();
        remove_overlapping_results(&mut results, &mut vec![0, 1, 2, 3, 4], &dummy_cli());

        assert_eq!(results, initial_results);
    }
}
