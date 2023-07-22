#![warn(clippy::pedantic)]

mod aligner;
mod cli;
mod db_file;
mod dotbracket;
mod fasta;
mod gapped_data;
mod gapped_reactivity;
mod gapped_sequence;
mod handle_query_entry;
mod iter;
mod mass;
mod norm_dist;
mod null_model;
mod query_aligner;
mod query_file;
mod stockholm;
mod viennarna;

use std::{
    fmt::{self, Display, Write},
    fs::{self, File},
    io::{self, BufWriter},
    iter::Sum,
    num::ParseFloatError,
    ops::{self, Not, Range, RangeInclusive},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use aligner::{trimmed_range, AlignedSequence, Aligner, AlignmentResult};
use anyhow::{bail, Context};
use clap::Parser;
use cli::MinMax;
use db_file::{ReactivityLike, ReactivityWithPlaceholder};
use dotbracket::{DotBracketBuffered, DotBracketOwnedSorted};
use fftw::{
    array::AlignedVec,
    plan::{C2CPlan, C2CPlan32, C2CPlan64},
    types::{Flag, Sign},
};
use fnv::FnvHashMap;
use gapped_data::GappedData;
use itertools::Itertools;
use mass::{ComplexExt, Mass};
use null_model::make_shuffled_db;
use num_complex::Complex;
use num_traits::{cast, float::FloatCore, Float, FromPrimitive, NumAssignRef, NumRef, RefNum};
use query_aligner::QueryAlignResult;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use smallvec::SmallVec;
use tabled::{Table, Tabled};
use viennarna::Structure;

use crate::{
    cli::{Cli, ReportAlignment},
    dotbracket::DotBracket,
    handle_query_entry::handle_query_entry,
    viennarna::{FoldCompound, FoldCompoundOptions, ModelDetails},
};
pub(crate) use crate::{
    gapped_reactivity::{GappedReactivity, GappedReactivityLike},
    gapped_sequence::{GappedSequence, GappedSequenceLike},
};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let Cli {
        overwrite,
        ref output,
        threads,
        db_shufflings,
        db_block_size,
        ..
    } = cli;

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.unwrap_or(0).into())
        .build_global()
        .context("Unable to create thread pool")?;

    if matches!(output.try_exists(), Ok(true)) {
        if overwrite {
            if let Err(err) = fs::remove_dir_all(output) {
                eprintln!(
                    "Warning: cannot remove {} directory: {err}",
                    output.display()
                );
            }
        } else {
            bail!(
                "Cannot run, output directory \"{}\" already exists. If you want to overwrite results, use the `--overwrite` argument.",
                output.display(),
            );
        }
    }

    fs::create_dir_all(output).context("Unable to create output directory")?;
    write_cli_to_file(&cli)?;

    let query_entries_orig = query_file::read_file(&cli.query)?;
    let db_entries_orig = db_file::read_file(&cli.database)?;

    let query_entries: Vec<_> = query_entries_orig
        .iter()
        .map(|entry| {
            let mut entry = entry.clone();
            entry.cap_reactivities(cli.max_reactivity);
            entry
        })
        .collect();

    let db_entries: Vec<_> = db_entries_orig
        .iter()
        .map(|entry| {
            let mut entry = entry.clone();
            entry.cap_reactivities(cli.max_reactivity);
            entry
        })
        .collect();

    let db_entries_shuffled = cli
        .shuffled_db
        .as_deref()
        .map(|shuffled_db_path| {
            let mut shuffled_db = db_file::read_file(shuffled_db_path)?;
            shuffled_db
                .iter_mut()
                .for_each(|entry| entry.cap_reactivities(cli.max_reactivity));
            Ok::<_, db_file::Error>(shuffled_db)
        })
        .transpose()?
        .unwrap_or_else(|| {
            make_shuffled_db(&db_entries, db_block_size.into(), db_shufflings.into())
        });

    if let Some(shuffled_db_output_path) = cli.dump_shuffled_db.as_deref() {
        db_file::write_entries(
            &db_entries_shuffled,
            BufWriter::new(
                File::create(shuffled_db_output_path)
                    .expect("unable to open output file for dumping shuffled db"),
            ),
        )
        .expect("unable to write shuffled db to file");
    }

    run(
        &query_entries,
        &query_entries_orig,
        &db_entries,
        &db_entries_orig,
        &db_entries_shuffled,
        &cli,
    )
}

fn run(
    query_entries: &[query_file::Entry],
    query_entries_orig: &[query_file::Entry],
    db_entries: &[db_file::Entry],
    db_entries_orig: &[db_file::Entry],
    db_entries_shuffled: &[db_file::Entry],
    cli: &Cli,
) -> anyhow::Result<()> {
    let &Cli {
        ref output,
        report_alignment,
        report_reactivity,
        ..
    } = cli;

    let results_path = output.join("results.out");

    let mut results = query_entries
        .par_iter()
        .zip(query_entries_orig)
        .try_fold(
            || MutableHandlerData::new(cli),
            |mutable, (query_entry, query_entries_orig)| {
                handle_query_entry(
                    query_entry,
                    query_entries_orig,
                    HandlerData {
                        shared: SharedHandlerData {
                            cli,
                            db_entries,
                            db_entries_orig,
                            db_entries_shuffled,
                        },
                        mutable,
                    },
                )
            },
        )
        .map(|mutable_handler_data| {
            mutable_handler_data.map(|mutable_handler_data| mutable_handler_data.results)
        })
        .try_reduce_with(|mut a, b| {
            a.extend(b);
            Ok(a)
        })
        .transpose()?
        .unwrap_or_default();

    results.sort_unstable_by(|a, b| {
        a.evalue
            .total_cmp(&b.evalue)
            .then_with(|| a.query.cmp(&b.query))
            .then_with(|| a.query_start.cmp(&b.query_start))
            .then_with(|| a.db_entry.cmp(&b.db_entry))
            .then_with(|| a.db_start.cmp(&b.db_start))
    });

    let mut results_writer = {
        let file = File::create(results_path).context("Unable to create results.out file")?;
        create_csv_from_writer(file)
    };
    if results.is_empty() {
        // This is a dirty workaround to make csv write the header
        use std::io::Write;

        let mut tmp_data = Vec::new();
        create_csv_from_writer(&mut tmp_data).serialize(QueryResult::new(""))?;
        let header = tmp_data
            .splitn(2, |&c| c == b'\n')
            .next()
            .expect("CSV should have written at least one line");
        let mut results_writer = results_writer.into_inner()?;
        results_writer.write_all(header)?;
    } else {
        results
            .iter()
            .try_for_each(|result| results_writer.serialize(result))
            .context("Unable to write to results.out file")?;

        results_writer
            .flush()
            .context("Unable to flush results.out file")?;
    }

    if let Some(report_alignment) = report_alignment {
        handle_report_alignment(
            report_alignment,
            &results,
            query_entries_orig,
            db_entries_orig,
            cli,
        )?;
    }

    if report_reactivity {
        write_results_reactivity(&results, db_entries_orig, query_entries_orig, output)
            .context("Unable to write result reactivity in JSON format")?;
    }

    // Clear screen and position cursor to row 1, column 1
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
    if results.is_empty() {
        println!("No match found with current search settings.");
    } else {
        use tabled::settings::{object, Disable, Style};

        let mut table = Table::new(results);
        table.with(Style::empty());
        if cli.alignment_folding_eval_args.eval_align_fold.not() {
            table.with(Disable::column(object::Columns::new(11..14)));
        }
        println!("{table}");
    }

    Ok(())
}

fn handle_report_alignment(
    report_alignment: ReportAlignment,
    results: &[QueryResult],
    query_entries_orig: &[query_file::Entry],
    db_entries_orig: &[db_file::Entry],
    cli: &Cli,
) -> anyhow::Result<()> {
    let alignments_path = cli.output.join("alignments");
    fs::create_dir_all(&alignments_path).context("Unable to create alignments output directory")?;

    let mut results = results
        .iter()
        .filter(|result| matches!(result.status, QueryResultStatus::PassInclusionEvalue));
    match report_alignment {
        ReportAlignment::Fasta => {
            results
                .try_for_each(|result| {
                    fasta::write_result(
                        result,
                        db_entries_orig,
                        query_entries_orig,
                        &alignments_path,
                    )
                })
                .context("Unable to write report alignment in FASTA format")?;
        }
        ReportAlignment::Stockholm => {
            results
                .try_for_each(|result| {
                    stockholm::write_result(
                        result,
                        db_entries_orig,
                        query_entries_orig,
                        &alignments_path,
                    )
                })
                .context("Unable to write report alignment in stockholm format")?;
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct SharedHandlerData<'a> {
    cli: &'a Cli,
    db_entries: &'a [db_file::Entry],
    db_entries_orig: &'a [db_file::Entry],
    db_entries_shuffled: &'a [db_file::Entry],
}

struct MutableHandlerData<'a> {
    aligner: Aligner<'a>,
    null_all_scores: Vec<DbEntryMatches<'a>>,
    null_scores: Vec<Reactivity>,
    query_all_results: Vec<DbEntryMatches<'a>>,
    reusable_query_results: Vec<(QueryAlignResult<'a, AlignedSequence>, f64, f64)>,
    index_to_remove: Vec<usize>,
    results: Vec<QueryResult>,
}

impl<'a> MutableHandlerData<'a> {
    fn new(cli: &'a Cli) -> Self {
        Self {
            aligner: Aligner::new(cli),
            null_all_scores: Vec::default(),
            null_scores: Vec::default(),
            query_all_results: Vec::default(),
            reusable_query_results: Vec::default(),
            index_to_remove: Vec::default(),
            results: Vec::default(),
        }
    }
}

struct HandlerData<'a> {
    shared: SharedHandlerData<'a>,
    mutable: MutableHandlerData<'a>,
}

// Small hack to reuse vec allocation
#[inline]
fn reuse_vec<T, U>(mut v: Vec<T>) -> Vec<U> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    v.clear();
    v.into_iter().map(|_| unreachable!()).collect()
}

#[derive(Debug, Deserialize, Serialize, Tabled)]
struct QueryResult {
    #[serde(rename = "Query")]
    query: Arc<str>,

    #[serde(rename = "DB entry")]
    db_entry: String,

    #[serde(rename = "Qstart")]
    query_start: usize,

    #[serde(rename = "Qend")]
    query_end: usize,

    #[serde(rename = "Dstart")]
    db_start: usize,

    #[serde(rename = "Dend")]
    db_end: usize,

    #[serde(rename = "Qseed")]
    query_seed: QueryResultRange,

    #[serde(rename = "Dseed")]
    db_seed: QueryResultRange,

    #[serde(rename = "Score")]
    score: f32,

    #[serde(rename = "P-value")]
    #[tabled(display_with = "display_scientific")]
    pvalue: f64,

    #[serde(rename = "E-value")]
    #[tabled(display_with = "display_scientific")]
    evalue: f64,

    #[serde(rename = "TargetBpSupport")]
    #[tabled(display_with = "display_scientific_opt")]
    target_bp_support: Option<f32>,

    #[serde(rename = "QueryBpSupport")]
    #[tabled(display_with = "display_scientific_opt")]
    query_bp_support: Option<f32>,

    #[serde(rename = "MfePvalue")]
    #[tabled(display_with = "display_scientific_opt")]
    mfe_pvalue: Option<f64>,

    #[serde(rename = "")]
    status: QueryResultStatus,

    #[serde(skip)]
    #[tabled(skip)]
    alignment: Arc<AlignmentResult<AlignedSequence>>,

    #[serde(skip)]
    #[tabled(skip)]
    dotbracket: Option<DotBracketOwnedSorted>,
}

impl QueryResult {
    fn new(query: impl Into<Arc<str>>) -> Self {
        let query = query.into();
        Self {
            query,
            db_entry: String::default(),
            query_start: Default::default(),
            query_end: Default::default(),
            db_start: Default::default(),
            db_end: Default::default(),
            query_seed: QueryResultRange::default(),
            db_seed: QueryResultRange::default(),
            score: Default::default(),
            pvalue: Default::default(),
            evalue: Default::default(),
            status: QueryResultStatus::default(),
            target_bp_support: Option::default(),
            query_bp_support: Option::default(),
            mfe_pvalue: Option::default(),
            alignment: Arc::default(),
            dotbracket: Option::default(),
        }
    }
}

#[derive(Debug)]
struct QueryResultRange(ops::RangeInclusive<usize>);

impl Default for QueryResultRange {
    fn default() -> Self {
        Self(0..=0)
    }
}

impl Serialize for QueryResultRange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for QueryResultRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        let raw = <&str>::deserialize(deserializer)?;
        let mut split = raw.split('-').map(str::parse);
        let start = split
            .next()
            .ok_or_else(|| Error::custom("missing start in range"))?
            .map_err(|_| Error::custom("invalid start in range"))?;

        let end = split
            .next()
            .ok_or_else(|| Error::custom("missing end in range"))?
            .map_err(|_| Error::custom("invalid end in range"))?;

        if split.next().is_some() {
            return Err(Error::custom("invalid range format"));
        }

        Ok(Self(start..=end))
    }
}

impl fmt::Display for QueryResultRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.0.start(), self.0.end())
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum QueryResultStatus {
    #[serde(rename = "!")]
    PassInclusionEvalue,

    #[serde(rename = "?")]
    PassReportEvalue,

    #[default]
    #[serde(rename = "")]
    NotPass,
}

impl fmt::Display for QueryResultStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::PassInclusionEvalue => f.write_str("!"),
            Self::PassReportEvalue => f.write_str("?"),
            Self::NotPass => f.write_str(""),
        }
    }
}

pub(crate) type Reactivity = f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Base {
    A,
    C,
    G,
    T,
    N,
}

impl Base {
    fn try_from_nibble(nibble: u8) -> Result<Self, InvalidEncodedBase> {
        use Base::{A, C, G, N, T};
        Ok(match nibble {
            0 => A,
            1 => C,
            2 => G,
            3 => T,
            4 => N,
            _ => return Err(InvalidEncodedBase),
        })
    }

    fn try_pair_from_byte(byte: u8) -> Result<[Self; 2], InvalidEncodedBase> {
        let first = Base::try_from_nibble(byte >> 4)?;
        let second = Base::try_from_nibble(byte & 0x0F)?;

        Ok([first, second])
    }

    fn try_to_byte(self, molecule: Molecule) -> Option<u8> {
        Some(match self {
            Self::A => b'A',
            Self::C => b'C',
            Self::G => b'G',
            Self::T => match molecule {
                Molecule::Dna => b'T',
                Molecule::Rna => b'U',
                Molecule::Unknown => return None,
            },
            Self::N => b'N',
        })
    }

    #[inline]
    fn to_byte(self, molecule: Molecule) -> u8 {
        self.try_to_byte(molecule).expect("cannot convert a timine-like residue to an ASCII representation when the molecule is unknown")
    }

    fn to_nibble(self) -> u8 {
        match self {
            Self::A => 0,
            Self::C => 1,
            Self::G => 2,
            Self::T => 3,
            Self::N => 4,
        }
    }

    fn pair_to_nibble(pair: [Self; 2]) -> u8 {
        (Self::to_nibble(pair[0]) << 4) | Self::to_nibble(pair[1])
    }

    #[inline]
    fn display(self, molecule: Molecule) -> BaseDisplay {
        BaseDisplay {
            base: self,
            molecule,
        }
    }
}

impl TryFrom<u8> for Base {
    type Error = InvalidEncodedBase;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            b'A' => Self::A,
            b'C' => Self::C,
            b'G' => Self::G,
            b'T' | b'U' => Self::T,
            b'N' => Self::N,
            _ => return Err(InvalidEncodedBase),
        })
    }
}

struct BaseDisplay {
    base: Base,
    molecule: Molecule,
}

impl fmt::Display for BaseDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char(self.base.to_byte(self.molecule).into())
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
struct Sequence<'a> {
    pub bases: &'a [Base],
    pub molecule: Molecule,
}

impl fmt::Display for Sequence<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.bases
            .iter()
            .try_for_each(|base| write!(f, "{}", base.display(self.molecule)))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InvalidEncodedBase;

fn calc_seed_alignment_score_from_reactivity(
    query: &[ReactivityWithPlaceholder],
    target: &[ReactivityWithPlaceholder],
    cli: &Cli,
) -> f32 {
    assert_eq!(query.len(), target.len());

    query
        .iter()
        .zip(target)
        .map(|(&query, &target)| calc_base_alignment_score(query, target, cli))
        .sum()
}

fn calc_seed_alignment_score_from_sequence(query: &[Base], target: &[Base], cli: &Cli) -> f32 {
    assert_eq!(query.len(), target.len());

    query
        .iter()
        .zip(target)
        .map(|(&query, &target)| {
            get_sequence_base_alignment_score(query, target, &cli.alignment_args)
        })
        .sum()
}

#[inline]
fn calc_base_alignment_score(
    query: ReactivityWithPlaceholder,
    target: ReactivityWithPlaceholder,
    cli: &Cli,
) -> f32 {
    let Cli {
        alignment_args:
            cli::AlignmentArgs {
                align_match_score: MinMax(align_match),
                align_mismatch_score: MinMax(align_mismatch),
                ..
            },
        max_reactivity,
        ..
    } = cli;

    match (target.get_non_nan(), query.get_non_nan()) {
        (None, _) | (_, None) => align_match.start,
        (Some(target), Some(query)) if query > 1. && target > 1. => 0f32,
        (Some(target), Some(query)) => {
            let diff = (query - target).abs();
            if diff < 0.5 {
                (0.5 - diff) * (align_match.end - align_match.start) / 0.5 + align_match.start
            } else {
                (max_reactivity - diff) * (align_mismatch.end - align_mismatch.start)
                    / (max_reactivity - 0.5)
                    + align_mismatch.start
            }
        }
    }
}

#[inline]
fn get_sequence_base_alignment_score(
    query: Base,
    target: Base,
    alignment_args: &cli::AlignmentArgs,
) -> f32 {
    if query == target {
        alignment_args.align_seq_match_score
    } else {
        alignment_args.align_seq_mismatch_score
    }
}

#[derive(Debug)]
struct DbData<'a> {
    sequence: &'a [Base],
    reactivity: &'a [ReactivityWithPlaceholder],
    transformed_reactivity: AlignedVec<Complex<Reactivity>>,
}

impl<'a> DbData<'a> {
    fn new(
        sequence: &'a [Base],
        reactivity: &'a [ReactivityWithPlaceholder],
    ) -> Result<Self, fftw::error::Error> {
        let transformed_reactivity = transform_db(reactivity)?;

        Ok(Self {
            sequence,
            reactivity,
            transformed_reactivity,
        })
    }
}

trait C2CPlanExt: FloatCore + FromPrimitive {
    type Plan: C2CPlan<Complex = Complex<Self>>;
}

impl C2CPlanExt for f32 {
    type Plan = C2CPlan32;
}

impl C2CPlanExt for f64 {
    type Plan = C2CPlan64;
}

fn transform_db(
    db: &[ReactivityWithPlaceholder],
) -> Result<AlignedVec<Complex<Reactivity>>, fftw::error::Error> {
    let ts_len = db.len();
    let mut aligned_db = AlignedVec::new(ts_len);
    db.iter()
        .copied()
        .zip(aligned_db.iter_mut())
        .for_each(|(t, x)| *x = Complex::new(t.to_maybe_placeholder(), 0.));

    let mut db_transform = AlignedVec::<Complex<_>>::new(ts_len);
    let mut fw_plan: <Reactivity as C2CPlanExt>::Plan =
        C2CPlan::aligned(&[ts_len], Sign::Forward, Flag::ESTIMATE)?;
    fw_plan.c2c(&mut aligned_db, &mut db_transform)?;

    Ok(db_transform)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("FFTW error: {0}")]
    Fftw(#[from] fftw::error::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("DB reader error: {0}")]
    Reader(#[from] db_file::ReaderError),

    #[error("DB reader entry error: {0}")]
    ReaderEntry(#[from] db_file::EntryError),
}

fn get_matching_kmers(
    query_reactivity: &[ReactivityWithPlaceholder],
    query_sequence: &[Base],
    db_data: &DbData,
    cli: &Cli,
) -> Result<Vec<[usize; 2]>, Error> {
    let &Cli {
        kmer_lookup_args:
            cli::KmerLookupArgs {
                kmer_len,
                kmer_offset,
                kmer_max_seq_dist,
                kmer_max_gc_diff,
                kmer_min_complexity,
                kmer_max_match_every_nt,
                match_kmer_gc_content,
                ..
            },
        ..
    } = cli;

    let trimmed_query_range = trimmed_range(query_reactivity);

    let max_sequence_distance = kmer_max_seq_dist.map(|dist| dist.to_absolute(kmer_len.into()));
    let max_gc_diff = match_kmer_gc_content.then(|| {
        kmer_max_gc_diff.unwrap_or_else(|| {
            // We *want* to eventually lost precision in case Reactivity changes to f64
            #[allow(clippy::cast_lossless)]
            let len = kmer_len as Reactivity;

            0.2676 * (len * -0.053).exp()
        })
    });

    let mut mass = Mass::new(db_data.reactivity.len())?;

    let kmer_len_usize = kmer_len.into();
    let last_kmer_index = trimmed_query_range.end.saturating_sub(kmer_len_usize);
    let matches = query_reactivity
        .windows(kmer_len_usize)
        .zip(query_sequence.windows(kmer_len_usize))
        .enumerate()
        .step_by(kmer_offset.into())
        .skip_while(|&(index, _)| index < trimmed_query_range.start)
        .take_while(|&(index, _)| index <= last_kmer_index)
        .filter(|(_, (kmer, _))| kmer.iter().any(|&x| x.is_nan()).not())
        .filter(|(_, (kmer, _))| {
            gini_index(ReactivityWithPlaceholder::as_inner_slice(kmer)) >= kmer_min_complexity
        })
        .map(|(kmer_index, (kmer, kmer_sequence))| {
            let mut complex_distances = mass.run(
                db_data.reactivity,
                &db_data.transformed_reactivity,
                ReactivityWithPlaceholder::as_inner_slice(kmer),
            )?;
            for dist in &mut *complex_distances {
                *dist = Complex::new(dist.norm(), 0.);
            }

            let (mean_dist, stddev_dist) = mean_stddev(complex_distances.iter().copied(), 1);
            let max_distance = mean_dist.re - stddev_dist.re * 3.;

            // It is ok to lost precision to evaluate the fraction
            #[allow(clippy::cast_lossless, clippy::cast_precision_loss)]
            let kmer_gc_fraction = max_gc_diff.is_some().then(|| {
                let gc_count = kmer_sequence
                    .iter()
                    .filter(|&&c| matches!(c, Base::C | Base::G))
                    .count();

                gc_count as Reactivity / kmer_len as Reactivity
            });

            let mut kmer_data = complex_distances
                .into_iter()
                .enumerate()
                .map(|(index, dist)| (index, dist.re))
                .zip(db_data.sequence.windows(kmer_len.into()))
                .filter(move |&((_, dist), _)| dist <= max_distance)
                .map(|((index, _), db_sequence)| (index, db_sequence))
                .collect::<Vec<_>>();

            if kmer_data.is_empty().not()
                && u32::try_from(db_data.reactivity.len() / kmer_data.len()).unwrap_or(u32::MAX)
                    < kmer_max_match_every_nt
            {
                kmer_data.clear();
            }

            Ok::<_, fftw::error::Error>(
                kmer_data
                    .into_iter()
                    .filter(move |(_, db_sequence)| {
                        max_sequence_distance.map_or(true, move |max_sequence_distance| {
                            {
                                u32::try_from(hamming_distance(kmer_sequence, db_sequence))
                                    .unwrap_or(u32::MAX)
                                    <= max_sequence_distance
                            }
                        })
                    })
                    .filter(move |(_, db_sequence)| {
                        max_gc_diff.map_or(true, move |max_gc_diff| {
                            let gc_count = db_sequence
                                .iter()
                                .filter(|&&c| matches!(c, Base::C | Base::G))
                                .count();
                            // It is ok to evaluate a fraction
                            #[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
                            let gc_fraction = gc_count as Reactivity / kmer_len as Reactivity;
                            Float::abs(gc_fraction - kmer_gc_fraction.unwrap()) <= max_gc_diff
                        })
                    })
                    .map(move |(sequence_index, _)| [kmer_index, sequence_index]),
            )
        })
        .flatten_ok()
        .collect::<Result<Vec<_>, _>>()?;

    Ok(matches)
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(untagged)]
pub enum Distance<T> {
    Integral(T),
    Fractional(f64),
}

impl<T: num_traits::NumCast + Copy> Distance<T> {
    pub(crate) fn to_absolute(self, total_len: T) -> T {
        match self {
            Self::Integral(x) => x,
            Self::Fractional(fraction) => {
                debug_assert!(fraction <= 1.);
                cast(cast::<_, f64>(total_len).unwrap() * fraction).unwrap()
            }
        }
    }
}

impl<T> fmt::Display for Distance<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integral(x) => x.fmt(f),
            Self::Fractional(x) => x.fmt(f),
        }
    }
}

impl<T, E> FromStr for Distance<T>
where
    T: FromStr<Err = E>,
{
    type Err = ParseDistanceError<E>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.contains('.') {
            let x = s.parse::<f64>()?;
            if x <= 1. {
                Ok(Self::Fractional(x))
            } else {
                Err(ParseDistanceError::InvalidFractional)
            }
        } else {
            let x = s.parse::<T>().map_err(ParseDistanceError::Integral)?;
            Ok(Self::Integral(x))
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ParseDistanceError<E> {
    #[error("invalid integral distance: {0}")]
    Integral(E),

    #[error("invalid fractional distance: {0}")]
    Fractional(#[from] ParseFloatError),

    #[error("invalid fractional distance greater than 1")]
    InvalidFractional,
}

fn gini_index<T>(data: &[T]) -> T
where
    T: NumAssignRef + Float + Sum,
    for<'a> &'a T: RefNum<T>,
{
    let mut iter = data.iter();
    let mut diffs = T::zero();
    let mut sums = T::zero();
    while let Some(x_a) = iter.next() {
        sums += x_a;
        diffs += iter.clone().map(|x_b| (x_a - x_b).abs()).sum::<T>();
    }

    diffs / (T::from(data.len()).unwrap() * sums)
}

pub(crate) fn mean_stddev<I, T>(data: I, degree_of_freedom: u8) -> (T, T)
where
    I: IntoIterator<Item = T> + Clone,
    T: num_traits::NumCast + NumRef + Sum + for<'a> Sum<&'a T> + ComplexExt,
    for<'a> &'a T: RefNum<T>,
{
    let (sample_size, sum) = data
        .clone()
        .into_iter()
        .filter(mass::ComplexExt::is_finite)
        .fold((0usize, T::zero()), |(count, sum), x| (count + 1, sum + x));
    let sample_size = T::from(sample_size).unwrap();
    let mean = sum / &sample_size;
    let var_sum: T = data
        .into_iter()
        .filter(mass::ComplexExt::is_finite)
        .map(|x| (x - &mean).powi(2))
        .sum();
    (
        mean,
        (var_sum / (sample_size - T::from(degree_of_freedom).unwrap())).sqrt(),
    )
}

fn hamming_distance<T: Eq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(a, b)| a != b).count()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchRanges {
    db: RangeInclusive<usize>,
    query: RangeInclusive<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DbEntryMatches<'a> {
    db_entry: &'a db_file::Entry,
    db_entry_orig: &'a db_file::Entry,
    matches: Vec<MatchRanges>,
}

fn group_matching_kmers(matching_kmers: &[[usize; 2]], cli: &Cli) -> Vec<MatchRanges> {
    let &Cli {
        kmer_lookup_args:
            cli::KmerLookupArgs {
                min_kmers,
                kmer_len,
                max_kmer_dist,
                ..
            },
        ..
    } = cli;

    let mut diagonals = FnvHashMap::<_, SmallVec<[_; 8]>>::default();
    for pair in matching_kmers {
        let &[i, j] = pair;
        let diagonal = isize::try_from(i).unwrap() - isize::try_from(j).unwrap();
        diagonals.entry(diagonal).or_default().push(pair);
    }

    let kmer_len: usize = kmer_len.into();
    let max_distance = usize::from(max_kmer_dist) + kmer_len;
    let mut groups = Vec::new();
    let mut add_group = |first: &[usize; 2], last: &[usize; 2]| {
        // False positive, the destination type is a RangeInclusive
        #[allow(clippy::range_minus_one)]
        let group = MatchRanges {
            query: first[0]..=(last[0] + kmer_len - 1),
            db: first[1]..=(last[1] + kmer_len - 1),
        };
        groups.push(group);
    };

    for diagonal in diagonals.values_mut() {
        diagonal.sort_unstable();
        let mut iter = diagonal.iter().copied();
        // Each diagonal contains at least one element
        let mut first = iter.next().unwrap();
        let mut last = first;
        let mut group_kmers = 1;
        for pair in iter {
            if pair[0] > last[0] + max_distance {
                if group_kmers >= min_kmers {
                    add_group(first, last);
                }
                group_kmers = 1;
                first = pair;
            } else {
                group_kmers += 1;
            }
            last = pair;
        }

        if group_kmers >= min_kmers {
            add_group(first, last);
        }
    }

    groups
}

#[inline]
fn calc_seed_alignment_score(
    query: &query_file::Entry,
    target: &db_file::Entry,
    query_range: RangeInclusive<usize>,
    target_range: RangeInclusive<usize>,
    cli: &Cli,
) -> Reactivity {
    let query_seed = &query.reactivity()[query_range.clone()];
    let target_seed = &target.reactivity[target_range.clone()];

    let mut seed_score = calc_seed_alignment_score_from_reactivity(query_seed, target_seed, cli);
    if cli.alignment_args.align_score_seq {
        seed_score += calc_seed_alignment_score_from_sequence(
            &query.sequence()[query_range],
            &target.sequence[target_range],
            cli,
        );
    }

    seed_score
}

trait SequenceEntry {
    type Reactivity;

    fn name(&self) -> &str;
    fn sequence(&self) -> &[Base];
    fn reactivity(&self) -> &[Self::Reactivity];
    fn molecule(&self) -> Molecule;
}

fn write_cli_to_file(cli: &Cli) -> anyhow::Result<()> {
    use io::Write;

    let mut file =
        File::create(cli.output.join("params.out")).context("Unable to create params.out file")?;
    file.write_all(&toml_edit::ser::to_vec(cli).context("Unable to convert cli to TOML")?)
        .context("Unable to write to params.out")?;

    Ok(())
}

fn create_csv_from_writer<W: io::Write>(writer: W) -> csv::Writer<W> {
    csv::WriterBuilder::new()
        .delimiter(b'\t')
        .quote_style(csv::QuoteStyle::Necessary)
        .from_writer(writer)
}

fn display_scientific<T>(x: &T) -> String
where
    T: Float + FromPrimitive + Display + fmt::LowerExp,
{
    if *x >= T::from_f32(0.1).unwrap() {
        format!("{x:.3}")
    } else {
        format!("{x:.3e}")
    }
}

fn display_scientific_opt<T>(x: &Option<T>) -> String
where
    T: Float + FromPrimitive + Display + fmt::LowerExp,
{
    x.as_ref().map(display_scientific).unwrap_or_default()
}

struct ResultFileFormat<'a> {
    db_name: &'a str,
    db_range: RangeInclusive<usize>,
    query_name: &'a str,
    query_range: RangeInclusive<usize>,
}

impl<'a> From<&'a QueryResult> for ResultFileFormat<'a> {
    fn from(result: &'a QueryResult) -> Self {
        let &QueryResult {
            query: ref query_name,
            db_entry: ref db_name,
            query_start,
            query_end,
            db_start,
            db_end,
            ..
        } = result;

        let db_range = db_start..=db_end;
        let query_range = query_start..=query_end;

        Self {
            db_name,
            db_range,
            query_name,
            query_range,
        }
    }
}

impl fmt::Display for ResultFileFormat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            db_name,
            db_range,
            query_name,
            query_range,
        } = self;

        write!(
            f,
            "{}_{}-{}_{}_{}-{}",
            db_name,
            db_range.start(),
            db_range.end(),
            query_name,
            query_range.start(),
            query_range.end()
        )
    }
}

fn write_results_reactivity(
    results: &[QueryResult],
    db_entries: &[db_file::Entry],
    query_entries: &[query_file::Entry],
    output_dir: &Path,
) -> Result<(), anyhow::Error> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Data<'a> {
        query: GappedReactivity<'a, ReactivityWithPlaceholder>,
        query_id: &'a str,
        query_from: usize,
        query_to: usize,
        target: GappedReactivity<'a, ReactivityWithPlaceholder>,
        target_id: &'a str,
        target_from: usize,
        target_to: usize,
    }

    let mut reactivities_path = PathBuf::from(output_dir);
    reactivities_path.push("reactivities");
    fs::create_dir_all(&reactivities_path)
        .context("Unable to create output reactivities directory")?;

    let mut iter = results
        .iter()
        .filter(|result| matches!(result.status, QueryResultStatus::PassInclusionEvalue))
        .map(|result| {
            let query_entry = query_entries
                .iter()
                .find(|query| query.name() == &*result.query)
                .expect("query must be available in queries");

            let db_entry = db_entries
                .iter()
                .find(|db| db.name() == result.db_entry)
                .expect("target must be available in db");

            let query = &query_entry.reactivity()[result.query_start..=result.query_end];
            let target = &db_entry.reactivity()[result.db_start..=result.db_end];

            let data = Data {
                query: GappedReactivity {
                    reactivity: query,
                    alignment: result.alignment.query.to_ref(),
                },
                query_id: query_entry.name(),
                query_from: result.query_start,
                query_to: result.query_end,
                target: GappedReactivity {
                    reactivity: target,
                    alignment: result.alignment.target.to_ref(),
                },
                target_id: db_entry.name(),
                target_from: result.db_start,
                target_to: result.db_end,
            };

            (result, data)
        });

    let Some((result, data)) = iter.next() else {
        return Ok(());
    };

    let mut filename = format!("{}.json", ResultFileFormat::from(result));
    reactivities_path.push(&filename);

    let file = File::create(&reactivities_path)?;
    serde_json::to_writer(file, &data)?;

    iter.try_for_each(move |(result, data)| {
        filename.clear();
        write!(filename, "{}.json", ResultFileFormat::from(result)).unwrap();
        reactivities_path.set_file_name(&filename);

        let file = File::create(&reactivities_path)?;
        serde_json::to_writer(file, &data)?;
        Ok(())
    })
}

#[derive(Debug, Default, Clone, Copy, Eq, Hash, PartialEq)]
enum Molecule {
    Dna,
    Rna,
    #[default]
    Unknown,
}

#[derive(Debug, Default)]
struct BasePairsCount {
    pub total: usize,
    pub canonical: usize,
}

fn count_base_pairs(
    sequence: &GappedSequence,
    slice: Range<usize>,
    paired_last: usize,
) -> BasePairsCount {
    let slice_len = slice.end - slice.start;
    let canonical = sequence
        .get(slice)
        .unwrap()
        .iter()
        .zip({
            // I don't have the implementation for InclusiveRange
            #[allow(clippy::range_plus_one)]
            let range = (paired_last - slice_len)..(paired_last + 1);

            sequence.get(range).unwrap().iter().rev()
        })
        .filter_map(|(a, b)| a.to_base().and_then(|a| b.to_base().map(|b| (a, b))))
        .filter(|(a, b)| {
            matches!(
                (a, b),
                (Base::A | Base::G, Base::T)
                    | (Base::T, Base::A | Base::G)
                    | (Base::C, Base::G)
                    | (Base::G, Base::C)
            )
        })
        .count();

    BasePairsCount {
        total: slice_len,
        canonical,
    }
}

#[derive(Debug)]
struct AlifoldOnResult<'a, 'b> {
    pub dotbracket: Option<DotBracketBuffered<'b>>,
    pub ignore: bool,
    pub mfe: f32,
    pub gapped_data: GappedData<'a, <query_file::Entry as SequenceEntry>::Reactivity>,
    pub target_bp_support: f32,
    pub query_bp_support: f32,
}

#[inline]
fn alifold_on_result<'a, 'b>(
    result: &'a QueryAlignResult<AlignedSequence>,
    query_entry: &'a query_file::Entry,
    query_entry_orig: &'a query_file::Entry,
    cli: &Cli,
    dotbracket_results_buffer: &'b mut Vec<dotbracket::PairedBlock>,
    dotbracket_temp_buffer: &mut Vec<dotbracket::PartialPairedBlock>,
) -> AlifoldOnResult<'a, 'b> {
    let sequences = get_gapped_sequences(result, query_entry);
    let reactivities = get_gapped_reactivities(result, query_entry_orig);

    let (structure, mfe) = alifold_mfe(&sequences, &reactivities, cli);
    let dot_bracket_bp_count_result = DotBracket::from_bytes_with_buffer(
        structure.usable(),
        dotbracket_results_buffer,
        dotbracket_temp_buffer,
    )
    .map(|dotbracket| {
        let (target, query) = dotbracket
            .paired_blocks()
            .iter()
            .map(|paired_block| {
                let left = paired_block.left();
                let paired_last = paired_block.right().end - 1;
                let bp_count_target = count_base_pairs(&sequences[0], left.clone(), paired_last);
                let bp_count_query = count_base_pairs(&sequences[1], left.clone(), paired_last);

                (bp_count_target, bp_count_query)
            })
            .fold(
                (BasePairsCount::default(), BasePairsCount::default()),
                |(mut target, mut query), (other_target, other_query)| {
                    target.total += other_target.total;
                    target.canonical += other_target.canonical;

                    query.total += other_query.total;
                    query.canonical += other_query.canonical;

                    (target, query)
                },
            );
        (dotbracket, target, query)
    });

    let (dotbracket, ignore, target_bp_support, query_bp_support) =
        match dot_bracket_bp_count_result {
            Ok((dotbracket, bp_count_target, bp_count_query)) => {
                // It is ok to lose precision to evaluate fractions
                #[allow(clippy::cast_precision_loss)]
                let target_canonical_fraction =
                    bp_count_target.canonical as f64 / bp_count_target.total as f64;
                #[allow(clippy::cast_precision_loss)]
                let query_canonical_fraction =
                    bp_count_query.canonical as f64 / bp_count_query.total as f64;

                let min_bp_support = cli.alignment_folding_eval_args.min_bp_support.into();
                let ignore = target_canonical_fraction < min_bp_support
                    || query_canonical_fraction < min_bp_support;

                // We don't need the precision *after* the calculation
                #[allow(clippy::cast_possible_truncation)]
                (
                    Some(dotbracket),
                    ignore,
                    target_canonical_fraction as f32,
                    query_canonical_fraction as f32,
                )
            }
            Err(err) => {
                eprintln!("WARNING: cannot evaluate dot bracket for structure {structure}: {err}");
                (None, false, 0., 0.)
            }
        };

    let [target_sequence, query_sequence] = sequences;
    let [target_reactivity, query_reactivity] = reactivities;

    let gapped_data = GappedData::new_unchecked(
        query_sequence,
        &query_reactivity,
        target_sequence,
        &target_reactivity,
    );

    AlifoldOnResult {
        dotbracket,
        ignore,
        mfe,
        gapped_data,
        target_bp_support,
        query_bp_support,
    }
}

fn alifold_mfe<'a, S, R, RR>(sequences: &[S], reactivities: &[R], cli: &Cli) -> (Structure, f32)
where
    S: GappedSequenceLike + 'a,
    R: GappedReactivityLike<RR> + 'a,
    RR: ReactivityLike,
{
    let mut model_details = ModelDetails::default();
    model_details
        .set_no_lp(cli.folding_args.no_lonely_pairs)
        .set_no_gu_closure(cli.folding_args.no_closing_gu)
        .set_temperature(cli.folding_args.temperature.into())
        .set_ribo(cli.alignment_folding_eval_args.ribosum_scoring);

    *model_details.max_bp_span_mut() = cli.folding_args.max_bp_span.try_into().unwrap();
    let mut fold_compound = FoldCompound::new_comparative(
        sequences,
        Some(&model_details),
        FoldCompoundOptions::DEFAULT,
    )
    .unwrap();

    fold_compound
        .add_shape_reactivity_for_deigan_consensus_structure_prediction(
            reactivities,
            cli.folding_args.slope.into(),
            cli.folding_args.intercept.into(),
            FoldCompoundOptions::DEFAULT,
        )
        .unwrap();

    fold_compound.minimum_free_energy()
}

fn get_gapped_sequences<'a>(
    query_align_result: &'a QueryAlignResult<AlignedSequence>,
    query_entry: &'a query_file::Entry,
) -> [GappedSequence<'a>; 2] {
    let query = Sequence {
        bases: &query_entry.sequence()[query_align_result.query.clone()],
        molecule: query_entry.molecule(),
    };
    let target = Sequence {
        bases: &query_align_result.db_entry_orig.sequence()[query_align_result.db.clone()],
        molecule: query_align_result.db_entry_orig.molecule(),
    };

    [
        GappedSequence::new(target, &query_align_result.alignment.target),
        GappedSequence::new(query, &query_align_result.alignment.query),
    ]
}

fn get_gapped_reactivities<'a>(
    query_align_result: &'a QueryAlignResult<AlignedSequence>,
    query_entry_orig: &'a query_file::Entry,
) -> [GappedReactivity<'a, <query_file::Entry as SequenceEntry>::Reactivity>; 2] {
    let query_reactivity = &query_entry_orig.reactivity()[query_align_result.query.clone()];
    let target_reactivity =
        &query_align_result.db_entry_orig.reactivity()[query_align_result.db.clone()];

    [
        GappedReactivity {
            reactivity: target_reactivity,
            alignment: query_align_result.alignment.target.to_ref(),
        },
        GappedReactivity {
            reactivity: query_reactivity,
            alignment: query_align_result.alignment.query.to_ref(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};

    use crate::aligner::{AlignedSequenceRef, BaseOrGap};

    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn base_alignment_score() {
        let cli = dummy_cli();
        let cli = Cli {
            max_reactivity: 1.2,
            ..cli
        };

        assert_eq!(calc_base_alignment_score(1.1.into(), 1.2.into(), &cli), 0.);
        assert_eq!(
            calc_base_alignment_score(f32::NAN.into(), 1.2.into(), &cli),
            cli.alignment_args.align_match_score.0.start,
        );
        assert_eq!(
            calc_base_alignment_score(1.1.into(), (-999.).into(), &cli),
            cli.alignment_args.align_match_score.0.start,
        );
        assert!(
            (calc_base_alignment_score(0.2.into(), 0.4.into(), &cli) - 1.).abs() < f32::EPSILON
        );
        assert!(
            (calc_base_alignment_score(0.4.into(), 0.2.into(), &cli) - 1.).abs() < f32::EPSILON
        );
        assert!(
            (calc_base_alignment_score(0.1.into(), 0.6.into(), &cli) + 0.5).abs() < f32::EPSILON
        );
        assert!(
            (calc_base_alignment_score(0.1.into(), 0.8.into(), &cli) + 2.071_428_5).abs()
                < f32::EPSILON * 10.
        );
        assert!(
            (calc_base_alignment_score(0f32.into(), cli.max_reactivity.into(), &cli) + 6.).abs()
                < f32::EPSILON
        );
    }

    const QUERY_SEQUENCE: [u8; 200] = *b"GATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATCTGATACTGGCAAGCTTGAGTCTCGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGGCCCCCTGGACGAAGACTGACGCTCAGGTGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGG";
    #[allow(clippy::approx_constant)]
    const QUERY: [Reactivity; 200] = [
        0.052, 0.046, 0.108, 0.241, 0.221, 1.224, 0.246, 0.846, 1.505, 0.627, 0.078, 0.002, 0.056,
        0.317, 0.114, 0.157, 0.264, 1.016, 2.925, 2.205, 1.075, 1.210, 0.191, 0.016, 0.045, 0.015,
        0.087, 0.572, 0.052, 0.157, 0.796, 2.724, 0.027, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004,
        0.003, 0.063, 0.144, 0.072, 0.054, 0.096, 0.112, 0.002, 0.000, 0.019, 0.026, 0.021, 1.022,
        2.108, 0.111, 0.000, 0.007, 0.000, 0.002, 0.000, 0.010, 0.037, 0.078, 0.152, 0.355, 1.738,
        0.715, 0.211, 0.179, 0.036, 0.046, 0.159, 0.257, 0.312, 0.931, 0.798, 0.618, 0.935, 0.147,
        0.015, 0.014, 0.031, 0.147, 0.149, 0.577, 1.052, 1.410, 0.487, 0.636, 0.238, 0.286, 0.462,
        1.586, 1.683, 0.597, 1.165, 1.265, 2.094, 0.422, 0.462, 1.900, 4.055, 0.481, 0.511, 0.087,
        1.217, 1.180, 0.094, 0.018, 0.033, 0.081, 0.148, 0.163, 0.160, 1.019, 0.339, 0.507, 1.039,
        0.824, 0.122, 0.420, 0.429, 0.913, 1.383, 0.610, 0.417, 0.825, 0.743, 0.433, 0.401, 0.993,
        0.497, 0.404, 0.407, 0.316, 0.017, 0.005, 0.046, 0.072, 0.037, 0.091, 0.282, 0.203, 0.033,
        0.004, 0.021, 0.262, 0.157, 0.050, 0.019, 0.059, 0.102, 0.083, 0.066, 0.040, 0.075, 0.061,
        0.573, 0.631, 0.427, 0.265, 1.190, 0.066, 0.042, 0.085, 0.424, 0.413, 0.375, 0.447, 0.035,
        0.045, 0.037, 0.242, 0.221, 0.157, 0.170, 0.370, 1.238, 0.743, 0.571, 0.138, 0.837, 0.859,
        0.042, 0.021, 0.080, 0.318, 0.195, 0.792, 1.581, 1.058, 2.004, 1.512, 2.273, 1.256, 0.036,
        0.005, 0.094, 0.091, 0.464, 0.741,
    ];

    const DB_SEQUENCE: [u8; 1553] =
        *b"TTTATCGGAGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCAAGTCGAGCGGACAGATGGGAGCTTGC\
           TCCCTGATGTTAGCGGCGGACGGGTGAGTAACACGTGGGTAACCTGCCTGTAAGACTGGGATAACTCCGGGAAACCGGGGCTAATA\
           CCGGATGGTTGTTTGAACCGCATGGTTCAAACATAAAAGGTGGCTTCGGCTACCACTTACAGATGGACCCGCGGCGCATTAGCTAG\
           TTGGTGAGGTAACGGCTCACCAAGGCAACGATGCGTAGCCGACCTGAGAGGGTGATCGGCCACACTGGGACTGAGACACGGCCCAG\
           ACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCGCAATGGACGAAAGTCTGACGGAGCAACGCCGCGTGAGTGATGAAGGTTTTCG\
           GATCGTAAAGCTCTGTTGTTAGGGAAGAACAAGTACCGTTCGAATAGGGCGGTACCTTGACGGTACCTAACCAGAAAGCCACGGCT\
           AACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTTGTCCGGAATTATTGGGCGTAAAGGGCTCGCAGGCGGTTTCTT\
           AAGTCTGATGTGAAAGCCCCCGGCTCAACCGGGGAGGGTCATTGGAAACTGGGGAACTTGAGTGCAGAAGAGGAGAGTGGAATTCC\
           ACGTGTAGCGGTGAAATGCGTAGAGATGTGGAGGAACACCAGTGGCGAAGGCGACTCTCTGGTCTGTAACTGACGCTGAGGAGCGA\
           AAGCGTGGGGAGCGAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGAGTGCTAAGTGTTAGGGGGTTTCCGCCCCTT\
           AGTGCTGCAGCTAACGCATTAAGCACTCCGCCTGGGGAGTACGGTCGCAAGACTGAAACTCAAAGGAATTGACGGGGGCCCGCACA\
           AGCGGTGGAGCATGTGGTTTAATTCGAAGCAACGCGAAGAACCTTACCAGGTCTTGACATCCTCTGACAATCCTAGAGATAGGACG\
           TCCCCTTCGGGGGCAGAGTGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGAGATGTTGGGTTAAGTCCCGCAACGAGCGC\
           AACCCTTGATCTTAGTTGCCAGCATTCAGTTGGGCACTCTAAGGTGACTGCCGGTGACAAACCGGAGGAAGGTGGGGATGACGTCA\
           AATCATCATGCCCCTTATGACCTGGGCTACACACGTGCTACAATGGACAGAACAAAGGGCAGCGAAACCGCGAGGTTAAGCCAATC\
           CCACAAATCTGTTCTCAGTTCGGATCGCAGTCTGCAACTCGACTGCGTGAAGCTGGAATCGCTAGTAATCGCGGATCAGCATGCCG\
           CGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACCACGAGAGTTTGTAACACCCGAAGTCGGTGAGGTAACCTTTTA\
           GGAGCCAGCCGCCGAAGGTGGGACAGATGATTGGGGTGAAGTCGTAACAAGGTAGCCGTATCGGAAGGTGCGGCTGGATCACCTCC\
           TTTCT";

    const DB: [Reactivity; 1553] = {
        const NAN: Reactivity = Reactivity::NAN;
        [
            NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, 0.062, 0.341, 1.105,
            0.021, 0.320, 0.000, 0.054, 0.082, 0.289, 0.665, 0.174, 0.242, 1.248, 0.718, 0.035,
            0.178, 0.000, 0.273, 0.343, 0.000, 0.000, 0.175, 0.294, 0.142, 0.322, 0.885, 0.279,
            0.058, 0.000, 0.007, 0.000, 0.121, 0.106, 0.455, 0.137, 0.092, 0.112, 0.068, 0.159,
            0.800, 0.593, 0.715, 0.361, 0.581, 0.216, 0.185, 0.038, 0.593, 0.238, 0.072, 0.177,
            0.392, 0.139, 0.357, 0.235, 0.199, NAN, 0.082, 0.273, 0.316, 0.958, NAN, 1.218, 0.314,
            0.041, 0.090, 0.457, 0.592, 1.882, 2.266, 5.088, 2.483, 1.443, 1.552, 0.000, 0.000,
            0.101, 0.311, 0.335, 0.134, 0.718, 0.180, 0.110, 0.167, 0.291, 0.379, 0.458, 3.607,
            1.904, 0.119, 0.304, 0.201, 0.050, 0.309, 0.003, 0.200, 0.389, 0.590, 0.349, 0.465,
            0.184, 0.352, 0.160, 0.351, 0.130, 0.513, 0.593, 0.064, 0.061, 0.035, 0.000, 0.040,
            0.036, 0.077, 0.331, 0.083, 0.119, 0.581, 0.580, 0.011, 0.000, 0.095, 0.033, 0.126,
            0.019, 0.159, 0.421, 0.534, 0.240, 0.202, 0.000, 0.000, 0.061, 0.064, 0.179, 0.053,
            0.071, 0.119, 0.117, 0.114, 0.072, 0.073, 0.029, 0.107, 0.204, 0.279, 0.513, 0.010,
            0.084, 0.071, 0.046, 0.122, 0.049, 0.031, 0.228, 0.138, 0.201, 1.015, 0.777, 0.067,
            0.212, 0.034, 0.125, 0.309, 0.527, 3.717, 6.653, 0.267, 0.149, 0.014, 0.000, 0.000,
            0.000, 0.055, 0.103, 0.000, 0.104, 0.103, 1.352, 2.460, 1.574, 1.616, 0.000, 0.000,
            0.060, 0.038, 0.006, 0.064, 0.000, 0.044, 0.005, 0.000, 0.465, 0.035, 0.636, 0.087,
            0.350, 0.054, 0.018, 0.132, 0.049, 0.073, 0.440, 0.874, 1.338, 4.775, 3.409, 0.184,
            0.188, 0.252, 0.007, 0.000, 0.019, 0.158, 0.000, 0.021, 0.016, NAN, 0.070, 0.062,
            0.162, 0.152, 0.203, 0.045, 0.075, 0.136, 0.023, 0.032, 0.000, 0.062, 0.076, 0.188,
            0.183, 0.123, 0.124, 0.328, 3.735, 0.864, 3.097, 0.264, 0.084, 0.166, 0.797, 7.286,
            1.166, 0.124, 0.026, 0.000, 0.011, 0.000, 0.056, 0.081, 0.009, 0.090, 0.248, 0.382,
            0.809, NAN, 1.992, 0.118, 0.000, 0.000, 0.002, 0.000, 0.000, 0.000, 0.019, 0.302,
            0.088, 0.000, 0.058, NAN, 3.595, 1.139, 0.214, 0.010, 0.055, 0.014, 0.016, 0.075,
            0.050, 0.085, 0.112, 0.016, 0.058, 0.231, 0.513, 0.043, 0.151, 0.577, 0.522, 0.419,
            0.243, 1.276, 0.986, 0.234, 0.310, 0.445, 0.410, 0.498, 0.303, 0.306, 0.004, 0.068,
            0.030, 0.034, 0.279, 0.000, 0.190, 0.179, 0.088, 0.154, 0.265, 0.821, 0.242, 0.232,
            0.397, 0.353, 0.124, 0.521, 0.385, 0.339, 0.651, 0.865, 0.103, 0.023, 0.038, 0.011,
            0.000, 0.130, 0.327, 0.177, 0.068, 1.151, 0.052, 0.000, 0.682, 1.551, 7.803, 1.643,
            0.125, 0.132, 0.099, 0.068, 0.056, 0.077, 0.095, 0.145, 0.067, 0.166, 0.355, 1.197,
            0.435, 0.202, 0.160, 0.457, 0.139, 0.498, 0.246, 0.327, 0.444, 0.190, 0.061, 0.000,
            0.172, 0.453, 0.286, 1.260, 1.281, 0.211, 0.043, 0.023, 0.109, 0.662, 0.505, 0.229,
            0.442, 0.050, 0.000, 0.091, 0.137, 0.156, 0.115, 0.239, 0.102, 0.077, 0.173, 0.060,
            0.205, 0.154, 0.179, 0.000, 0.072, 0.006, 0.186, 0.133, 0.370, 0.737, 0.894, 0.328,
            0.442, 0.255, 1.351, 1.007, 6.596, 25.671, 2.392, 1.200, 0.636, 1.735, 2.845, 1.143,
            1.666, 2.465, 7.192, 12.230, 6.393, 0.498, 0.155, 0.205, 0.292, 0.280, 0.465, 0.058,
            0.272, 0.420, 0.190, 0.097, 0.000, 0.288, 0.303, 0.851, 0.855, 0.303, 0.176, 0.087,
            0.114, 0.104, 0.168, 0.528, 0.745, 0.571, 0.734, 0.302, 0.252, 0.330, 0.000, 0.011,
            0.012, 0.078, 0.000, NAN, 0.038, 0.092, 0.263, 0.050, 0.108, 0.286, 0.673, 0.833,
            3.207, NAN, 10.814, 0.050, 0.000, 0.000, 0.000, 0.000, 0.019, 0.014, 0.000, 0.017,
            0.019, 0.071, 0.075, 0.135, 0.000, 0.171, 0.039, 0.069, 0.062, 0.064, 0.004, 0.006,
            0.000, 0.012, 0.041, 0.060, 0.017, 0.958, 5.108, 1.616, 0.364, 0.312, 0.077, 0.018,
            0.047, 0.089, 0.141, 0.471, 0.363, 0.399, 0.174, 0.167, 0.315, 0.105, 0.146, 0.089,
            0.033, 0.043, 0.112, 0.812, 0.516, 0.320, 0.238, 0.105, 0.088, 0.249, 0.976, 0.231,
            1.131, 0.224, 0.444, 0.693, 1.946, 6.156, 0.440, 1.818, 1.084, 0.342, 0.007, 0.034,
            0.003, 0.026, 0.185, 0.356, 0.000, 0.000, 0.076, 0.419, 0.349, 0.110, 0.053, 0.043,
            0.128, 0.047, 0.169, NAN, 0.090, 0.049, 0.026, 0.220, 0.649, 1.070, 1.380, 2.489,
            1.953, 2.318, 2.673, 4.799, 0.757, 0.147, 0.049, 0.203, 0.091, 0.290, 0.142, 0.520,
            1.388, 0.747, 0.818, 0.290, 0.079, 0.260, 0.343, 0.636, 0.296, 1.774, 4.121, 32.001,
            0.089, 0.147, 0.000, 0.000, 0.040, 0.007, 0.044, 0.077, 0.146, 0.101, 0.160, 0.105,
            0.000, 0.075, 0.122, 0.193, 0.065, 0.090, 0.449, 0.265, 1.419, 0.316, 0.323, 0.626,
            0.264, 0.000, 0.000, 0.018, 0.000, 0.035, 0.048, 0.136, 0.325, 1.744, 1.853, 0.446,
            0.829, 0.157, 0.043, 0.040, 0.040, 0.090, 0.309, 1.043, 0.561, 0.497, 0.752, 8.159,
            0.041, 0.067, 0.055, 0.209, 0.294, 0.093, 0.045, 0.208, 0.240, 0.090, 0.352, 0.461,
            0.049, 0.288, NAN, 0.091, 0.124, 0.104, 0.414, 0.743, 0.115, 0.028, 0.000, 0.044,
            0.003, 0.054, 0.019, 0.089, 0.282, 0.244, 1.372, 1.179, 0.682, 0.333, 0.039, 0.026,
            0.134, 0.138, 0.593, 0.376, 0.971, 0.564, 1.306, 1.717, 0.280, 0.047, 0.047, 0.115,
            0.321, 0.800, 1.384, 2.098, 2.749, 1.163, 1.298, 0.832, 0.661, 1.129, 3.666, 3.748,
            1.502, 2.923, 3.276, 4.844, 1.418, 2.402, 7.579, 16.169, 0.797, 0.972, 0.208, 2.506,
            3.204, 2.611, 0.348, 0.490, 0.356, 0.392, 0.492, 0.342, 1.221, 0.629, 1.097, 0.296,
            0.736, 0.490, 0.235, 0.324, 0.565, 2.834, 2.200, 0.281, 0.814, 0.686, 0.402, 0.215,
            0.632, 0.336, 0.266, 0.396, 0.000, 0.009, 0.000, 0.000, 0.000, 0.022, 0.036, 0.161,
            0.261, 0.053, 0.068, 0.036, 0.194, 0.426, 0.578, 6.251, 0.012, 0.086, 0.020, 0.056,
            0.050, 0.000, 0.047, 0.433, 0.703, 0.756, 0.692, 1.834, 0.754, 0.780, 0.283, 0.688,
            0.436, 0.436, 0.391, 0.000, 0.093, 0.048, 0.585, 0.698, 0.396, 0.331, 1.120, 2.582,
            1.226, 0.738, 0.200, 0.601, 0.290, 0.049, 0.117, 0.268, 0.708, 0.497, 2.001, 3.928,
            2.840, 4.899, 3.712, 5.464, 2.886, 0.069, 0.000, 0.262, 0.130, 0.717, 0.970, 0.870,
            0.394, 0.748, 0.536, 0.053, 0.000, 0.064, 0.083, 0.325, 0.032, 0.163, 1.630, 1.937,
            0.330, 2.792, 3.440, 3.613, 2.724, 1.081, 2.358, 0.049, 0.013, 0.019, 0.000, 0.000,
            0.035, 0.151, 0.202, 0.316, 0.062, 0.070, 0.632, 0.406, 0.568, 0.306, 0.493, 0.228,
            0.222, 0.364, 2.987, 0.949, 6.935, 9.264, 2.962, 5.298, 1.860, 0.203, 0.000, 0.000,
            0.000, 0.004, 0.120, 0.128, 0.157, 0.331, 0.184, 0.114, 0.135, 0.062, 0.082, 0.030,
            0.048, 0.092, 0.097, 0.057, 0.063, 0.164, 0.000, 0.024, 0.175, 0.500, 1.068, 0.210,
            0.087, 0.000, 0.000, 0.000, 0.000, 0.000, 0.106, 0.131, 0.564, 0.000, 0.125, 0.928,
            0.648, 0.565, 1.062, 0.832, 0.102, 0.235, 0.269, 0.310, 0.341, 0.000, 0.055, 0.000,
            0.044, 0.285, 0.622, 0.026, 0.486, 0.019, 0.000, 0.061, 0.000, 0.887, 0.888, 0.356,
            0.452, 0.109, 0.000, 0.787, 0.127, 0.466, 0.745, 0.849, 0.985, 0.369, 0.836, 0.997,
            1.295, 0.061, 0.000, 0.140, 0.488, 0.286, 0.247, 0.449, 0.601, 0.052, 0.000, 0.000,
            0.129, 0.000, 0.000, 0.071, 0.094, 0.070, 0.000, 0.031, 0.002, 0.030, 0.067, 0.070,
            0.104, 0.000, 0.000, 0.000, 0.000, 0.027, 0.022, 0.038, 0.046, 0.215, 0.064, 0.095,
            0.147, 0.091, 0.121, 0.080, 0.147, 0.722, 2.543, 2.376, 2.516, 5.340, 2.277, 1.973,
            1.131, 2.245, 0.904, 1.524, 1.088, 0.368, 0.666, 0.243, 0.165, 0.196, 0.034, 0.134,
            0.403, 0.389, 1.234, 0.059, 0.000, 0.032, 0.072, 0.353, 0.113, 2.816, 4.629, 1.820,
            0.271, 0.110, 0.263, 0.545, 0.139, 0.003, 0.045, 0.000, 0.028, 0.064, 0.188, 0.158,
            0.530, 0.203, 0.416, 0.156, 0.010, 0.000, 0.046, 0.085, 0.115, 0.089, 0.611, 0.652,
            0.143, 0.107, 0.086, 0.164, 0.084, 0.783, 1.817, 1.600, 0.041, 0.061, 0.058, 0.780,
            1.567, 4.661, 8.240, 3.039, 0.083, 0.009, 0.121, 0.111, 0.016, 0.111, 0.090, 0.108,
            0.339, 2.636, 0.279, 0.593, 0.778, 0.684, 0.548, 2.324, 2.113, 1.190, NAN, 2.556,
            1.137, 0.206, 0.034, 0.108, 0.520, 0.783, 0.414, 0.776, 0.197, 0.162, 0.153, 0.375,
            0.547, 0.321, 0.450, 0.028, 0.032, 0.109, 0.091, 0.275, 0.433, 0.419, 0.302, 0.336,
            0.169, 0.438, 0.349, 0.842, 0.165, 0.286, 0.393, 0.389, 0.757, 0.643, 0.246, 0.470,
            0.578, 0.455, 1.020, 1.294, 1.442, 4.392, 0.737, 0.016, 0.060, 0.060, 0.089, 0.417,
            0.429, 0.483, 0.042, 0.041, 0.047, 0.064, 0.914, 0.965, 1.207, 1.162, 2.204, 0.000,
            0.021, 0.139, 0.631, 4.257, 0.748, 0.011, 0.120, 0.001, 0.010, 0.241, 0.049, 0.151,
            0.171, 0.257, 0.092, 0.053, 0.260, 0.491, 1.636, 1.045, 1.647, 1.923, 3.996, 1.572,
            1.294, 1.326, 1.102, 1.135, 0.767, 0.357, 0.265, 0.578, 4.447, 2.240, 0.344, 0.025,
            0.074, 0.126, 0.180, 0.336, 0.274, 0.154, 0.356, 0.131, 0.752, 1.198, 0.332, 0.010,
            0.076, 0.181, 0.328, 0.617, 0.995, 2.113, 7.438, 1.289, 2.026, 0.822, 0.090, 0.062,
            0.105, 0.096, 0.020, 0.202, 0.333, 0.158, 0.170, 0.838, 1.493, 3.024, 0.406, 0.089,
            0.020, 0.445, 1.038, 2.202, 0.056, 0.073, 0.039, 0.092, 0.237, 0.470, 0.247, 1.520,
            2.555, 2.694, 1.011, 0.253, 0.409, 0.361, 0.550, 2.247, 2.069, 0.000, 0.010, 0.217,
            0.843, 8.720, 10.132, 1.607, 2.414, 0.002, 0.000, 0.014, 0.047, 0.060, 0.000, 0.035,
            0.088, 0.048, 0.124, 0.038, 11.194, 0.165, 0.000, 1.831, 1.847, 0.085, 0.053, 0.000,
            0.000, 0.000, 0.004, 0.111, 0.101, 0.462, 1.792, 0.026, 0.029, 0.028, 0.042, 0.070,
            0.236, 0.222, 0.662, 0.246, 0.090, 0.158, 0.356, 0.506, 0.202, 0.471, 6.149, 8.865,
            2.984, 2.725, 0.159, 0.176, 0.178, 0.138, 0.029, 0.001, 0.100, 0.346, 0.344, 0.297,
            0.084, 0.104, 0.237, 0.236, 0.147, 0.110, 0.076, 0.303, 1.944, 0.288, 1.140, 0.936,
            0.029, 0.060, 0.111, 0.401, 1.406, 0.079, 0.072, 0.072, 0.000, 0.021, 0.225, 0.880,
            0.714, 0.471, 20.160, 5.704, 2.012, 0.052, 0.047, 0.119, 0.146, 0.070, 0.004, 0.000,
            0.065, 0.000, 0.000, 0.000, 0.000, 0.008, 0.000, 0.000, 0.036, 0.077, 0.082, 0.052,
            0.048, 0.065, 0.098, 0.006, 0.137, 0.017, 0.000, 0.000, 0.032, 0.000, 0.015, 0.000,
            0.004, 0.033, 0.000, 0.009, 0.694, 4.143, 3.463, 0.165, 0.538, 0.897, 0.162, 0.151,
            0.027, 0.001, 0.268, 0.335, 0.167, 0.084, 0.073, 0.022, 0.000, 0.000, 0.014, 0.006,
            0.065, 0.013, 0.099, 0.125, 1.148, 1.327, 0.051, 0.245, 0.132, 0.123, 3.628, 0.076,
            0.000, 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.272, 0.000, 0.000, 0.163, 0.118,
            0.966, 0.186, 0.338, 0.920, 0.064, 0.042, 0.007, 0.018, 0.076, 0.759, 0.047, 0.475,
            1.912, 4.365, 4.122, 3.026, 0.722, 1.905, 0.223, 0.371, 2.103, 2.450, 2.442, 1.295,
            0.129, 0.196, 0.103, 0.471, 0.323, 0.536, 0.412, 0.104, 0.607, 0.490, 0.369, 0.964,
            0.558, 0.588, 0.450, 0.064, 1.154, 1.427, 0.101, 0.634, 1.141, 0.567, 0.301, 0.284,
            0.109, 0.063, 0.073, 0.077, 0.296, 0.568, 0.990, 1.168, 0.749, 2.836, 3.365, 0.980,
            0.431, 0.381, 1.493, 0.817, 5.718, 1.144, 0.621, 4.504, 4.093, 5.693, 0.465, 1.289,
            3.555, 5.129, 9.864, 19.367, 2.306, 0.119, 0.158, 0.067, 0.033, 0.061, 0.132, 0.178,
            0.166, 0.000, 0.023, 0.034, 0.012, 0.000, 0.056, 0.048, 0.000, 0.177, 0.000, 0.000,
            0.000, 0.000, 0.003, 0.000, 0.000, 0.052, 0.270, 0.116, NAN, NAN, NAN, NAN, NAN, NAN,
            NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
            NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
            NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
            NAN, NAN, NAN, NAN, NAN, NAN, NAN,
        ]
    };

    const MATCHES: &[[usize; 2]] = &[
        [0, 17],
        [0, 676],
        [0, 1158],
        [0, 1380],
        [0, 1435],
        [1, 18],
        [1, 677],
        [1, 1159],
        [1, 1381],
        [2, 19],
        [2, 32],
        [2, 678],
        [2, 1160],
        [2, 1382],
        [3, 20],
        [3, 33],
        [3, 890],
        [4, 21],
        [4, 34],
        [4, 115],
        [4, 891],
        [5, 1001],
        [5, 1426],
        [6, 614],
        [6, 658],
        [6, 670],
        [6, 754],
        [6, 877],
        [6, 1002],
        [6, 1427],
        [7, 755],
        [7, 1236],
        [7, 1428],
        [8, 581],
        [8, 1429],
        [9, 1255],
        [10, 6],
        [10, 963],
        [10, 1314],
        [12, 1168],
        [13, 1111],
        [13, 1169],
        [14, 1033],
        [14, 1170],
        [14, 1214],
        [15, 1034],
        [15, 1171],
        [15, 1215],
        [16, 1035],
        [16, 1216],
        [17, 425],
        [17, 849],
        [19, 427],
        [19, 825],
        [19, 851],
        [19, 1038],
        [19, 1406],
        [20, 335],
        [20, 1039],
        [20, 1236],
        [21, 326],
        [21, 336],
        [21, 810],
        [21, 879],
        [21, 981],
        [21, 1040],
        [21, 1237],
        [32, 355],
        [33, 954],
        [33, 1042],
        [35, 630],
        [35, 1317],
        [36, 77],
        [36, 631],
        [36, 802],
        [37, 78],
        [37, 632],
        [37, 803],
        [38, 79],
        [38, 534],
        [38, 633],
        [38, 804],
        [38, 930],
        [38, 1230],
        [54, 160],
        [54, 239],
        [54, 354],
        [54, 603],
        [54, 661],
        [54, 1129],
        [63, 129],
        [63, 443],
        [63, 580],
        [63, 659],
        [63, 755],
        [63, 1003],
        [63, 1428],
        [64, 572],
        [64, 756],
        [65, 1166],
        [66, 184],
        [66, 1167],
        [66, 1256],
        [67, 185],
        [67, 620],
        [67, 1110],
        [67, 1168],
        [67, 1198],
        [67, 1257],
        [68, 186],
        [68, 621],
        [68, 1111],
        [68, 1169],
        [68, 1258],
        [69, 187],
        [69, 622],
        [69, 1033],
        [69, 1095],
        [69, 1170],
        [69, 1214],
        [69, 1259],
        [70, 188],
        [70, 383],
        [70, 623],
        [70, 1034],
        [70, 1171],
        [70, 1215],
        [70, 1260],
        [71, 189],
        [71, 1035],
        [71, 1172],
        [71, 1216],
        [71, 1261],
        [71, 1403],
        [72, 385],
        [74, 1013],
        [75, 683],
        [75, 810],
        [75, 1014],
        [75, 1165],
        [76, 131],
        [76, 1015],
        [77, 132],
        [77, 403],
        [77, 496],
        [77, 619],
        [77, 741],
        [78, 133],
        [78, 497],
        [78, 620],
        [78, 1359],
        [79, 134],
        [79, 498],
        [79, 621],
        [79, 1018],
        [80, 41],
        [80, 520],
        [80, 688],
        [80, 1019],
        [80, 1077],
        [80, 1104],
        [80, 1185],
        [81, 42],
        [81, 1020],
        [81, 1105],
        [82, 690],
        [83, 236],
        [83, 771],
        [84, 692],
        [86, 1295],
        [86, 1317],
        [87, 77],
        [87, 920],
        [87, 1296],
        [87, 1435],
        [88, 816],
        [88, 1394],
        [88, 1436],
        [89, 1395],
        [89, 1437],
        [90, 1416],
        [90, 1448],
        [93, 19],
        [93, 922],
        [93, 1230],
        [93, 1273],
        [94, 20],
        [94, 702],
        [94, 923],
        [95, 21],
        [95, 924],
        [96, 22],
        [96, 636],
        [96, 892],
        [96, 1233],
        [97, 23],
        [97, 304],
        [97, 545],
        [97, 1272],
        [97, 1363],
        [98, 894],
        [98, 1286],
        [98, 1364],
        [99, 1287],
        [100, 114],
        [100, 548],
        [104, 118],
        [104, 130],
        [104, 158],
        [104, 444],
        [104, 503],
        [104, 572],
        [104, 1265],
        [105, 119],
        [105, 287],
        [106, 292],
        [106, 882],
        [106, 955],
        [106, 1226],
        [107, 16],
        [107, 121],
        [107, 630],
        [107, 886],
        [107, 1227],
        [108, 17],
        [108, 122],
        [108, 887],
        [108, 1228],
        [109, 18],
        [109, 123],
        [109, 888],
        [109, 1229],
        [110, 19],
        [111, 1415],
        [112, 543],
        [112, 1114],
        [112, 1208],
        [112, 1416],
        [113, 544],
        [113, 870],
        [113, 1115],
        [113, 1417],
        [114, 545],
        [114, 871],
        [114, 1158],
        [114, 1363],
        [114, 1418],
        [115, 546],
        [115, 724],
        [115, 1159],
        [115, 1268],
        [115, 1286],
        [115, 1364],
        [115, 1419],
        [116, 141],
        [116, 547],
        [116, 801],
        [116, 1118],
        [116, 1190],
        [116, 1287],
        [116, 1420],
        [117, 608],
        [117, 874],
        [117, 1288],
        [118, 441],
        [118, 1289],
        [119, 323],
        [119, 657],
        [119, 876],
        [119, 1234],
        [119, 1290],
        [119, 1349],
        [120, 13],
        [120, 334],
        [121, 85],
        [121, 388],
        [121, 928],
        [121, 936],
        [122, 86],
        [122, 389],
        [122, 929],
        [122, 937],
        [122, 1000],
        [123, 1423],
        [124, 732],
        [125, 733],
        [126, 734],
        [127, 611],
        [128, 612],
        [128, 736],
        [129, 613],
        [129, 737],
        [131, 84],
        [131, 1176],
        [132, 38],
        [132, 85],
        [132, 928],
        [134, 404],
        [134, 742],
        [134, 1025],
        [134, 1249],
        [134, 1436],
        [135, 405],
        [135, 1026],
        [135, 1184],
        [135, 1250],
        [135, 1437],
        [136, 21],
        [136, 543],
        [137, 22],
        [137, 303],
        [137, 544],
        [137, 1271],
        [138, 23],
        [138, 304],
        [138, 545],
        [138, 1272],
        [139, 24],
        [139, 250],
        [139, 546],
        [139, 1147],
        [139, 1286],
        [140, 25],
        [140, 547],
        [140, 1148],
        [140, 1190],
        [141, 26],
        [141, 56],
        [141, 302],
        [141, 548],
        [141, 941],
        [141, 1149],
        [147, 4],
        [147, 182],
        [147, 493],
        [147, 662],
        [147, 962],
        [148, 494],
        [148, 618],
        [148, 663],
        [148, 1314],
        [149, 237],
        [149, 295],
        [149, 757],
        [149, 828],
        [150, 480],
        [150, 630],
        [150, 758],
        [150, 1109],
        [151, 337],
        [151, 481],
        [151, 631],
        [151, 1110],
        [151, 1224],
        [151, 1379],
        [152, 482],
        [152, 632],
        [152, 1111],
        [152, 1239],
        [152, 1380],
        [152, 1470],
        [163, 190],
        [163, 385],
        [163, 1036],
        [163, 1064],
        [163, 1262],
        [163, 1404],
        [166, 239],
        [166, 354],
        [166, 616],
        [166, 661],
        [166, 952],
        [166, 1129],
        [166, 1477],
        [167, 355],
        [167, 617],
        [167, 662],
        [167, 953],
        [168, 241],
        [168, 356],
        [168, 663],
        [168, 776],
        [168, 1181],
        [169, 357],
        [169, 1248],
        [170, 1249],
        [170, 1316],
        [171, 507],
        [171, 716],
        [171, 1250],
        [171, 1317],
        [172, 48],
        [172, 77],
        [172, 508],
        [172, 717],
        [172, 802],
        [173, 78],
        [173, 509],
        [173, 803],
        [174, 510],
        [177, 27],
        [177, 167],
        [177, 1020],
        [178, 28],
        [178, 168],
        [178, 681],
        [178, 1021],
        [179, 682],
        [182, 965],
        [182, 1133],
        [183, 76],
        [183, 562],
        [183, 791],
        [183, 1168],
        [184, 77],
        [184, 563],
        [184, 792],
        [184, 844],
        [184, 1169],
        [185, 78],
        [185, 564],
        [185, 793],
        [185, 845],
        [185, 1170],
        [186, 79],
        [186, 565],
        [186, 794],
        [186, 1171],
        [187, 80],
        [187, 795],
        [187, 1455],
        [187, 1482],
        [188, 81],
        [188, 796],
    ];

    const GROUPS: &[MatchRanges] = &[
        MatchRanges {
            db: 676..=689,
            query: 0..=13,
        },
        MatchRanges {
            db: 1158..=1171,
            query: 0..=13,
        },
        MatchRanges {
            db: 1380..=1393,
            query: 0..=13,
        },
        MatchRanges {
            db: 17..=32,
            query: 0..=15,
        },
        MatchRanges {
            db: 32..=45,
            query: 2..=15,
        },
        MatchRanges {
            db: 890..=902,
            query: 3..=15,
        },
        MatchRanges {
            db: 1001..=1013,
            query: 5..=17,
        },
        MatchRanges {
            db: 1426..=1440,
            query: 5..=19,
        },
        MatchRanges {
            db: 754..=766,
            query: 6..=18,
        },
        MatchRanges {
            db: 1168..=1182,
            query: 12..=26,
        },
        MatchRanges {
            db: 1214..=1227,
            query: 14..=27,
        },
        MatchRanges {
            db: 1033..=1051,
            query: 14..=32,
        },
        MatchRanges {
            db: 425..=438,
            query: 17..=30,
        },
        MatchRanges {
            db: 849..=862,
            query: 17..=30,
        },
        MatchRanges {
            db: 335..=347,
            query: 20..=32,
        },
        MatchRanges {
            db: 1236..=1248,
            query: 20..=32,
        },
        MatchRanges {
            db: 630..=644,
            query: 35..=49,
        },
        MatchRanges {
            db: 77..=90,
            query: 36..=49,
        },
        MatchRanges {
            db: 802..=815,
            query: 36..=49,
        },
        MatchRanges {
            db: 755..=767,
            query: 63..=75,
        },
        MatchRanges {
            db: 1166..=1183,
            query: 65..=82,
        },
        MatchRanges {
            db: 184..=200,
            query: 66..=82,
        },
        MatchRanges {
            db: 1256..=1272,
            query: 66..=82,
        },
        MatchRanges {
            db: 1110..=1122,
            query: 67..=79,
        },
        MatchRanges {
            db: 620..=634,
            query: 67..=81,
        },
        MatchRanges {
            db: 1033..=1046,
            query: 69..=82,
        },
        MatchRanges {
            db: 1214..=1227,
            query: 69..=82,
        },
        MatchRanges {
            db: 383..=396,
            query: 70..=83,
        },
        MatchRanges {
            db: 1013..=1031,
            query: 74..=92,
        },
        MatchRanges {
            db: 683..=713,
            query: 75..=105,
        },
        MatchRanges {
            db: 131..=145,
            query: 76..=90,
        },
        MatchRanges {
            db: 496..=509,
            query: 77..=90,
        },
        MatchRanges {
            db: 619..=632,
            query: 77..=90,
        },
        MatchRanges {
            db: 41..=53,
            query: 80..=92,
        },
        MatchRanges {
            db: 1104..=1116,
            query: 80..=92,
        },
        MatchRanges {
            db: 1295..=1307,
            query: 86..=98,
        },
        MatchRanges {
            db: 1435..=1448,
            query: 87..=100,
        },
        MatchRanges {
            db: 1394..=1406,
            query: 88..=100,
        },
        MatchRanges {
            db: 922..=935,
            query: 93..=106,
        },
        MatchRanges {
            db: 1230..=1244,
            query: 93..=107,
        },
        MatchRanges {
            db: 19..=34,
            query: 93..=108,
        },
        MatchRanges {
            db: 892..=905,
            query: 96..=109,
        },
        MatchRanges {
            db: 1363..=1375,
            query: 97..=109,
        },
        MatchRanges {
            db: 545..=559,
            query: 97..=111,
        },
        MatchRanges {
            db: 1286..=1298,
            query: 98..=110,
        },
        MatchRanges {
            db: 114..=134,
            query: 100..=120,
        },
        MatchRanges {
            db: 1226..=1240,
            query: 106..=120,
        },
        MatchRanges {
            db: 886..=899,
            query: 107..=120,
        },
        MatchRanges {
            db: 16..=30,
            query: 107..=121,
        },
        MatchRanges {
            db: 1415..=1431,
            query: 111..=127,
        },
        MatchRanges {
            db: 543..=558,
            query: 112..=127,
        },
        MatchRanges {
            db: 1114..=1129,
            query: 112..=127,
        },
        MatchRanges {
            db: 870..=887,
            query: 113..=130,
        },
        MatchRanges {
            db: 1158..=1170,
            query: 114..=126,
        },
        MatchRanges {
            db: 1363..=1375,
            query: 114..=126,
        },
        MatchRanges {
            db: 1286..=1301,
            query: 115..=130,
        },
        MatchRanges {
            db: 1234..=1261,
            query: 119..=146,
        },
        MatchRanges {
            db: 85..=97,
            query: 121..=133,
        },
        MatchRanges {
            db: 388..=400,
            query: 121..=133,
        },
        MatchRanges {
            db: 928..=940,
            query: 121..=133,
        },
        MatchRanges {
            db: 936..=948,
            query: 121..=133,
        },
        MatchRanges {
            db: 732..=807,
            query: 124..=199,
        },
        MatchRanges {
            db: 611..=624,
            query: 127..=140,
        },
        MatchRanges {
            db: 84..=96,
            query: 131..=143,
        },
        MatchRanges {
            db: 404..=416,
            query: 134..=146,
        },
        MatchRanges {
            db: 1025..=1037,
            query: 134..=146,
        },
        MatchRanges {
            db: 1436..=1448,
            query: 134..=146,
        },
        MatchRanges {
            db: 21..=37,
            query: 136..=152,
        },
        MatchRanges {
            db: 543..=559,
            query: 136..=152,
        },
        MatchRanges {
            db: 303..=315,
            query: 137..=149,
        },
        MatchRanges {
            db: 1271..=1283,
            query: 137..=149,
        },
        MatchRanges {
            db: 1147..=1160,
            query: 139..=152,
        },
        MatchRanges {
            db: 493..=505,
            query: 147..=159,
        },
        MatchRanges {
            db: 662..=674,
            query: 147..=159,
        },
        MatchRanges {
            db: 480..=493,
            query: 150..=163,
        },
        MatchRanges {
            db: 630..=643,
            query: 150..=163,
        },
        MatchRanges {
            db: 1109..=1122,
            query: 150..=163,
        },
        MatchRanges {
            db: 1379..=1391,
            query: 151..=163,
        },
        MatchRanges {
            db: 616..=628,
            query: 166..=178,
        },
        MatchRanges {
            db: 952..=964,
            query: 166..=178,
        },
        MatchRanges {
            db: 239..=252,
            query: 166..=179,
        },
        MatchRanges {
            db: 661..=674,
            query: 166..=179,
        },
        MatchRanges {
            db: 354..=368,
            query: 166..=180,
        },
        MatchRanges {
            db: 1248..=1261,
            query: 169..=182,
        },
        MatchRanges {
            db: 1316..=1328,
            query: 170..=182,
        },
        MatchRanges {
            db: 716..=728,
            query: 171..=183,
        },
        MatchRanges {
            db: 507..=521,
            query: 171..=185,
        },
        MatchRanges {
            db: 77..=89,
            query: 172..=184,
        },
        MatchRanges {
            db: 802..=814,
            query: 172..=184,
        },
        MatchRanges {
            db: 27..=39,
            query: 177..=189,
        },
        MatchRanges {
            db: 167..=179,
            query: 177..=189,
        },
        MatchRanges {
            db: 1020..=1032,
            query: 177..=189,
        },
        MatchRanges {
            db: 681..=693,
            query: 178..=190,
        },
        MatchRanges {
            db: 562..=576,
            query: 183..=197,
        },
        MatchRanges {
            db: 1168..=1182,
            query: 183..=197,
        },
        MatchRanges {
            db: 76..=92,
            query: 183..=199,
        },
        MatchRanges {
            db: 844..=856,
            query: 184..=196,
        },
    ];

    fn saturate_query<T, const N: usize>(query: [T; N], max_reactivity: T) -> [T; N]
    where
        T: FloatCore,
    {
        query.map(|n| if n.is_nan() { n } else { n.min(max_reactivity) })
    }

    struct TestData {
        cli: Cli,
        query_sequence: [Base; QUERY_SEQUENCE.len()],
        query: [ReactivityWithPlaceholder; QUERY.len()],
        db_sequence: [Base; DB_SEQUENCE.len()],
        db: [ReactivityWithPlaceholder; DB_SEQUENCE.len()],
    }

    pub(crate) fn dummy_cli() -> Cli {
        Cli::parse_from(["test", "--database", "test", "--query", "test"])
    }

    fn tweaked_cli() -> Cli {
        let cli = dummy_cli();
        Cli {
            kmer_lookup_args: cli::KmerLookupArgs {
                kmer_len: 12,
                max_kmer_dist: 10,
                match_kmer_seq: true,
                kmer_max_seq_dist: Some(Distance::Integral(200)),
                kmer_min_complexity: 0.2,
                min_kmers: 2,
                ..cli.kmer_lookup_args
            },
            max_reactivity: 1.5,
            ..cli
        }
    }

    impl TestData {
        fn new() -> Self {
            let cli = tweaked_cli();
            let query_sequence = QUERY_SEQUENCE.map(|c| Base::try_from(c).unwrap());

            let query =
                saturate_query(QUERY, cli.max_reactivity).map(ReactivityWithPlaceholder::from);
            let db_sequence = DB_SEQUENCE.map(|c| Base::try_from(c).unwrap());
            let db = DB.map(|x| {
                let reactivity = ReactivityWithPlaceholder::from(x);
                reactivity.get_non_nan().map_or(reactivity, |reactivity| {
                    reactivity.min(cli.max_reactivity).into()
                })
            });

            Self {
                cli,
                query_sequence,
                query,
                db_sequence,
                db,
            }
        }

        fn db_data(&self) -> DbData<'_> {
            DbData::new(&self.db_sequence, &self.db).unwrap()
        }
    }

    #[test]
    fn matching_kmers() {
        let data = TestData::new();
        let db_data = data.db_data();
        let mut matches =
            get_matching_kmers(&data.query, &data.query_sequence, &db_data, &data.cli).unwrap();
        matches.sort_unstable();
        assert_eq!(matches, MATCHES);
    }

    #[test]
    fn group_matching_kmers() {
        let cli = tweaked_cli();
        let mut groups = super::group_matching_kmers(MATCHES, &cli);
        groups.sort_unstable_by(|a, b| {
            a.query
                .start()
                .cmp(b.query.start())
                .then_with(|| a.query.end().cmp(b.query.end()))
                .then_with(|| a.db.start().cmp(b.db.start()))
                .then_with(|| a.db.end().cmp(b.db.end()))
        });
        assert_eq!(groups, GROUPS);
    }

    #[test]
    fn gini_index() {
        const DATA: &[f64] = &[
            9., 28., 43., 6., 21., 42., 4., 1., 50., 15., 32., 18., 37., 26., 7., 10., 24., 23.,
            40., 17., 35., 41., 2., 38., 36., 49., 3., 39., 45., 31., 46., 29., 22., 34., 11., 44.,
            13., 27., 5., 30., 25., 12., 33., 19., 47., 8., 48., 14., 20., 16.,
        ];

        assert_relative_eq!(super::gini_index(DATA), 0.327, epsilon = 0.0005);
    }

    #[test]
    fn check_saturate_query() {
        let query = saturate_query([f64::NAN, f64::INFINITY, -f64::INFINITY, 0., 1., 2.], 0.);

        assert!(query[3..].iter().all(|&n| n == 0.));
        assert!(query[0].is_nan());
        assert!(query[1] < f64::EPSILON);
        assert!(query[2] < 0. && query[2].is_infinite());
    }

    const QUERY2: [f32; 200] = {
        use std::f32::NAN;
        [
            0.885, 0.181, 0.189, 0.239, 0.531, 0.141, 0.126, 0.491, 0.648, 0.114, 0.171, 0.332,
            0.099, 0.601, 0.849, 1.000, 0.928, 0.221, 0.464, 0.482, 0.398, 0.082, 0.079, 0.141,
            0.649, 0.419, 0.172, 0.242, 0.391, 0.000, 0.000, 0.005, 0.220, 0.022, 0.270, 0.195,
            0.252, 0.140, 0.055, 0.060, 0.042, 0.150, 0.122, 0.146, 0.063, 0.000, 0.038, 0.042,
            0.000, 0.066, 0.075, 0.190, 0.144, 0.333, 0.025, 0.203, 0.374, 0.239, 0.219, 0.076,
            0.123, 0.427, 1.000, 0.698, 1.000, 1.000, 1.000, 1.000, 0.681, 1.000, 0.561, 0.632,
            0.716, 0.341, 0.412, 0.234, 0.142, 0.049, 0.029, 0.194, 0.346, 0.390, 0.842, 0.096,
            0.011, 0.054, 0.022, 0.135, 0.120, 0.868, 1.000, 1.000, 0.645, 0.327, 0.502, 0.611,
            0.236, 0.129, 0.095, 0.072, 0.041, NAN, 0.251, 0.222, 0.407, NAN, 0.040, 0.016, 0.023,
            NAN, 0.063, 0.075, 0.183, 0.071, 0.518, 0.520, 0.127, 0.040, NAN, 0.150, 0.055, NAN,
            NAN, 0.937, 1.000, 0.550, 0.028, 0.116, 0.380, 0.589, 1.000, 0.653, 0.081, 0.079,
            0.164, 0.117, 0.003, NAN, 0.069, 0.031, 0.084, 1.000, 0.397, 0.443, 0.767, 0.376,
            0.148, 0.443, 1.000, 0.486, 0.090, 0.352, 0.411, 0.768, 0.045, 0.034, 0.012, 0.034,
            0.028, 0.045, 0.085, 0.080, 0.065, 0.232, 0.234, 0.295, 0.346, 0.056, 0.040, 0.002,
            0.145, 0.709, 0.990, 1.000, 1.000, 1.000, 1.000, 0.820, 1.000, 0.641, 1.000, 1.000,
            1.000, 1.000, 1.000, 1.000, 0.252, 0.081, 0.240, 0.474, 1.000, 1.000, 1.000, 1.000,
            0.715, 0.003, 0.009, 0.021, 0.064, 0.168,
        ]
    };

    const DB2: [Reactivity; 1553] = [
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, 0.062, 0.341, 1.000, 0.021, 0.320, 0.000, 0.054,
        0.082, 0.289, 0.665, 0.174, 0.242, 1.000, 0.718, 0.035, 0.178, 0.000, 0.273, 0.343, 0.000,
        0.000, 0.175, 0.294, 0.142, 0.322, 0.885, 0.279, 0.058, 0.000, 0.007, 0.000, 0.121, 0.106,
        0.455, 0.137, 0.092, 0.112, 0.068, 0.159, 0.800, 0.593, 0.715, 0.361, 0.581, 0.216, 0.185,
        0.038, 0.593, 0.238, 0.072, 0.177, 0.392, 0.139, 0.357, 0.235, 0.199, -999.000, 0.082,
        0.273, 0.316, 0.958, -999.000, 1.000, 0.314, 0.041, 0.090, 0.457, 0.592, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 0.000, 0.000, 0.101, 0.311, 0.335, 0.134, 0.718, 0.180, 0.110,
        0.167, 0.291, 0.379, 0.458, 1.000, 1.000, 0.119, 0.304, 0.201, 0.050, 0.309, 0.003, 0.200,
        0.389, 0.590, 0.349, 0.465, 0.184, 0.352, 0.160, 0.351, 0.130, 0.513, 0.593, 0.064, 0.061,
        0.035, 0.000, 0.040, 0.036, 0.077, 0.331, 0.083, 0.119, 0.581, 0.580, 0.011, 0.000, 0.095,
        0.033, 0.126, 0.019, 0.159, 0.421, 0.534, 0.240, 0.202, 0.000, 0.000, 0.061, 0.064, 0.179,
        0.053, 0.071, 0.119, 0.117, 0.114, 0.072, 0.073, 0.029, 0.107, 0.204, 0.279, 0.513, 0.010,
        0.084, 0.071, 0.046, 0.122, 0.049, 0.031, 0.228, 0.138, 0.201, 1.000, 0.777, 0.067, 0.212,
        0.034, 0.125, 0.309, 0.527, 1.000, 1.000, 0.267, 0.149, 0.014, 0.000, 0.000, 0.000, 0.055,
        0.103, 0.000, 0.104, 0.103, 1.000, 1.000, 1.000, 1.000, 0.000, 0.000, 0.060, 0.038, 0.006,
        0.064, 0.000, 0.044, 0.005, 0.000, 0.465, 0.035, 0.636, 0.087, 0.350, 0.054, 0.018, 0.132,
        0.049, 0.073, 0.440, 0.874, 1.000, 1.000, 1.000, 0.184, 0.188, 0.252, 0.007, 0.000, 0.019,
        0.158, 0.000, 0.021, 0.016, -999.000, 0.070, 0.062, 0.162, 0.152, 0.203, 0.045, 0.075,
        0.136, 0.023, 0.032, 0.000, 0.062, 0.076, 0.188, 0.183, 0.123, 0.124, 0.328, 1.000, 0.864,
        1.000, 0.264, 0.084, 0.166, 0.797, 1.000, 1.000, 0.124, 0.026, 0.000, 0.011, 0.000, 0.056,
        0.081, 0.009, 0.090, 0.248, 0.382, 0.809, -999.000, 1.000, 0.118, 0.000, 0.000, 0.002,
        0.000, 0.000, 0.000, 0.019, 0.302, 0.088, 0.000, 0.058, -999.000, 1.000, 1.000, 0.214,
        0.010, 0.055, 0.014, 0.016, 0.075, 0.050, 0.085, 0.112, 0.016, 0.058, 0.231, 0.513, 0.043,
        0.151, 0.577, 0.522, 0.419, 0.243, 1.000, 0.986, 0.234, 0.310, 0.445, 0.410, 0.498, 0.303,
        0.306, 0.004, 0.068, 0.030, 0.034, 0.279, 0.000, 0.190, 0.179, 0.088, 0.154, 0.265, 0.821,
        0.242, 0.232, 0.397, 0.353, 0.124, 0.521, 0.385, 0.339, 0.651, 0.865, 0.103, 0.023, 0.038,
        0.011, 0.000, 0.130, 0.327, 0.177, 0.068, 1.000, 0.052, 0.000, 0.682, 1.000, 1.000, 1.000,
        0.125, 0.132, 0.099, 0.068, 0.056, 0.077, 0.095, 0.145, 0.067, 0.166, 0.355, 1.000, 0.435,
        0.202, 0.160, 0.457, 0.139, 0.498, 0.246, 0.327, 0.444, 0.190, 0.061, 0.000, 0.172, 0.453,
        0.286, 1.000, 1.000, 0.211, 0.043, 0.023, 0.109, 0.662, 0.505, 0.229, 0.442, 0.050, 0.000,
        0.091, 0.137, 0.156, 0.115, 0.239, 0.102, 0.077, 0.173, 0.060, 0.205, 0.154, 0.179, 0.000,
        0.072, 0.006, 0.186, 0.133, 0.370, 0.737, 0.894, 0.328, 0.442, 0.255, 1.000, 1.000, 1.000,
        1.000, 1.000, 1.000, 0.636, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.498,
        0.155, 0.205, 0.292, 0.280, 0.465, 0.058, 0.272, 0.420, 0.190, 0.097, 0.000, 0.288, 0.303,
        0.851, 0.855, 0.303, 0.176, 0.087, 0.114, 0.104, 0.168, 0.528, 0.745, 0.571, 0.734, 0.302,
        0.252, 0.330, 0.000, 0.011, 0.012, 0.078, 0.000, -999.000, 0.038, 0.092, 0.263, 0.050,
        0.108, 0.286, 0.673, 0.833, 1.000, -999.000, 1.000, 0.050, 0.000, 0.000, 0.000, 0.000,
        0.019, 0.014, 0.000, 0.017, 0.019, 0.071, 0.075, 0.135, 0.000, 0.171, 0.039, 0.069, 0.062,
        0.064, 0.004, 0.006, 0.000, 0.012, 0.041, 0.060, 0.017, 0.958, 1.000, 1.000, 0.364, 0.312,
        0.077, 0.018, 0.047, 0.089, 0.141, 0.471, 0.363, 0.399, 0.174, 0.167, 0.315, 0.105, 0.146,
        0.089, 0.033, 0.043, 0.112, 0.812, 0.516, 0.320, 0.238, 0.105, 0.088, 0.249, 0.976, 0.231,
        1.000, 0.224, 0.444, 0.693, 1.000, 1.000, 0.440, 1.000, 1.000, 0.342, 0.007, 0.034, 0.003,
        0.026, 0.185, 0.356, 0.000, 0.000, 0.076, 0.419, 0.349, 0.110, 0.053, 0.043, 0.128, 0.047,
        0.169, -999.000, 0.090, 0.049, 0.026, 0.220, 0.649, 1.000, 1.000, 1.000, 1.000, 1.000,
        1.000, 1.000, 0.757, 0.147, 0.049, 0.203, 0.091, 0.290, 0.142, 0.520, 1.000, 0.747, 0.818,
        0.290, 0.079, 0.260, 0.343, 0.636, 0.296, 1.000, 1.000, 1.000, 0.089, 0.147, 0.000, 0.000,
        0.040, 0.007, 0.044, 0.077, 0.146, 0.101, 0.160, 0.105, 0.000, 0.075, 0.122, 0.193, 0.065,
        0.090, 0.449, 0.265, 1.000, 0.316, 0.323, 0.626, 0.264, 0.000, 0.000, 0.018, 0.000, 0.035,
        0.048, 0.136, 0.325, 1.000, 1.000, 0.446, 0.829, 0.157, 0.043, 0.040, 0.040, 0.090, 0.309,
        1.000, 0.561, 0.497, 0.752, 1.000, 0.041, 0.067, 0.055, 0.209, 0.294, 0.093, 0.045, 0.208,
        0.240, 0.090, 0.352, 0.461, 0.049, 0.288, -999.000, 0.091, 0.124, 0.104, 0.414, 0.743,
        0.115, 0.028, 0.000, 0.044, 0.003, 0.054, 0.019, 0.089, 0.282, 0.244, 1.000, 1.000, 0.682,
        0.333, 0.039, 0.026, 0.134, 0.138, 0.593, 0.376, 0.971, 0.564, 1.000, 1.000, 0.280, 0.047,
        0.047, 0.115, 0.321, 0.800, 1.000, 1.000, 1.000, 1.000, 1.000, 0.832, 0.661, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.797, 0.972, 0.208, 1.000,
        1.000, 1.000, 0.348, 0.490, 0.356, 0.392, 0.492, 0.342, 1.000, 0.629, 1.000, 0.296, 0.736,
        0.490, 0.235, 0.324, 0.565, 1.000, 1.000, 0.281, 0.814, 0.686, 0.402, 0.215, 0.632, 0.336,
        0.266, 0.396, 0.000, 0.009, 0.000, 0.000, 0.000, 0.022, 0.036, 0.161, 0.261, 0.053, 0.068,
        0.036, 0.194, 0.426, 0.578, 1.000, 0.012, 0.086, 0.020, 0.056, 0.050, 0.000, 0.047, 0.433,
        0.703, 0.756, 0.692, 1.000, 0.754, 0.780, 0.283, 0.688, 0.436, 0.436, 0.391, 0.000, 0.093,
        0.048, 0.585, 0.698, 0.396, 0.331, 1.000, 1.000, 1.000, 0.738, 0.200, 0.601, 0.290, 0.049,
        0.117, 0.268, 0.708, 0.497, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.069, 0.000,
        0.262, 0.130, 0.717, 0.970, 0.870, 0.394, 0.748, 0.536, 0.053, 0.000, 0.064, 0.083, 0.325,
        0.032, 0.163, 1.000, 1.000, 0.330, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.049, 0.013,
        0.019, 0.000, 0.000, 0.035, 0.151, 0.202, 0.316, 0.062, 0.070, 0.632, 0.406, 0.568, 0.306,
        0.493, 0.228, 0.222, 0.364, 1.000, 0.949, 1.000, 1.000, 1.000, 1.000, 1.000, 0.203, 0.000,
        0.000, 0.000, 0.004, 0.120, 0.128, 0.157, 0.331, 0.184, 0.114, 0.135, 0.062, 0.082, 0.030,
        0.048, 0.092, 0.097, 0.057, 0.063, 0.164, 0.000, 0.024, 0.175, 0.500, 1.000, 0.210, 0.087,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.106, 0.131, 0.564, 0.000, 0.125, 0.928, 0.648, 0.565,
        1.000, 0.832, 0.102, 0.235, 0.269, 0.310, 0.341, 0.000, 0.055, 0.000, 0.044, 0.285, 0.622,
        0.026, 0.486, 0.019, 0.000, 0.061, 0.000, 0.887, 0.888, 0.356, 0.452, 0.109, 0.000, 0.787,
        0.127, 0.466, 0.745, 0.849, 0.985, 0.369, 0.836, 0.997, 1.000, 0.061, 0.000, 0.140, 0.488,
        0.286, 0.247, 0.449, 0.601, 0.052, 0.000, 0.000, 0.129, 0.000, 0.000, 0.071, 0.094, 0.070,
        0.000, 0.031, 0.002, 0.030, 0.067, 0.070, 0.104, 0.000, 0.000, 0.000, 0.000, 0.027, 0.022,
        0.038, 0.046, 0.215, 0.064, 0.095, 0.147, 0.091, 0.121, 0.080, 0.147, 0.722, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.904, 1.000, 1.000, 0.368, 0.666, 0.243, 0.165,
        0.196, 0.034, 0.134, 0.403, 0.389, 1.000, 0.059, 0.000, 0.032, 0.072, 0.353, 0.113, 1.000,
        1.000, 1.000, 0.271, 0.110, 0.263, 0.545, 0.139, 0.003, 0.045, 0.000, 0.028, 0.064, 0.188,
        0.158, 0.530, 0.203, 0.416, 0.156, 0.010, 0.000, 0.046, 0.085, 0.115, 0.089, 0.611, 0.652,
        0.143, 0.107, 0.086, 0.164, 0.084, 0.783, 1.000, 1.000, 0.041, 0.061, 0.058, 0.780, 1.000,
        1.000, 1.000, 1.000, 0.083, 0.009, 0.121, 0.111, 0.016, 0.111, 0.090, 0.108, 0.339, 1.000,
        0.279, 0.593, 0.778, 0.684, 0.548, 1.000, 1.000, 1.000, -999.000, 1.000, 1.000, 0.206,
        0.034, 0.108, 0.520, 0.783, 0.414, 0.776, 0.197, 0.162, 0.153, 0.375, 0.547, 0.321, 0.450,
        0.028, 0.032, 0.109, 0.091, 0.275, 0.433, 0.419, 0.302, 0.336, 0.169, 0.438, 0.349, 0.842,
        0.165, 0.286, 0.393, 0.389, 0.757, 0.643, 0.246, 0.470, 0.578, 0.455, 1.000, 1.000, 1.000,
        1.000, 0.737, 0.016, 0.060, 0.060, 0.089, 0.417, 0.429, 0.483, 0.042, 0.041, 0.047, 0.064,
        0.914, 0.965, 1.000, 1.000, 1.000, 0.000, 0.021, 0.139, 0.631, 1.000, 0.748, 0.011, 0.120,
        0.001, 0.010, 0.241, 0.049, 0.151, 0.171, 0.257, 0.092, 0.053, 0.260, 0.491, 1.000, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.767, 0.357, 0.265, 0.578, 1.000,
        1.000, 0.344, 0.025, 0.074, 0.126, 0.180, 0.336, 0.274, 0.154, 0.356, 0.131, 0.752, 1.000,
        0.332, 0.010, 0.076, 0.181, 0.328, 0.617, 0.995, 1.000, 1.000, 1.000, 1.000, 0.822, 0.090,
        0.062, 0.105, 0.096, 0.020, 0.202, 0.333, 0.158, 0.170, 0.838, 1.000, 1.000, 0.406, 0.089,
        0.020, 0.445, 1.000, 1.000, 0.056, 0.073, 0.039, 0.092, 0.237, 0.470, 0.247, 1.000, 1.000,
        1.000, 1.000, 0.253, 0.409, 0.361, 0.550, 1.000, 1.000, 0.000, 0.010, 0.217, 0.843, 1.000,
        1.000, 1.000, 1.000, 0.002, 0.000, 0.014, 0.047, 0.060, 0.000, 0.035, 0.088, 0.048, 0.124,
        0.038, 1.000, 0.165, 0.000, 1.000, 1.000, 0.085, 0.053, 0.000, 0.000, 0.000, 0.004, 0.111,
        0.101, 0.462, 1.000, 0.026, 0.029, 0.028, 0.042, 0.070, 0.236, 0.222, 0.662, 0.246, 0.090,
        0.158, 0.356, 0.506, 0.202, 0.471, 1.000, 1.000, 1.000, 1.000, 0.159, 0.176, 0.178, 0.138,
        0.029, 0.001, 0.100, 0.346, 0.344, 0.297, 0.084, 0.104, 0.237, 0.236, 0.147, 0.110, 0.076,
        0.303, 1.000, 0.288, 1.000, 0.936, 0.029, 0.060, 0.111, 0.401, 1.000, 0.079, 0.072, 0.072,
        0.000, 0.021, 0.225, 0.880, 0.714, 0.471, 1.000, 1.000, 1.000, 0.052, 0.047, 0.119, 0.146,
        0.070, 0.004, 0.000, 0.065, 0.000, 0.000, 0.000, 0.000, 0.008, 0.000, 0.000, 0.036, 0.077,
        0.082, 0.052, 0.048, 0.065, 0.098, 0.006, 0.137, 0.017, 0.000, 0.000, 0.032, 0.000, 0.015,
        0.000, 0.004, 0.033, 0.000, 0.009, 0.694, 1.000, 1.000, 0.165, 0.538, 0.897, 0.162, 0.151,
        0.027, 0.001, 0.268, 0.335, 0.167, 0.084, 0.073, 0.022, 0.000, 0.000, 0.014, 0.006, 0.065,
        0.013, 0.099, 0.125, 1.000, 1.000, 0.051, 0.245, 0.132, 0.123, 1.000, 0.076, 0.000, 0.000,
        0.000, 0.000, 0.001, 0.000, 0.000, 0.272, 0.000, 0.000, 0.163, 0.118, 0.966, 0.186, 0.338,
        0.920, 0.064, 0.042, 0.007, 0.018, 0.076, 0.759, 0.047, 0.475, 1.000, 1.000, 1.000, 1.000,
        0.722, 1.000, 0.223, 0.371, 1.000, 1.000, 1.000, 1.000, 0.129, 0.196, 0.103, 0.471, 0.323,
        0.536, 0.412, 0.104, 0.607, 0.490, 0.369, 0.964, 0.558, 0.588, 0.450, 0.064, 1.000, 1.000,
        0.101, 0.634, 1.000, 0.567, 0.301, 0.284, 0.109, 0.063, 0.073, 0.077, 0.296, 0.568, 0.990,
        1.000, 0.749, 1.000, 1.000, 0.980, 0.431, 0.381, 1.000, 0.817, 1.000, 1.000, 0.621, 1.000,
        1.000, 1.000, 0.465, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.119, 0.158, 0.067, 0.033,
        0.061, 0.132, 0.178, 0.166, 0.000, 0.023, 0.034, 0.012, 0.000, 0.056, 0.048, 0.000, 0.177,
        0.000, 0.000, 0.000, 0.000, 0.003, 0.000, 0.000, 0.052, 0.270, 0.116, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
        -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000, -999.000,
    ];

    #[test]
    #[allow(clippy::too_many_lines)]
    fn seed_score() {
        struct Result {
            db_range: RangeInclusive<usize>,
            query_range: RangeInclusive<usize>,
            score: f32,
        }

        macro_rules! results {
            ($({db:[$db_start:literal, $db_end:literal]; query:[$query_start:literal, $query_end:literal]; score:$score:literal})*) => {
                [
                    $(
                        Result {
                            db_range: $db_start..=$db_end,
                            query_range: $query_start..=$query_end,
                            score: $score,
                        }
                    ),*
                ]
            };
        }

        const RESULTS: [Result; 71] = results! [
            { db:[1017,1036]; query:[1,20]; score:21.728 }
            { db:[925,1009]; query:[16,100]; score:122.498 }
            { db:[1092,1108]; query:[183,199]; score:21.524 }
            { db:[3,22]; query:[161,180]; score:-29.464 }
            { db:[90,109]; query:[5,24]; score:22.56 }
            { db:[1210,1227]; query:[17,34]; score:5.449 }
            { db:[668,683]; query:[38,53]; score:-5.279 }
            { db:[927,943]; query:[26,42]; score:11.112 }
            { db:[379,398]; query:[17,36]; score:25.631 }
            { db:[1189,1205]; query:[36,52]; score:-11.271 }
            { db:[1036,1051]; query:[68,83]; score:17.78 }
            { db:[1129,1148]; query:[161,180]; score:28.91 }
            { db:[1128,1147]; query:[51,70]; score:29.385 }
            { db:[996,1011]; query:[145,160]; score:20.389 }
            { db:[1199,1216]; query:[29,46]; score:-11.405 }
            { db:[145,160]; query:[39,54]; score:27.195 }
            { db:[1371,1418]; query:[0,47]; score:-26.986 }
            { db:[446,462]; query:[7,23]; score:19.575 }
            { db:[2,21]; query:[51,70]; score:-29.207 }
            { db:[1030,1046]; query:[26,42]; score:-8.454 }
            { db:[1078,1093]; query:[138,153]; score:17.638 }
            { db:[1104,1119]; query:[51,66]; score:21.44 }
            { db:[249,264]; query:[35,50]; score:-3.861 }
            { db:[1363,1380]; query:[145,162]; score:21.948 }
            { db:[1128,1147]; query:[161,180]; score:30.835 }
            { db:[296,311]; query:[138,153]; score:15.775 }
            { db:[206,227]; query:[79,100]; score:26.41 }
            { db:[633,649]; query:[21,37]; score:17.936 }
            { db:[1331,1363]; query:[4,36]; score:33.639 }
            { db:[837,853]; query:[161,177]; score:21.58 }
            { db:[496,511]; query:[83,98]; score:21.347 }
            { db:[959,977]; query:[51,69]; score:24.72 }
            { db:[288,305]; query:[42,59]; score:26.875 }
            { db:[1102,1118]; query:[0,16]; score:18.229 }
            { db:[1261,1279]; query:[11,29]; score:21.025 }
            { db:[497,516]; query:[9,28]; score:23.266 }
            { db:[1486,1501]; query:[14,29]; score:-17.229 }
            { db:[1029,1045]; query:[17,33]; score:3.884 }
            { db:[505,521]; query:[7,23]; score:13.314 }
            { db:[1117,1132]; query:[35,50]; score:-2.058 }
            { db:[38,59]; query:[0,21]; score:24.056 }
            { db:[599,616]; query:[39,56]; score:24.158 }
            { db:[1300,1315]; query:[148,163]; score:15.993 }
            { db:[507,523]; query:[84,100]; score:15.311 }
            { db:[960,979]; query:[161,180]; score:32.495 }
            { db:[144,183]; query:[1,40]; score:18.737 }
            { db:[590,605]; query:[14,29]; score:16.333 }
            { db:[182,204]; query:[4,26]; score:15.74 }
            { db:[798,813]; query:[182,197]; score:19.616 }
            { db:[1030,1046]; query:[183,199]; score:23.803 }
            { db:[740,764]; query:[0,24]; score:17.139 }
            { db:[1146,1161]; query:[35,50]; score:-13.542 }
            { db:[1009,1024]; query:[37,52]; score:20.362 }
            { db:[187,202]; query:[85,100]; score:19.617 }
            { db:[380,395]; query:[35,50]; score:11.105 }
            { db:[1339,1357]; query:[21,39]; score:18.825 }
            { db:[757,773]; query:[54,70]; score:19.21 }
            { db:[445,463]; query:[82,100]; score:24.29 }
            { db:[1245,1261]; query:[26,42]; score:20.026 }
            { db:[507,522]; query:[85,100]; score:13.322 }
            { db:[1161,1178]; query:[77,94]; score:15.812 }
            { db:[1211,1227]; query:[183,199]; score:22.302 }
            { db:[1345,1360]; query:[35,50]; score:22.105 }
            { db:[3,19]; query:[162,178]; score:-28.132 }
            { db:[1235,1251]; query:[70,86]; score:18.784 }
            { db:[1302,1318]; query:[13,29]; score:15.506 }
            { db:[167,189]; query:[4,26]; score:22.55 }
            { db:[86,101]; query:[0,15]; score:20.185 }
            { db:[347,378]; query:[11,42]; score:34.237 }
            { db:[1467,1482]; query:[71,86]; score:12.362 }
            { db:[581,597]; query:[183,199]; score:20.696 }
        ];

        let cli = dummy_cli();
        let query = QUERY2.map(ReactivityWithPlaceholder::from);
        let db = DB2.map(ReactivityWithPlaceholder::from);

        for result in RESULTS {
            let query = &query[result.query_range];
            let db = &db[result.db_range];

            assert_abs_diff_eq!(
                calc_seed_alignment_score_from_reactivity(query, db, &cli),
                result.score,
                epsilon = 1e-5,
            );
        }
    }

    #[test]
    fn serialize_gapped_reactivity() {
        use std::f64::NAN;
        use BaseOrGap::*;

        fn check(reactivity: &[f32], alignment: AlignedSequenceRef<'_>, expected: &[f64]) {
            let values = serde_json::to_value(GappedReactivity {
                reactivity,
                alignment,
            })
            .unwrap();
            let values: Vec<Option<f64>> = serde_json::from_value(values).unwrap();

            assert_eq!(values.len(), expected.len());
            assert!(values.iter().zip(expected).all(|(a, b)| match a {
                Some(a) => (a - b).abs() < 0.00001,
                None => b.is_nan(),
            }));
        }

        check(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            AlignedSequenceRef(&[
                Gap, Gap, Base, Base, Gap, Base, Gap, Gap, Base, Base, Base, Gap,
            ]),
            &[
                NAN, NAN, 0.1, 0.2, NAN, 0.3, NAN, NAN, 0.4, 0.5, 0.6, NAN, 0.7,
            ],
        );

        check(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            AlignedSequenceRef(&[Base, Base, Gap, Base]),
            &[0.1, 0.2, NAN, 0.3, 0.4, 0.5, 0.6, 0.7],
        );

        check(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            AlignedSequenceRef(&[Base; 40]),
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        );

        check(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            AlignedSequenceRef(&[Base, Base, Base, Base, Base, Base, Base, Gap, Gap, Gap]),
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        );

        check(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            AlignedSequenceRef(&[
                Gap, Gap, Gap, Base, Base, Base, Base, Base, Base, Base, Gap, Gap, Gap,
            ]),
            &[NAN, NAN, NAN, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        );
    }

    #[test]
    fn base_from_u8() {
        assert_eq!(Base::try_from(b'A'), Ok(Base::A));
        assert_eq!(Base::try_from(b'C'), Ok(Base::C));
        assert_eq!(Base::try_from(b'G'), Ok(Base::G));
        assert_eq!(Base::try_from(b'T'), Ok(Base::T));
        assert_eq!(Base::try_from(b'U'), Ok(Base::T));
    }

    #[test]
    fn display_sequence() {
        let bases = [Base::C, Base::T, Base::G, Base::A];
        assert_eq!(
            Sequence {
                bases: &bases,
                molecule: Molecule::Dna,
            }
            .to_string(),
            "CTGA",
        );

        assert_eq!(
            Sequence {
                bases: &bases,
                molecule: Molecule::Rna,
            }
            .to_string(),
            "CUGA",
        );
    }

    #[test]
    #[should_panic]
    fn cannot_display_sequence_for_unknown_molecule() {
        let bases = [Base::C, Base::T, Base::G, Base::A];
        Sequence {
            bases: &bases,
            molecule: Molecule::Unknown,
        }
        .to_string();
    }
}
