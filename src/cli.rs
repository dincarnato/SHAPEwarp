// We are defining CLI structs
#![allow(clippy::struct_excessive_bools)]

use clap::{Args, Parser, ValueEnum};
use serde::Serialize;
use std::{fmt, ops::Range, path::PathBuf, str::FromStr};

use crate::{Distance, Reactivity};

#[derive(Debug, Parser, Serialize)]
#[clap(author, version, about, allow_negative_numbers = true)]
#[serde(rename_all = "kebab-case")]
/// SHAPE-guided RNA structural homology search
pub struct Cli {
    /// Path to a database folder generated with swBuildDb
    #[clap(long, visible_alias = "db")]
    #[serde(skip)]
    pub database: PathBuf,

    /// Path of the shuffled database.
    ///
    /// Use a file containing the shuffled db instead of the one generated on the fly. You can dump
    /// a shuffled db using `--dump-shuffled-db` in order to reuse it later with this parameter.
    #[clap(
        long,
        conflicts_with_all = &[
            "dump_shuffled_db",
            "db_shufflings",
            "db_block_size",
            "db_in_block_shuffle",
        ],
    )]
    #[serde(skip)]
    pub shuffled_db: Option<PathBuf>,

    /// Dump the shuffled DB to the specified path.
    ///
    /// You can load the file for further analyses using the `--shuffled-db` parameter.
    #[clap(long)]
    #[serde(skip)]
    pub dump_shuffled_db: Option<PathBuf>,

    /// Path to the query file
    ///
    /// Note: each entry should contain (one per row) the sequence id, the nucleotide sequence and
    /// a comma-separated list of SHAPE reactivities
    #[clap(short, long)]
    #[serde(skip)]
    pub query: PathBuf,

    /// Output directory
    #[clap(short, long, default_value = "sw_out/")]
    pub output: PathBuf,

    /// Overwrites the output directory (if the specified path already exists)
    #[clap(long, visible_alias = "ow")]
    pub overwrite: bool,

    /// Number of processors to use
    ///
    /// Uses all available processors if not specified
    #[clap(long)]
    pub threads: Option<u16>,

    /// Number of shufflings to perform for each sequence in db
    #[clap(long, alias = "dbShufflings", default_value_t = 100)]
    pub db_shufflings: u16,

    /// Size (in nt) of the blocks for shuffling the sequences in db
    #[clap(long, alias = "dbBlockSize", default_value_t = 10)]
    pub db_block_size: u16,

    /// Besides shuffling blocks, residues within each block in db will be shuffled as well
    #[clap(long, alias = "dbInBlockShuffle")]
    pub db_in_block_shuffle: bool,

    /// Maximum value to which reactivities will be capped
    #[clap(long, default_value_t = 1., alias = "maxReactivity")]
    pub max_reactivity: Reactivity,

    /// If two significant alignments overlap by more than this value, the least significant one
    /// (the one with the lowest alignment score) will be discarded
    #[clap(long, default_value_t = 0.5, alias = "maxAlignOverlap")]
    pub max_align_overlap: f32,

    /// Number of HSGs in the shuffled database to be extended to build the null model
    #[clap(long, default_value_t = 10_000, alias = "nullHSGs")]
    pub null_hsgs: u32,

    /// E-value threshold to consider an alignment significant
    #[clap(long, default_value_t = 0.01, aliases = &["inclusionEvalue", "incE"], visible_alias = "inc-e")]
    pub inclusion_evalue: f64,

    /// E-value threshold to report a match
    #[clap(long, default_value_t = 0.1, aliases = &["reportEvalue", "repE"], visible_alias = "rep-e")]
    pub report_evalue: f64,

    /// Reports sequence alignments in the specified format ([f]asta or [s]tockholm)
    ///
    /// Note: alignments are reported only for matches below the inclusion E-value cutoff
    #[clap(long, alias = "reportAln", value_enum)]
    pub report_alignment: Option<ReportAlignment>,

    /// Reports reactivity for sequence alignments in the "reactivities" folder inside the output
    /// directory, using JSON format
    #[clap(long)]
    pub report_reactivity: bool,

    #[clap(flatten, next_help_heading = "Kmer lookup options")]
    #[serde(flatten)]
    pub kmer_lookup_args: KmerLookupArgs,

    #[clap(flatten, next_help_heading = "Alignment options")]
    #[serde(flatten)]
    pub alignment_args: AlignmentArgs,

    #[clap(
        flatten,
        next_help_heading = r#"Alignment folding evaluation options (see also "Folding options")"#
    )]
    #[serde(flatten)]
    pub alignment_folding_eval_args: AlignmentFoldingEvaluationArgs,
}

#[derive(Debug, Args, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct KmerLookupArgs {
    /// Minimum number of kmers required to form a High Scoring Group (HSG)
    #[clap(long, default_value_t = 2, alias = "minKmers")]
    pub min_kmers: u16,

    /// Maximum distance between two kmers to be merged in a HSG
    #[clap(long, default_value_t = 30, alias = "maxKmerDist")]
    pub max_kmer_dist: u16,

    /// Length (in nt) of the kmers
    #[clap(long, default_value_t = 15, alias = "kmerLen")]
    pub kmer_len: u16,

    /// Sliding offset for extracting candidate kmers from the query
    #[clap(long, default_value_t = 1, alias = "kmerOffset")]
    pub kmer_offset: u16,

    /// The sequence of a query kmer and the corresponding database match must have GC% contents
    /// differing no more than --kmer-max-gc-diff
    #[clap(long, alias = "matchKmerGCcontent")]
    pub match_kmer_gc_content: bool,

    /// Maximum allowed GC% difference to retain a kmer match
    ///
    /// Note: the default value is automatically determined based on the chosen kmer length
    #[clap(long, requires = "match_kmer_gc_content", alias = "kmerMaxGCdiff")]
    pub kmer_max_gc_diff: Option<f32>,

    /// The sequence of a query kmer and the corresponding database match must differ no more than
    /// --kmer-max-seq-dist
    #[clap(long, alias = "matchKmerSeq")]
    pub match_kmer_seq: bool,

    /// Maximum allowed sequence distance to retain a kmer match
    ///
    /// Note: when >= 1, this is interpreted as the absolute number of bases that are allowed to
    /// differ between the kmer and the matching region. When < 1, this is interpreted as a
    /// fraction of the kmer's length
    #[clap(long, requires = "match_kmer_seq", alias = "kmerMaxSeqDist")]
    pub kmer_max_seq_dist: Option<Distance<u32>>,

    /// Minimum complexity (measured as Gini coefficient) of candidate kmers
    #[clap(long, default_value_t = 0.3, alias = "kmerMinComplexity")]
    pub kmer_min_complexity: f32,

    /// A kmer is allowed to match a database entry on average every this many nt
    #[clap(long, default_value_t = 200, alias = "kmerMaxMatchEveryNt")]
    pub kmer_max_match_every_nt: u32,
}

#[derive(Debug, Args, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct AlignmentArgs {
    /// Minimum and maximum score reactivity differences below 0.5 will be mapped to
    #[clap(long, default_value_t = MinMax (-0.5..2.), alias = "alignMatchScore", allow_hyphen_values = true)]
    pub align_match_score: MinMax<Reactivity>,

    /// Minimum and maximum score reactivity differences above 0.5 will be mapped to
    #[clap(long, default_value_t = MinMax (-6.0..-0.5), alias = "alignMismatchScore", allow_hyphen_values = true)]
    pub align_mismatch_score: MinMax<Reactivity>,

    /// Gap open penalty
    #[clap(long, default_value_t = -14., alias = "alignGapOpenPenal")]
    pub align_gap_open_penalty: f32,

    /// Gap extension penalty
    #[clap(long, default_value_t = -5., alias = "alignGapExtPenal")]
    pub align_gap_ext_penalty: f32,

    /// An alignment is allowed to drop by maximum this fraction of the best score encountered so
    /// far, before extension is interrupted
    #[clap(long, default_value_t = 0.8, alias = "alignMaxDropOffRate")]
    pub align_max_drop_off_rate: f32,

    /// An alignment is allowed to drop below the best score encountered so far *
    /// --align-max-drop-off-rate by this number of bases, before extension is interrupted
    #[clap(long, default_value_t = 8, alias = "alignMaxDropOffBases")]
    pub align_max_drop_off_bases: u16,

    /// The maximum allowed tollerated length difference between the query and db sequences to look
    /// for the ideal alignment along the diagonal (measured as a fraction of the length of the
    /// shortest sequence between the db and the query)
    #[clap(long, default_value_t = 0.1, alias = "alignLenTolerance")]
    pub align_len_tolerance: f32,

    /// Sequence matches are rewarded during the alignment
    #[clap(long, alias = "alignScoreSeq")]
    pub align_score_seq: bool,

    /// Score reward for matching bases
    #[clap(
        long,
        default_value_t = 0.5,
        requires = "align_score_seq",
        alias = "alignSeqMatchScore"
    )]
    pub align_seq_match_score: f32,

    /// Score penalty for mismatching bases
    #[clap(
        long,
        default_value_t = -2.,
        requires = "align_score_seq",
        alias = "alignSeqMismatchScore"
    )]
    pub align_seq_mismatch_score: f32,
}

#[derive(Debug, Args, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct AlignmentFoldingEvaluationArgs {
    /// Alignments passing the --inclusion-evalue threshold, are further evaluated for the presence
    /// or a conserved RNA structure by using RNAalifold
    #[clap(long, alias = "evalAlignFold")]
    pub eval_align_fold: bool,

    /// Number of shufflings to perform for each alignment during folding evaluation
    #[clap(long, default_value_t = 100)]
    pub shufflings: u16,

    /// Size (in nt) of the blocks for shuffling the alignment during folding evaluation
    #[clap(long, alias = "blockSize", default_value_t = 3)]
    pub block_size: u16,

    /// Besides shuffling blocks, residues within each block will be shuffled as well during
    /// folding evaluation
    #[clap(long, alias = "inBlockShuffle")]
    pub in_block_shuffle: bool,

    /// Minimum fraction of base-pairs of the RNAalifold-inferred structure that should be
    /// supported by both query and db sequence to retain a match
    #[clap(long, default_value_t = 0.75, alias = "minBpSupport")]
    pub min_bp_support: f32,

    /// Use RIBOSUM scoring matrix
    #[clap(long, alias = "ribosumScoring")]
    pub ribosum_scoring: bool,

    /// Slope for SHAPE reactivities conversion into pseudo-free energy contributions
    #[clap(long, default_value_t = 1.8, requires = "eval_align_fold")]
    pub slope: Reactivity,

    /// Intercept for SHAPE reactivities conversion into pseudo-free energy contributions
    #[clap(long, default_value_t = -0.6, requires = "eval_align_fold")]
    pub intercept: Reactivity,

    /// Maximum allowed base-pairing distance
    #[clap(
        long,
        default_value_t = 600,
        alias = "maxBPspan",
        requires = "eval_align_fold"
    )]
    pub max_bp_span: u32,

    /// Disallows lonely pairs (helices of 1 bp)
    #[clap(long, alias = "noLonelyPairs", requires = "eval_align_fold")]
    pub no_lonely_pairs: bool,

    /// Disallows G:U wobbles at the end of helices
    #[clap(long, alias = "noClosingGU", requires = "eval_align_fold")]
    pub no_closing_gu: bool,

    /// Folding temperature
    #[clap(long, default_value_t = 37., requires = "eval_align_fold")]
    pub temperature: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MinMax<T>(pub Range<T>);

impl<T> fmt::Display for MinMax<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{}", self.0.start, self.0.end)
    }
}

impl<T> Serialize for MinMax<T>
where
    T: fmt::Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseMinMaxError<T> {
    InvalidFormat,
    InnerError { index: u8, error: T },
}

impl<T> FromStr for MinMax<T>
where
    T: FromStr,
{
    type Err = ParseMinMaxError<T::Err>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (start, end) = s.split_once(',').ok_or(ParseMinMaxError::InvalidFormat)?;

        let start = start
            .parse()
            .map_err(|error| ParseMinMaxError::InnerError { index: 0, error })?;

        let end = end
            .parse()
            .map_err(|error| ParseMinMaxError::InnerError { index: 1, error })?;

        Ok(Self(start..end))
    }
}

impl<T> fmt::Display for ParseMinMaxError<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat => {
                f.write_str("invalid min-max format, two comma-separated values expected")
            }
            Self::InnerError { index, error } => {
                let part = match index {
                    0 => "min",
                    1 => "max",
                    _ => unreachable!(),
                };
                write!(f, "{part} part of min-max format is invalid: {error}")
            }
        }
    }
}

impl<T> std::error::Error for ParseMinMaxError<T> where T: std::error::Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
pub enum ReportAlignment {
    #[clap(alias = "f")]
    Fasta,

    #[clap(alias = "s")]
    Stockholm,
}

impl Cli {
    #[cfg(test)]
    pub(crate) fn dummy() -> Self {
        Self::parse_from(["test", "--database", "test", "--query", "test"])
    }
}
