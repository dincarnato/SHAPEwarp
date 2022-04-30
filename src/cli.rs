use clap::{ArgEnum, ArgGroup, Args, Parser};
use std::{fmt, ops::Range, path::PathBuf, str::FromStr};

use crate::Reactivity;

#[derive(Debug, Parser)]
#[clap(
    author,
    version,
    about,
    group(
        ArgGroup::new("fold-opt-group").args(&["fold-query", "eval-align-fold"])
    )
)]
/// SHAPE-guided RNA structural homology search
pub struct Cli {
    /// Path to a database folder generated with swBuildDb
    #[clap(long, visible_alias = "db")]
    pub database: PathBuf,

    /// Path to the query file
    ///
    /// Note: each entry should contain (one per row) the sequence id, the nucleotide sequence and
    /// a comma-separated list of SHAPE reactivities
    #[clap(short, long)]
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
    pub inclusion_evalue: f32,

    /// E-value threshold to report a match
    #[clap(long, default_value_t = 0.1, aliases = &["reportEvalue", "repE"], visible_alias = "rep-e")]
    pub report_evalue: f32,

    /// Besides printing the result summary to screen, matches are reported as a TSV file
    #[clap(long, alias = "tblOut")]
    pub output_table: bool,

    /// Generates plots of aligned SHAPE reactivity (or probability) profiles
    ///
    /// Note: plots are generated only for matches below the inclusion E-value cutoff
    #[clap(long, alias = "makePlot")]
    pub make_plot: bool,

    /// Reports sequence alignments in the specified format ([f]asta or [s]tockholm)
    /// Note: alignments are reported only for matches below the inclusion E-value cutoff
    #[clap(long, alias = "reportAln", arg_enum)]
    pub report_alignment: Option<ReportAlignment>,

    /// Query SHAPE profile is first used to calculate a base-pairing probability profile, that is
    /// then used to search into the db
    ///
    /// Note: the db must have been generated with the --foldDb option
    #[clap(long, alias = "foldQuery")]
    pub fold_query: bool,

    #[clap(
        flatten,
        next_help_heading = "Folding options (require --fold-query or --eval-align-fold)"
    )]
    pub folding_args: FoldingArgs,

    #[clap(flatten, next_help_heading = "Kmer lookup options")]
    pub kmer_lookup_args: KmerLookupArgs,

    #[clap(flatten, next_help_heading = "Alignment options")]
    pub alignment_args: AlignmentArgs,

    #[clap(
        flatten,
        next_help_heading = r#"Alignment folding evaluation options (see also "Folding options")"#
    )]
    pub alignment_folding_eval_args: AlignmentFoldingEvaluationArgs,
}

#[derive(Debug, Args)]
pub struct FoldingArgs {
    /// Slope for SHAPE reactivities conversion into pseudo-free energy contributions
    #[clap(long, default_value_t = 1.8, requires = "fold-opt-group")]
    pub slope: Reactivity,

    /// Intercept for SHAPE reactivities conversion into pseudo-free energy contributions
    #[clap(long, default_value_t = -0.6, requires = "fold-opt-group")]
    pub intercept: Reactivity,

    /// Maximum allowed base-pairing distance
    #[clap(
        long,
        default_value_t = 600,
        alias = "maxBPspan",
        requires = "fold-opt-group"
    )]
    pub max_bp_span: u32,

    /// Disallows lonely pairs (helices of 1 bp)
    #[clap(long, alias = "noLonelyPairs", requires = "fold-opt-group")]
    pub no_lonely_pairs: bool,

    /// Disallows G:U wobbles at the end of helices
    #[clap(long, alias = "noClosingGU", requires = "fold-opt-group")]
    pub no_closing_gu: bool,

    /// Folding temperature
    #[clap(long, default_value_t = 37., requires = "fold-opt-group")]
    pub temperature: f32,

    #[clap(
        flatten,
        next_help_heading = "Query folding-specific options (require --fold-query)"
    )]
    pub query_folding_args: QueryFoldingArgs,
}

#[derive(Debug, Args)]
pub struct QueryFoldingArgs {
    /// Size (in nt) of the sliding window for partition function calculation
    #[clap(
        long,
        default_value_t = 800,
        alias = "winSize",
        requires = "fold-query"
    )]
    pub win_size: u32,

    /// Offset (in nt) for partition function window sliding
    #[clap(long, default_value_t = 200, requires = "fold-query")]
    pub offset: u32,

    /// Number of bases to trim from both ends of partition function windows to avoid terminal
    /// biases
    #[clap(long, default_value_t = 50, alias = "winTrim", requires = "fold-query")]
    pub win_trim: u32,

    /// SHAPE reactivity is ignored when folding the query
    #[clap(long = "ignore-react", alias = "ignoreReact", requires = "fold-query")]
    pub ignore_reactivity: bool,
}

#[derive(Debug, Args)]
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
    #[clap(long, requires = "match-kmer-gc-content", alias = "kmerMaxGCdiff")]
    kmer_max_gc_diff: Option<f32>,

    /// The sequence of a query kmer and the corresponding database match must differ no more than
    /// --kmer-max-seq-dist
    #[clap(long, alias = "matchKmerSeq")]
    pub match_kmer_seq: bool,

    /// Maximum allowed sequence distance to retain a kmer match
    ///
    /// Note: when >= 1, this is interpreted as the absolute number of bases that are allowed to
    /// differ between the kmer and the matching region. When < 1, this is interpreted as a
    /// fraction of the kmer's length
    #[clap(
        long,
        default_value_t = 0,
        requires = "match-kmer-seq",
        alias = "kmerMaxSeqDist"
    )]
    pub kmer_max_seq_dist: u32,

    /// Minimum complexity (measured as Gini coefficient) of candidate kmers
    #[clap(long, default_value_t = 0.3, alias = "kmerMinComplexity")]
    pub kmer_min_complexity: f32,

    /// A kmer is allowed to match a database entry on average every this many nt
    #[clap(long, default_value_t = 200, alias = "kmerMaxMatchEveryNt")]
    pub kmer_max_match_every_nt: u32,
}

#[derive(Debug, Args)]
pub struct AlignmentArgs {
    /// Minimum and maximum score reactivity differences below 0.5 will be mapped to
    #[clap(long, default_value_t = MinMax (0.0..2.), alias = "alignMatchScore")]
    pub align_match_score: MinMax<Reactivity>,

    /// Minimum and maximum score reactivity differences above 0.5 will be mapped to
    #[clap(long, default_value_t = MinMax (-6.0..-0.5), alias = "alignMismatchScore")]
    pub align_mismatch_score: MinMax<Reactivity>,

    /// Gap open penalty
    #[clap(long, default_value_t = -14., alias = "alignGapOpenPenal")]
    pub align_gap_open_penalty: f32,

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
        requires = "align-score-seq",
        alias = "alignSeqMatchScore"
    )]
    pub align_seq_match_score: f32,

    /// Score penalty for mismatching bases
    #[clap(
        long,
        default_value_t = -2.,
        requires = "align-score-seq",
        alias = "alignSeqMismatchScore"
    )]
    pub align_seq_mismatch_score: f32,
}

#[derive(Debug, Args)]
pub struct AlignmentFoldingEvaluationArgs {
    /// Alignments passing the --inclusion-evalue threshold, are further evaluated for the presence
    /// or a conserved RNA structure by using RNAalifold
    #[clap(long, alias = "evalAlignFold")]
    pub eval_align_fold: bool,

    /// Number of shufflings to perform for each alignment
    #[clap(long, default_value_t = 100)]
    pub shufflings: u16,

    /// Size (in nt) of the blocks for shuffling the alignment
    #[clap(long, alias = "blockSize", default_value_t = 3)]
    pub block_size: u16,

    /// Besides shuffling blocks, residues within each block will be shuffled as well
    #[clap(long, alias = "inBlockShuffle")]
    pub in_block_shuffle: bool,

    /// Minimum fraction of base-pairs of the RNAalifold-inferred structure that should be
    /// supported by both query and db sequence to retain a match
    #[clap(long, default_value_t = 0.75, alias = "minBpSupport")]
    pub min_bp_support: f32,

    /// Use RIBOSUM scoring matrix
    #[clap(long, alias = "ribosumScoring")]
    pub ribosum_scoring: bool,

    /// P-value threshold to consider signficant an RNA structure predicted by RNAalifold
    #[clap(long, default_value_t = 0.05, alias = "alignFoldPvalue")]
    pub align_fold_pvalue: f32,

    /// Path to RNAalifold executable (Default: assumes RNAalifold is in PATH)
    #[clap(long, alias = "RNAalifold")]
    pub rna_alifold: Option<PathBuf>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ArgEnum)]
pub enum ReportAlignment {
    #[clap(alias = "f")]
    Fasta,

    #[clap(alias = "s")]
    Stockholm,
}
