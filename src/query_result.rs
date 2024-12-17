use std::{
    fmt::{self, Display},
    ops,
    sync::Arc,
};

use num_traits::{Float, FromPrimitive};
use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};
use tabled::Tabled;

use crate::{
    aligner::{AlignedSequence, AlignmentResult},
    dotbracket::DotBracketOwnedSorted,
};

#[derive(Debug, Deserialize, Serialize, Tabled)]
pub struct QueryResult {
    #[serde(rename = "Query")]
    pub query: Arc<str>,

    #[serde(rename = "DB entry")]
    pub db_entry: String,

    #[serde(rename = "Qstart")]
    pub query_start: usize,

    #[serde(rename = "Qend")]
    pub query_end: usize,

    #[serde(rename = "Dstart")]
    pub db_start: usize,

    #[serde(rename = "Dend")]
    pub db_end: usize,

    #[serde(rename = "Qseed")]
    pub query_seed: Range,

    #[serde(rename = "Dseed")]
    pub db_seed: Range,

    #[serde(rename = "Score")]
    pub score: f32,

    #[serde(rename = "P-value")]
    #[tabled(display_with = "display_scientific")]
    pub pvalue: f64,

    #[serde(rename = "E-value")]
    #[tabled(display_with = "display_scientific")]
    pub evalue: f64,

    #[serde(rename = "TargetBpSupport")]
    #[tabled(display_with = "display_scientific_opt")]
    pub target_bp_support: Option<f32>,

    #[serde(rename = "QueryBpSupport")]
    #[tabled(display_with = "display_scientific_opt")]
    pub query_bp_support: Option<f32>,

    #[serde(rename = "MfePvalue")]
    #[tabled(display_with = "display_scientific_opt")]
    pub mfe_pvalue: Option<f64>,

    #[serde(rename = "")]
    pub status: Status,

    #[serde(skip)]
    #[tabled(skip)]
    pub alignment: Arc<AlignmentResult<AlignedSequence>>,

    #[serde(skip)]
    #[tabled(skip)]
    pub dotbracket: Option<DotBracketOwnedSorted>,
}

impl QueryResult {
    pub fn new(query: impl Into<Arc<str>>) -> Self {
        let query = query.into();
        Self {
            query,
            db_entry: String::default(),
            query_start: Default::default(),
            query_end: Default::default(),
            db_start: Default::default(),
            db_end: Default::default(),
            query_seed: Range::default(),
            db_seed: Range::default(),
            score: Default::default(),
            pvalue: Default::default(),
            evalue: Default::default(),
            status: Status::default(),
            target_bp_support: Option::default(),
            query_bp_support: Option::default(),
            mfe_pvalue: Option::default(),
            alignment: Arc::default(),
            dotbracket: Option::default(),
        }
    }
}

#[derive(Debug)]
pub struct Serializeable<'a> {
    pub query_result: &'a QueryResult,
    pub eval_align_fold: bool,
}

impl serde::Serialize for Serializeable<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let fields_count = if self.eval_align_fold { 15 } else { 12 };
        let mut struc = serializer.serialize_struct("QueryResult", fields_count)?;

        let query_result = self.query_result;
        struc.serialize_field("Query", &query_result.query)?;
        struc.serialize_field("DB field", &query_result.db_entry)?;
        struc.serialize_field("Qstart", &query_result.query_start)?;
        struc.serialize_field("Qend", &query_result.query_end)?;
        struc.serialize_field("Dstart", &query_result.db_start)?;
        struc.serialize_field("Dend", &query_result.db_end)?;
        struc.serialize_field("Qseed", &query_result.query_seed)?;
        struc.serialize_field("Dseed", &query_result.db_seed)?;
        struc.serialize_field("Score", &query_result.score)?;
        struc.serialize_field("P-value", &display_scientific(&query_result.pvalue))?;
        struc.serialize_field("E-value", &display_scientific(&query_result.evalue))?;

        if self.eval_align_fold {
            struc.serialize_field(
                "TargetBpSupport",
                &display_scientific_opt(&query_result.target_bp_support),
            )?;
            struc.serialize_field(
                "QueryBpSupport",
                &display_scientific_opt(&query_result.query_bp_support),
            )?;
            struc.serialize_field(
                "MfePvalue",
                &display_scientific_opt(&query_result.mfe_pvalue),
            )?;
        }

        struc.serialize_field("", &query_result.status)?;
        struc.end()
    }
}

#[derive(Debug)]
pub struct Range(pub ops::RangeInclusive<usize>);

impl Default for Range {
    fn default() -> Self {
        Self(0..=0)
    }
}

impl Serialize for Range {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Range {
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

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.0.start(), self.0.end())
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    #[serde(rename = "!")]
    PassInclusionEvalue,

    #[serde(rename = "?")]
    PassReportEvalue,

    #[default]
    #[serde(rename = "")]
    NotPass,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::PassInclusionEvalue => f.write_str("!"),
            Self::PassReportEvalue => f.write_str("?"),
            Self::NotPass => f.write_str(""),
        }
    }
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

#[allow(clippy::ref_option)] // The signature is because of serde
fn display_scientific_opt<T>(x: &Option<T>) -> String
where
    T: Float + FromPrimitive + Display + fmt::LowerExp,
{
    x.as_ref().map(display_scientific).unwrap_or_default()
}
