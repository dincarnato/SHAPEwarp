use std::{
    fmt::{self, Display},
    fs::File,
    io::{self, BufWriter},
    path::Path,
    rc::Rc,
};

use anyhow::Context;

use crate::{
    db_file, gapped_sequence::GappedSequence, query_file, QueryResult, ResultFileFormat,
    SequenceEntry,
};

pub(crate) fn write_result(
    result: &QueryResult,
    db_entries: &[db_file::Entry],
    query_entries: &[query_file::Entry],
    alignments_path: &Path,
) -> Result<(), anyhow::Error> {
    let stockholm_path = alignments_path.join(format!("{}.sto", ResultFileFormat::from(result)));
    let file = File::create(stockholm_path).context("Unable to create stockholm file")?;
    let writer = BufWriter::new(file);

    write_result_to_writer(result, db_entries, query_entries, writer)
}

#[inline]
fn write_result_to_writer<W: io::Write>(
    result: &QueryResult,
    db_entries: &[db_file::Entry],
    query_entries: &[query_file::Entry],
    writer: W,
) -> Result<(), anyhow::Error> {
    let &QueryResult {
        ref query,
        db_entry: ref db,
        query_start,
        query_end,
        db_start,
        db_end,
        ref alignment,
        ref dotbracket,
        ..
    } = result;

    let db_entry = db_entries
        .iter()
        .find(|entry| entry.name() == db)
        .expect("db entry should be available");
    let query_entry = query_entries
        .iter()
        .find(|entry| entry.name() == &**query)
        .expect("query entry should be available");

    let db_sequence = GappedSequence {
        sequence: crate::Sequence {
            bases: &db_entry.sequence()[db_start..=db_end],
            molecule: db_entry.molecule(),
        },
        alignment: alignment.target.to_ref(),
    };

    let query_sequence = GappedSequence {
        sequence: crate::Sequence {
            bases: &query_entry.sequence()[query_start..=query_end],
            molecule: query_entry.molecule(),
        },
        alignment: alignment.query.to_ref(),
    };

    let seq_label_align = db.len().max(query.len()).max("#=GC SS_cons".len()) + 1;

    let mut stockholm = Stockholm::default()
        .with_identification(ResultFileFormat::from(result))
        .with_author(format!("SHAPEwarp {}", env!("CARGO_PKG_VERSION")))
        .with_empty_line()
        .with_sequence(format!("{db:seq_label_align$}"), db_sequence)
        .with_sequence(format!("{query:seq_label_align$}"), query_sequence);

    if let Some(dotbracket) = dotbracket {
        stockholm = stockholm.with_column_annotation(
            format!("{:1$}", "SS_cons", seq_label_align - "#=GC ".len()),
            dotbracket,
        );
    }

    stockholm.write(writer)?;

    Ok::<_, anyhow::Error>(())
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Stockholm(Vec<Entry>);

impl Stockholm {
    pub fn write<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        writeln!(writer, "# STOCKHOLM 1.0")?;
        self.0
            .iter()
            .try_for_each(|entry| entry.write(&mut writer))?;
        writeln!(writer, "//")?;
        Ok(())
    }

    pub fn with_identification(mut self, id: impl Display) -> Self {
        self.0
            .push(Entry::FeatureAnnotation(FeatureAnnotation::Identification(
                id.to_string(),
            )));
        self
    }

    pub fn with_author(mut self, author: impl Display) -> Self {
        self.0
            .push(Entry::FeatureAnnotation(FeatureAnnotation::Author(
                author.to_string(),
            )));
        self
    }

    pub fn with_sequence(mut self, name: impl Into<Rc<str>>, aligned: impl Display) -> Self {
        let name = name.into();
        let aligned = aligned.to_string();

        self.0.push(Entry::Sequence(Sequence { name, aligned }));
        self
    }

    pub fn with_column_annotation(
        mut self,
        feature: impl Display,
        annotation: impl Display,
    ) -> Self {
        let feature = feature.to_string();
        let annotation = annotation.to_string();

        self.0.push(Entry::ColumnAnnotation {
            feature,
            annotation,
        });
        self
    }

    pub fn with_empty_line(mut self) -> Self {
        self.0.push(Entry::Empty);
        self
    }
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    Sequence(Sequence),
    FeatureAnnotation(FeatureAnnotation),
    ColumnAnnotation {
        feature: String,
        annotation: String,
    },
    SequenceAnnotation {
        sequence: Rc<str>,
        feature: String,
        annotation: String,
    },
    ResidueAnnotation {
        sequence: Rc<str>,
        feature: String,
        annotation: String,
    },
    Empty,
}

impl Entry {
    pub fn write<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        match self {
            Entry::Sequence(sequence) => {
                writeln!(writer, "{} {}", sequence.name, sequence.aligned)
            }
            Entry::FeatureAnnotation(ann) => ann.write(writer),
            Entry::ColumnAnnotation {
                feature,
                annotation,
            } => writeln!(writer, "#=GC {feature} {annotation}"),
            Entry::SequenceAnnotation {
                sequence,
                feature,
                annotation,
            } => writeln!(writer, "#=GS {sequence} {feature} {annotation}"),
            Entry::ResidueAnnotation {
                sequence,
                feature,
                annotation,
            } => writeln!(writer, "#=GR {sequence} {feature} {annotation}"),
            Entry::Empty => writeln!(writer),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Sequence {
    pub name: Rc<str>,
    pub aligned: String,
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureAnnotation {
    /// Accession number in form `PFxxxxx` (Pfam) or `RFxxxxx` (Rfam).
    AccessionNumber(String),

    /// One word name for family.
    Identification(String),

    /// Short description of family.
    Definition(String),

    /// Authors of the entry.
    Author(String),

    /// The source suggesting the seed members belong to one family.
    SourceOfSeed(String),

    /// The source (prediction or publication) of the consensus RNA secondary structure used by Rfam.
    SourceOfStructure(String),

    /// Command line used to generate the model
    BuildMethod(String),

    /// Command line used to perform the search
    SearchMethod(String),

    /// Search threshold to build the full alignment.
    GatheringThreshold(f32),

    /// Lowest sequence score (and domain score for Pfam) of match in the full alignment.
    TrustedCutoff(f32),

    /// Highest sequence score (and domain score for Pfam) of match not in full alignment.
    NoiseCutoff(f32),

    /// Type of family.
    Type(FamilyType),

    /// Number of sequences in alignment.
    Sequence(u8),

    /// Comment about database reference.
    DatabaseComment(String),

    /// Reference to external database.
    DatabaseReference(String),

    /// Comment about literature reference.
    ReferenceComment(String),

    /// Reference Number.
    ReferenceNumber(String),

    /// Eight digit medline UI number.
    ReferenceMedline(u32),

    /// Reference Title.
    ReferenceTitle(String),

    /// Reference Author
    ReferenceAuthor(String),

    /// Journal location.
    ReferenceLocation(String),

    /// Record of all previous ID lines.
    PreviousIdentifier(String),

    /// Keywords.
    Keywords(Vec<String>),

    /// Comments.
    Comment(String),

    /// Indicates a nested domain.
    PfamAccession(String),

    /// Location of nested domains - sequence ID, start and end of insert.
    Location(String),

    /// Wikipedia page
    WikipediaLink(String),

    /// Clan accession
    Clan(String),

    /// Used for listing Clan membership
    Membership(String),

    /// A method used to set the bit score threshold based on the ratio of expected false positives
    /// to true positives. Floating point number between 0 and 1.
    FalseDiscoveryRate(f32),

    /// Command line used to calibrate the model (Rfam only, release 12.0 and later)
    CalibrationMethod(String),
}

impl FeatureAnnotation {
    pub fn write<W>(&self, mut writer: W) -> io::Result<()>
    where
        W: io::Write,
    {
        macro_rules! match_features {
            (@inner $($pat:pat => $expr:expr,)* ; $(,)? ) => {
                match self {
                    $($pat => $expr,)*
                }
            };

            (@inner $($pat:pat => $expr:expr,)* ; $feature:ident => $repr:literal, $($rest:tt)*) => {
                match_features!(
                    @inner
                    $($pat => $expr,)*
                    FeatureAnnotation::$feature(ann) => writeln!(writer, "#=GF {} {ann}", $repr),
                    ; $($rest)*
                )
            };

            (@inner $($pat:pat => $expr:expr,)* ; $feature:ident($feat_pat:pat) => $handle_ann:expr, $($rest:tt)*) => {
                match_features!(
                    @inner
                    $($pat => $expr,)*
                    FeatureAnnotation::$feature($feat_pat) => $handle_ann,
                    ; $($rest)*
                )
            };

            ($($tt:tt)*) => {
                match_features!(@inner ; $($tt)*)
            };
        }

        match_features!(
            AccessionNumber => "AC",
            Identification => "ID",
            Definition => "DE",
            Author => "AU",
            SourceOfSeed => "SE",
            SourceOfStructure => "SS",
            BuildMethod => "BM",
            SearchMethod => "SM",
            GatheringThreshold => "GA",
            TrustedCutoff => "TC",
            NoiseCutoff => "NC",
            Type => "TP",
            Sequence => "SQ",
            DatabaseComment => "DC",
            DatabaseReference => "DR",
            ReferenceComment => "RC",
            ReferenceNumber => "RN",
            ReferenceMedline => "RM",
            ReferenceTitle => "RT",
            ReferenceAuthor => "RA",
            ReferenceLocation => "RL",
            PreviousIdentifier => "PI",
            Keywords(keywords) => {
                writer.write_all(b"#=GF KW")?;
                let mut keywords = keywords.iter();
                if let Some(keyword) = keywords.next() {
                    write!(writer, " {keyword}")?;
                    keywords.try_for_each(|keyword| {
                        write!(writer, ",{keyword}")
                    })?;
                }
                writeln!(writer)
            },
            Comment => "CC",
            PfamAccession => "NE",
            Location => "NL",
            WikipediaLink => "WK",
            Clan => "CL",
            Membership => "MB",
            FalseDiscoveryRate => "FR",
            CalibrationMethod => "CB",
        )
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FamilyType {
    Family,
    Domain,
    Motif,
    Repeat,
}

impl Display for FamilyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FamilyType::Family => "Family",
            FamilyType::Domain => "Domain",
            FamilyType::Motif => "Motif",
            FamilyType::Repeat => "Repeat",
        };

        f.write_str(s)
    }
}
