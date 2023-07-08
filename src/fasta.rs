use std::{
    fmt,
    fs::File,
    io::{self, BufWriter},
    path::Path,
};

use anyhow::Context;

use crate::{
    aligner::{AlignedSequence, BaseOrGap},
    db_file, query_file, QueryResult, ResultFileFormat, Sequence, SequenceEntry,
};

pub(crate) struct Entry<'a> {
    pub(crate) description: &'a str,
    pub(crate) sequence: Sequence<'a>,
    pub(crate) alignment: Option<&'a AlignedSequence>,
}

impl fmt::Display for Entry<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, ">{}", self.description)?;
        match self.alignment {
            Some(alignment) => {
                let mut sequence = self.sequence.bases.iter();
                for base_or_gap in &alignment.0 {
                    match base_or_gap {
                        BaseOrGap::Base => match sequence.next() {
                            Some(base) => write!(f, "{}", base.display(self.sequence.molecule))?,
                            None => break,
                        },
                        BaseOrGap::Gap => f.write_str("-")?,
                    }
                }
                sequence.try_for_each(|base| write!(f, "{}", base.display(self.sequence.molecule)))
            }
            None => write!(f, "{}", self.sequence),
        }
    }
}

pub(crate) fn write_result(
    result: &QueryResult,
    db_entries: &[db_file::Entry],
    query_entries: &[query_file::Entry],
    alignments_path: &Path,
) -> Result<(), anyhow::Error> {
    let fasta_path = alignments_path.join(result_filename(result));
    let file = File::create(fasta_path).context("Unable to create FASTA file")?;
    let writer = BufWriter::new(file);

    write_result_to_writer(result, db_entries, query_entries, writer)
}

fn result_filename(result: &QueryResult) -> String {
    format!("{}.fasta", ResultFileFormat::from(result))
}

#[inline]
fn write_result_to_writer<W: io::Write>(
    result: &QueryResult,
    db_entries: &[db_file::Entry],
    query_entries: &[query_file::Entry],
    mut writer: W,
) -> Result<(), anyhow::Error> {
    let &QueryResult {
        ref query,
        ref db_entry,
        query_start,
        query_end,
        db_start,
        db_end,
        ref alignment,
        ..
    } = result;

    let db_entry = db_entries
        .iter()
        .find(|entry| entry.name() == db_entry)
        .expect("db entry should be available");
    let query_entry = query_entries
        .iter()
        .find(|entry| entry.name() == &**query)
        .expect("query entry should be available");

    let db_sequence = Sequence {
        bases: &db_entry.sequence()[db_start..=db_end],
        molecule: db_entry.molecule(),
    };
    let query_sequence = Sequence {
        bases: &query_entry.sequence()[query_start..=query_end],
        molecule: query_entry.molecule(),
    };

    writeln!(
        writer,
        "{}\n{}",
        Entry {
            description: db_entry.name(),
            sequence: db_sequence,
            alignment: Some(&alignment.target),
        },
        Entry {
            description: query_entry.name(),
            sequence: query_sequence,
            alignment: Some(&alignment.query),
        }
    )
    .context("Unable to write to FASTA file")?;

    Ok::<_, anyhow::Error>(())
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{aligner::BaseOrGap, Molecule, QueryResultRange, QueryResultStatus};

    use super::*;

    #[test]
    fn write_result() {
        let query = query_file::read_file(Path::new("./test_data/query.txt")).unwrap();
        let db = db_file::read_file(Path::new("./test_data/test.db")).unwrap();
        let query_name = query[0].name().into();
        let db_name = db[0].name().to_owned();

        let query_result = QueryResult {
            query: query_name,
            db_entry: db_name,
            query_start: 5,
            query_end: 10,
            db_start: 15,
            db_end: 20,
            query_seed: QueryResultRange(0..=10),
            db_seed: QueryResultRange(0..=10),
            score: 0.,
            pvalue: 0.,
            evalue: 0.,
            target_bp_support: Option::default(),
            query_bp_support: Option::default(),
            mfe_pvalue: Option::default(),
            status: QueryResultStatus::PassInclusionEvalue,
            alignment: Arc::default(),
            dotbracket: Option::default(),
        };

        let mut writer = vec![];
        write_result_to_writer(&query_result, &db, &query, &mut writer).unwrap();
        let written = String::from_utf8(writer).unwrap();
        assert_eq!(written, ">16S_Bsubtilis\nGATCCT\n>16S_750\nCTCAGG\n");
    }

    #[test]
    fn write_result_rna() {
        let mut query = query_file::read_file(Path::new("./test_data/query.txt")).unwrap();
        query[0].molecule = Molecule::Rna;
        let db = db_file::read_file(Path::new("./test_data/test.db")).unwrap();
        let query_name = query[0].name().into();
        let db_name = db[0].name().to_owned();

        let query_result = QueryResult {
            query: query_name,
            db_entry: db_name,
            query_start: 5,
            query_end: 10,
            db_start: 15,
            db_end: 20,
            query_seed: QueryResultRange(0..=10),
            db_seed: QueryResultRange(0..=10),
            score: 0.,
            pvalue: 0.,
            evalue: 0.,
            target_bp_support: Option::default(),
            query_bp_support: Option::default(),
            mfe_pvalue: Option::default(),
            status: QueryResultStatus::PassInclusionEvalue,
            alignment: Arc::default(),
            dotbracket: Option::default(),
        };

        let mut writer = vec![];
        write_result_to_writer(&query_result, &db, &query, &mut writer).unwrap();
        let written = String::from_utf8(writer).unwrap();
        assert_eq!(written, ">16S_Bsubtilis\nGATCCT\n>16S_750\nCUCAGG\n");
    }

    #[test]
    fn result_filename() {
        let query = query_file::read_file(Path::new("./test_data/query.txt")).unwrap();
        let db = db_file::read_file(Path::new("./test_data/test.db")).unwrap();
        let query_name = query[0].name().into();
        let db_name = db[0].name().to_owned();

        let query_result = QueryResult {
            query: query_name,
            db_entry: db_name,
            query_start: 5,
            query_end: 10,
            db_start: 15,
            db_end: 20,
            query_seed: QueryResultRange(0..=10),
            db_seed: QueryResultRange(0..=10),
            score: 0.,
            pvalue: 0.,
            evalue: 0.,
            target_bp_support: Option::default(),
            query_bp_support: Option::default(),
            mfe_pvalue: Option::default(),
            status: QueryResultStatus::PassInclusionEvalue,
            alignment: Arc::default(),
            dotbracket: Option::default(),
        };

        assert_eq!(
            super::result_filename(&query_result),
            "16S_Bsubtilis_15-20_16S_750_5-10.fasta"
        );
    }

    #[test]
    fn display_aligned_entry() {
        use crate::Base::*;
        use BaseOrGap::*;

        let alignment = AlignedSequence(vec![Base, Base, Gap, Base, Gap, Gap, Base]);
        let entry = Entry {
            description: "test",
            sequence: Sequence {
                bases: &[A, C, T, G, A, A],
                molecule: crate::Molecule::Dna,
            },
            alignment: Some(&alignment),
        };

        assert_eq!(entry.to_string(), ">test\nAC-T--GAA");
    }
}
