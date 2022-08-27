use std::{
    fmt,
    fs::File,
    io::{self, BufWriter},
    path::Path,
};

use anyhow::Context;

use crate::{db_file, query_file, QueryResult, Sequence, SequenceEntry};

pub(crate) struct Entry<'a> {
    pub(crate) description: &'a str,
    pub(crate) sequence: Sequence<'a>,
}

impl fmt::Display for Entry<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ">{}\n{}", self.description, self.sequence)
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
    let &QueryResult {
        ref query,
        ref db_entry,
        query_start,
        query_end,
        db_start,
        db_end,
        ..
    } = result;

    format!(
        "{}_{}-{}_{}_{}-{}.fasta",
        db_entry, db_start, db_end, query, query_start, query_end
    )
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

    let db_sequence = Sequence(&db_entry.sequence()[db_start..=db_end]);
    let query_sequence = Sequence(&query_entry.sequence()[query_start..=query_end]);

    writeln!(
        writer,
        "{}\n{}",
        Entry {
            description: db_entry.name(),
            sequence: db_sequence
        },
        Entry {
            description: query_entry.name(),
            sequence: query_sequence
        }
    )
    .context("Unable to write to FASTA file")?;

    Ok::<_, anyhow::Error>(())
}

#[cfg(test)]
mod test {
    use crate::{QueryResultRange, QueryResultStatus};

    use super::*;

    #[test]
    fn write_result() {
        let query = query_file::read_file(Path::new("./test_data/query.txt"), 1.).unwrap();
        let db = db_file::read_file(Path::new("./test_data/test.db"), 1.).unwrap();
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
            status: QueryResultStatus::PassInclusionEvalue,
        };

        let mut writer = vec![];
        write_result_to_writer(&query_result, &db, &query, &mut writer).unwrap();
        let written = String::from_utf8(writer).unwrap();
        assert_eq!(written, ">16S_Bsubtilis\nGATCCT\n>16S_750\nCTCAGG\n")
    }
}
