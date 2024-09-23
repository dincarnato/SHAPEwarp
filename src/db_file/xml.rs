use std::{
    borrow::Cow,
    error::Error as StdError,
    fmt::{self, Display},
    fs::File,
    io::{self, BufReader},
    num::ParseFloatError,
    ops::Not,
    path::Path,
    str::Utf8Error,
};

use quick_xml::{
    events::{BytesEnd, BytesStart, BytesText},
    Reader,
};
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{Base, InvalidBase, Reactivity};

use super::{Entry, ReactivityWithPlaceholder};

pub fn read_file(path: &Path) -> Result<Entry, ReadFileError> {
    use quick_xml::events::Event;
    use ReadFileError as E;

    let mut reader = Reader::from_file(path).map_err(E::ReaderFromFile)?;
    let mut buffer = Vec::new();
    let mut state = XmlState::default();

    let mut id = None;
    let mut sequence = None;
    let mut reactivity = None;

    loop {
        let event = reader
            .read_event_into(&mut buffer)
            .map_err(|source| E::ReadEvent {
                buffer_position: reader.buffer_position(),
                source,
            })?;

        match event {
            Event::Start(start) => {
                state = handle_start_event(&start, state, &mut id)?;
            }

            Event::End(end) => {
                state = handle_end_event(&end, state)?;
            }

            Event::Empty(tag) => return Err(E::UnexpectedEmptyTag(tag.name().as_ref().to_owned())),
            Event::Text(text) => {
                handle_text_event(&text, &state, &mut sequence, &mut reactivity, &reader)?;
            }

            Event::CData(_)
            | Event::Comment(_)
            | Event::Decl(_)
            | Event::PI(_)
            | Event::DocType(_) => {}

            Event::Eof => break,
        }
    }

    let id = id.ok_or(E::MissingTranscript)?;
    let sequence = sequence.ok_or(E::MissingSequence)?;
    let reactivity = reactivity.ok_or(E::MissingReactivity)?;

    if sequence.len() != reactivity.len() {
        return Err(E::InconsistentLength {
            sequence: sequence.len(),
            reactivity: reactivity.len(),
        });
    }

    Ok(Entry {
        id,
        sequence,
        reactivity,
    })
}

fn handle_start_event(
    start: &BytesStart<'_>,
    state: XmlState,
    id: &mut Option<String>,
) -> Result<XmlState, ReadFileError> {
    use ReadFileError as E;

    match (start.name().as_ref(), state) {
        (b"data", XmlState::Start) => Ok(XmlState::Data),
        (b"meta-data", XmlState::Data) => Ok(XmlState::MetaData),
        (b"organism", XmlState::MetaData) => Ok(XmlState::Organism),
        (b"probe", XmlState::MetaData) => Ok(XmlState::Probe),
        (b"source", XmlState::MetaData) => Ok(XmlState::Source),
        (b"citation", XmlState::Source) => Ok(XmlState::Citation),
        (b"pmid", XmlState::Source) => Ok(XmlState::Pmid),
        (b"replicate", XmlState::MetaData) => Ok(XmlState::Replicate),
        (b"condition", XmlState::MetaData) => Ok(XmlState::Condition),
        (b"transcript", XmlState::Data) => {
            if id.is_some() {
                return Err(E::MultipleTranscripts);
            }

            let id_attr = start
                .try_get_attribute("id")
                .map_err(E::MalformedTranscriptTag)?
                .ok_or(E::MissingId)?;

            let id_string = match id_attr.value {
                Cow::Borrowed(id) => std::str::from_utf8(id)
                    .map(str::to_owned)
                    .map_err(E::InvalidId)?,
                Cow::Owned(id) => {
                    String::from_utf8(id).map_err(|err| E::InvalidId(err.utf8_error()))?
                }
            };
            *id = Some(id_string);

            Ok(XmlState::Transcript)
        }
        (b"sequence", XmlState::Transcript) => Ok(XmlState::Sequence),
        (b"reactivity", XmlState::Transcript) => Ok(XmlState::Reactivity),
        _ => Err(E::UnexpectedOpenTag(start.name().as_ref().to_owned())),
    }
}

fn handle_end_event(end: &BytesEnd<'_>, state: XmlState) -> Result<XmlState, ReadFileError> {
    use ReadFileError as E;

    match (end.name().as_ref(), state) {
        (b"data", XmlState::Data) => Ok(XmlState::End),

        (b"meta-data", XmlState::MetaData) | (b"transcript", XmlState::Transcript) => {
            Ok(XmlState::Data)
        }

        (b"organism", XmlState::Organism)
        | (b"probe", XmlState::Probe)
        | (b"source", XmlState::Source)
        | (b"replicate", XmlState::Replicate)
        | (b"condition", XmlState::Condition) => Ok(XmlState::MetaData),

        (b"citation", XmlState::Citation) | (b"pmid", XmlState::Pmid) => Ok(XmlState::Source),

        (b"sequence", XmlState::Sequence) | (b"reactivity", XmlState::Reactivity) => {
            Ok(XmlState::Transcript)
        }

        _ => Err(E::UnexpectedCloseTag(end.name().as_ref().to_owned())),
    }
}

fn handle_text_event(
    text: &BytesText<'_>,
    state: &XmlState,
    sequence: &mut Option<Vec<Base>>,
    reactivity: &mut Option<Vec<ReactivityWithPlaceholder>>,
    reader: &Reader<BufReader<File>>,
) -> Result<(), ReadFileError> {
    use ReadFileError as E;

    if text.iter().all(u8::is_ascii_whitespace) {
        return Ok(());
    }

    match state {
        XmlState::Start
        | XmlState::Data
        | XmlState::MetaData
        | XmlState::Source
        | XmlState::Transcript
        | XmlState::End => return Err(E::UnexpectedText(reader.buffer_position())),

        XmlState::Organism
        | XmlState::Probe
        | XmlState::Citation
        | XmlState::Pmid
        | XmlState::Replicate
        | XmlState::Condition => {}

        XmlState::Sequence => {
            if sequence.is_some() {
                return Err(E::MultipleSequences);
            }
            *sequence = Some(parse_sequence(text).map_err(E::InvalidSequence)?);
        }
        XmlState::Reactivity => {
            if reactivity.is_some() {
                return Err(E::MultipleReactivities);
            }

            *reactivity = Some(parse_reactivity(text).map_err(E::InvalidReactivity)?);
        }
    }

    Ok(())
}

#[derive(Debug, Default)]
enum XmlState {
    #[default]
    Start,
    Data,
    MetaData,
    Organism,
    Probe,
    Source,
    Citation,
    Pmid,
    Replicate,
    Condition,
    Transcript,
    Sequence,
    Reactivity,
    End,
}

#[derive(Debug)]
pub enum ReadFileError {
    ReaderFromFile(quick_xml::Error),
    ReadEvent {
        buffer_position: usize,
        source: quick_xml::Error,
    },
    UnexpectedOpenTag(Vec<u8>),
    UnexpectedCloseTag(Vec<u8>),
    UnexpectedEmptyTag(Vec<u8>),
    UnexpectedText(usize),
    MultipleTranscripts,
    MalformedTranscriptTag(quick_xml::Error),
    MissingId,
    InvalidId(Utf8Error),
    MultipleSequences,
    InvalidSequence(InvalidBase),
    MultipleReactivities,
    InvalidReactivity(InvalidReactivity),
    MissingTranscript,
    MissingSequence,
    MissingReactivity,
    InconsistentLength {
        sequence: usize,
        reactivity: usize,
    },
}

impl Display for ReadFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadFileError::ReaderFromFile(_) => {
                f.write_str("unable to create XML reader from file")
            }
            ReadFileError::ReadEvent {
                buffer_position,
                source: _,
            } => write!(f, "unable to read XML event at position {buffer_position}"),
            ReadFileError::UnexpectedOpenTag(tag) => write!(
                f,
                r#"unexpected opening tag "{}""#,
                String::from_utf8_lossy(tag)
            ),
            ReadFileError::UnexpectedCloseTag(tag) => write!(
                f,
                r#"unexpected closing tag "{}""#,
                String::from_utf8_lossy(tag),
            ),
            ReadFileError::UnexpectedEmptyTag(tag) => write!(
                f,
                r#"unexpected empty tag "{}""#,
                String::from_utf8_lossy(tag),
            ),
            ReadFileError::UnexpectedText(position) => {
                write!(f, "unexpected text content at position {position}")
            }
            ReadFileError::MultipleTranscripts => f.write_str("more than one transcript tag found"),
            ReadFileError::MalformedTranscriptTag(_) => {
                f.write_str("transcript tag has invalid or duplicated attributes")
            }
            ReadFileError::MissingId => {
                f.write_str(r#""id" attribute is missing from transcript tag"#)
            }
            ReadFileError::InvalidId(_) => f.write_str("transcript id is not a valid UTF-8 string"),
            ReadFileError::MultipleSequences => f.write_str("more than one sequence tag found"),
            ReadFileError::InvalidSequence(_) => f.write_str("sequence is invalid"),
            ReadFileError::MultipleReactivities => {
                f.write_str("more than one reactivity tag found")
            }
            ReadFileError::InvalidReactivity(_) => f.write_str("reactivity data is invalid"),
            ReadFileError::MissingTranscript => f.write_str("transcript tag is missing"),
            ReadFileError::MissingSequence => f.write_str("sequence tag is missing"),
            ReadFileError::MissingReactivity => f.write_str("reactivity tag is missing"),
            ReadFileError::InconsistentLength {
                sequence,
                reactivity,
            } => write!(
                f,
                "sequence length ({sequence}) is different from reactivity sequence {reactivity}"
            ),
        }
    }
}

impl StdError for ReadFileError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ReadFileError::ReaderFromFile(source) | ReadFileError::ReadEvent { source, .. } => {
                Some(source)
            }

            ReadFileError::UnexpectedOpenTag(_)
            | ReadFileError::UnexpectedCloseTag(_)
            | ReadFileError::UnexpectedEmptyTag(_)
            | ReadFileError::UnexpectedText(_)
            | ReadFileError::MultipleTranscripts
            | ReadFileError::MissingId
            | ReadFileError::MultipleSequences
            | ReadFileError::MultipleReactivities
            | ReadFileError::MissingTranscript
            | ReadFileError::MissingSequence
            | ReadFileError::MissingReactivity
            | ReadFileError::InconsistentLength { .. } => None,

            ReadFileError::MalformedTranscriptTag(source) => Some(source),
            ReadFileError::InvalidId(source) => Some(source),
            ReadFileError::InvalidSequence(source) => Some(source),
            ReadFileError::InvalidReactivity(source) => Some(source),
        }
    }
}

fn parse_sequence(raw: &[u8]) -> Result<Vec<Base>, InvalidBase> {
    raw.iter()
        .filter(|c| c.is_ascii_whitespace().not())
        .copied()
        .map(Base::try_from)
        .collect()
}

fn parse_reactivity(raw: &[u8]) -> Result<Vec<ReactivityWithPlaceholder>, InvalidReactivity> {
    use InvalidReactivity as E;

    raw.split(|&c| c == b',')
        .map(|raw| {
            let raw = std::str::from_utf8(raw).map_err(E::Utf8)?.trim();

            if raw == "NaN" {
                Ok(ReactivityWithPlaceholder::nan_placeholder())
            } else {
                raw.parse::<Reactivity>()
                    .map(ReactivityWithPlaceholder::from)
                    .map_err(InvalidReactivity::Value)
            }
        })
        .collect()
}

#[derive(Debug)]
pub enum InvalidReactivity {
    Utf8(Utf8Error),
    Value(ParseFloatError),
}

impl Display for InvalidReactivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            InvalidReactivity::Utf8(_) => "rectivity is not a valid UTF-8 string",
            InvalidReactivity::Value(_) => "unable to parse reactivity value",
        };

        f.write_str(s)
    }
}

impl StdError for InvalidReactivity {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            InvalidReactivity::Utf8(source) => Some(source),
            InvalidReactivity::Value(source) => Some(source),
        }
    }
}

pub fn read_directory(path: &Path) -> Result<Vec<Entry>, ReadDirectoryError> {
    use ReadDirectoryError as E;

    path.read_dir()
        .map_err(E::Dir)?
        .filter_map(|entry| {
            entry
                .map(|entry| {
                    let path = entry.path();
                    let extension = path.extension()?;
                    extension.eq_ignore_ascii_case("xml").then_some(path)
                })
                .transpose()
        })
        .par_bridge()
        .filter_map(|path| {
            let path = match path {
                Ok(path) => path,
                Err(err) => return Some(Err(E::DirEntry(err))),
            };
            match read_file(&path) {
                Ok(entry) => Some(Ok(entry)),
                Err(err) => {
                    eprintln!(
                        "WARNING: unable to read XML path {}: {:#}",
                        path.display(),
                        anyhow::Error::from(err)
                    );
                    None
                }
            }
        })
        .collect()
}

#[derive(Debug)]
pub enum ReadDirectoryError {
    Dir(io::Error),
    DirEntry(io::Error),
}

impl Display for ReadDirectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadDirectoryError::Dir(_) => f.write_str("unable to read directory"),
            ReadDirectoryError::DirEntry(_) => f.write_str("unable to read directory entry"),
        }
    }
}

impl StdError for ReadDirectoryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            ReadDirectoryError::Dir(source) | ReadDirectoryError::DirEntry(source) => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::OnceLock,
    };

    use tempfile::tempdir;

    use crate::{db_file::ReactivityWithPlaceholder, Base};

    use super::{read_directory, read_file};

    fn raw_xml_db_path() -> &'static Path {
        static RAW_XML_DB_PATH: OnceLock<PathBuf> = OnceLock::new();

        RAW_XML_DB_PATH.get_or_init(|| {
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            manifest_dir.join("test_data/test_db.xml")
        })
    }

    #[test]
    fn read_valid_xml() {
        let entry = read_file(raw_xml_db_path()).unwrap();
        assert_eq!(entry.id, "Saccharomyces.cerevisiae_rc:URS00005F2C2D_18S");
        assert_eq!(entry.sequence.len(), 1800);
        assert_eq!(
            entry.sequence[..5],
            [Base::T, Base::A, Base::T, Base::C, Base::T]
        );
        assert!(entry.reactivity[..37]
            .iter()
            .copied()
            .all(ReactivityWithPlaceholder::is_nan));
        assert!((entry.reactivity[37].get_non_nan().unwrap() - 0.389).abs() < 0.001);
    }

    #[test]
    fn read_directory_ignores_non_xml_files() {
        let tempdir = tempdir().unwrap();
        let temp_path = tempdir.path();
        fs::write(temp_path.join("test.txt"), "hello world").unwrap();
        fs::copy(raw_xml_db_path(), temp_path.join("valid.xml")).unwrap();
        let entries = read_directory(temp_path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].id,
            "Saccharomyces.cerevisiae_rc:URS00005F2C2D_18S",
        );
    }
}
