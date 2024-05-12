use std::{
    convert::TryInto,
    error::Error as StdError,
    fmt::{self, Display},
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::Path,
    ptr,
    string::FromUtf8Error,
};

use itertools::Itertools;
use serde::{Serialize, Serializer};

use crate::{Base, InvalidBasePair, Molecule, Reactivity, SequenceEntry};

const END_SIZE: u8 = 17;
const END_MARKER: &[u8] = b"[eofdb]";
const VERSION: u16 = 1;

#[derive(Debug)]
pub struct Reader<R> {
    inner: R,
    _db_len: u64,
    _version: u16,
    end_offset: u64,
}

impl<R> Reader<R>
where
    R: Read + Seek,
{
    pub fn new(mut reader: R) -> Result<Self, NewReaderError> {
        use NewReaderError as E;

        let end_offset = reader
            .seek(SeekFrom::End(-i64::from(END_SIZE)))
            .map_err(E::SeekToMetadata)?;
        let mut end_buf = [0; END_SIZE as usize];
        reader.read_exact(&mut end_buf).map_err(E::ReadMetadata)?;

        if &end_buf[10..17] != END_MARKER {
            return Err(E::InvalidMarker);
        }

        let db_len = u64::from_le_bytes(end_buf[0..8].try_into().unwrap());
        let version = u16::from_le_bytes(end_buf[8..10].try_into().unwrap());
        Ok(Self {
            inner: reader,
            _db_len: db_len,
            _version: version,
            end_offset,
        })
    }

    pub fn entries(&mut self) -> EntryIter<R> {
        let &mut Self {
            ref mut inner,
            end_offset,
            ..
        } = self;

        EntryIter {
            reader: inner,
            end_offset,
            offset: 0,
        }
    }
}

#[derive(Debug)]
pub enum NewReaderError {
    SeekToMetadata(io::Error),
    ReadMetadata(io::Error),
    InvalidMarker,
}

impl Display for NewReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            NewReaderError::SeekToMetadata(_) => "unable to seek to metadata",
            NewReaderError::ReadMetadata(_) => "unable to read metadata",
            NewReaderError::InvalidMarker => "invalid metadata marker",
        };

        f.write_str(s)
    }
}

impl StdError for NewReaderError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            NewReaderError::SeekToMetadata(source) | NewReaderError::ReadMetadata(source) => {
                Some(source)
            }
            NewReaderError::InvalidMarker => None,
        }
    }
}

#[derive(Debug)]
pub struct EntryIter<'a, R> {
    reader: &'a mut R,
    end_offset: u64,
    offset: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub id: String,
    pub(crate) sequence: Vec<Base>,
    pub reactivity: Vec<ReactivityWithPlaceholder>,
}

const NAN_PLACEHOLDER: Reactivity = -999.;

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct ReactivityWithPlaceholder(Reactivity);

impl ReactivityWithPlaceholder {
    pub fn is_nan(self) -> bool {
        self.0.is_nan() | (self.0 == NAN_PLACEHOLDER)
    }

    pub fn get_non_nan(self) -> Option<Reactivity> {
        if self.is_nan() {
            None
        } else {
            Some(self.0)
        }
    }

    pub fn to_maybe_placeholder(self) -> Reactivity {
        if self.0.is_nan() {
            NAN_PLACEHOLDER
        } else {
            self.0
        }
    }

    pub fn as_inner_slice(this: &[ReactivityWithPlaceholder]) -> &[Reactivity] {
        // Safety:
        // - `ReactivityWithPlaceholder` is transparent and it contains only a `Reactivity`
        // - lifetime is maintained
        unsafe { &*(ptr::from_ref(this) as *const [Reactivity]) }
    }

    pub fn inner(self) -> Reactivity {
        self.0
    }
}

impl PartialEq for ReactivityWithPlaceholder {
    fn eq(&self, other: &Self) -> bool {
        if (self.0 == NAN_PLACEHOLDER) | (other.0 == NAN_PLACEHOLDER) {
            false
        } else {
            self.0 == other.0
        }
    }
}

impl PartialEq<Reactivity> for ReactivityWithPlaceholder {
    fn eq(&self, other: &Reactivity) -> bool {
        if self.0 == NAN_PLACEHOLDER {
            false
        } else {
            self.0 == *other
        }
    }
}

impl PartialOrd for ReactivityWithPlaceholder {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if (self.0 == NAN_PLACEHOLDER) | (other.0 == NAN_PLACEHOLDER) {
            None
        } else {
            self.0.partial_cmp(&other.0)
        }
    }
}

impl PartialOrd<Reactivity> for ReactivityWithPlaceholder {
    fn partial_cmp(&self, other: &Reactivity) -> Option<std::cmp::Ordering> {
        if self.0 == NAN_PLACEHOLDER {
            None
        } else {
            self.0.partial_cmp(other)
        }
    }
}

impl From<Reactivity> for ReactivityWithPlaceholder {
    fn from(reactivity: Reactivity) -> Self {
        Self(reactivity)
    }
}

impl Serialize for ReactivityWithPlaceholder {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.get_non_nan()
            .unwrap_or(Reactivity::NAN)
            .serialize(serializer)
    }
}

pub trait ReactivityLike: Copy + PartialOrd + PartialEq {
    fn is_nan(self) -> bool;
    fn value(self) -> Reactivity;
}

impl ReactivityLike for Reactivity {
    #[inline]
    fn is_nan(self) -> bool {
        Reactivity::is_nan(self)
    }

    #[inline]
    fn value(self) -> Reactivity {
        self
    }
}

impl ReactivityLike for ReactivityWithPlaceholder {
    #[inline]
    fn is_nan(self) -> bool {
        ReactivityWithPlaceholder::is_nan(self)
    }

    #[inline]
    fn value(self) -> Reactivity {
        self.to_maybe_placeholder()
    }
}

impl Entry {
    pub fn cap_reactivities(&mut self, max_reactivity: Reactivity) {
        self.reactivity.iter_mut().for_each(|reactivity| {
            if let Some(r) = reactivity.get_non_nan() {
                *reactivity = r.min(max_reactivity).into();
            }
        });
    }
}

impl SequenceEntry for Entry {
    type Reactivity = ReactivityWithPlaceholder;

    fn name(&self) -> &str {
        &self.id
    }

    fn sequence(&self) -> &[Base] {
        &self.sequence
    }

    fn reactivity(&self) -> &[Self::Reactivity] {
        &self.reactivity
    }

    fn molecule(&self) -> crate::Molecule {
        Molecule::Dna
    }
}

impl<R> Iterator for EntryIter<'_, R>
where
    R: Seek + Read,
{
    type Item = Result<Entry, NextEntryError>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.offset != self.end_offset).then(|| self.next_entry())
    }
}

impl<R> EntryIter<'_, R>
where
    R: Seek + Read,
{
    fn next_entry(&mut self) -> Result<Entry, NextEntryError> {
        use NextEntryError as E;

        if self.offset == 0 {
            self.reader.seek(SeekFrom::Start(0)).map_err(E::SeekStart)?;
        }

        let mut id_len_with_nul_buf = [0; 4];
        self.reader
            .read_exact(&mut id_len_with_nul_buf)
            .map_err(E::ReadIdLen)?;
        let id_len_with_nul: usize = u32::from_le_bytes(id_len_with_nul_buf)
            .try_into()
            .expect("cannot represent id length as usize for the current architecture");
        let mut sequence_id = vec![0; id_len_with_nul];
        self.reader
            .read_exact(&mut sequence_id)
            .map_err(E::ReadSequenceId)?;
        if sequence_id.pop().filter(|&b| b == 0).is_none() {
            return Err(E::MissingSequenceIdNul);
        }
        let sequence_id =
            String::from_utf8(sequence_id).map_err(NextEntryError::InvalidSequenceId)?;
        let mut sequence_len_buf = [0; 4];
        self.reader
            .read_exact(&mut sequence_len_buf)
            .map_err(E::ReadSequenceLen)?;
        let sequence_len: usize = u32::from_le_bytes(sequence_len_buf)
            .try_into()
            .expect("cannot represent sequence length as usize for the current architecture");

        let sequence_bytes = sequence_len / 2 + sequence_len % 2;
        let mut sequence = self
            .reader
            .bytes()
            .take(sequence_bytes)
            .map(|result| {
                result.map_err(E::ReadSequence).and_then(|byte| {
                    Base::try_pair_from_byte(byte)
                        .map(|[first, second]| [first, second])
                        .map_err(E::InvalidEncodedBase)
                })
            })
            .flatten_ok()
            .collect::<Result<Vec<_>, _>>()?;

        if sequence_len > 0 && sequence_len % 2 == 1 {
            sequence.pop().unwrap();
        }

        if sequence.len() != sequence_len {
            return Err(E::UnexpectedEof);
        }

        let reactivity = (0..sequence_len)
            .map(|_| {
                let mut reactivity_buffer = [0; 8];
                self.reader
                    .read_exact(&mut reactivity_buffer)
                    .map(|()| reactivity_buffer)
                    .map_err(E::ReadReactivity)
            })
            // Reactivity is an alias to either f32 or f64
            .map_ok(|bytes| {
                // We internally use a fixed type that can be f32, there is no need to necessarily
                // have 64 bits of precision
                #[allow(clippy::cast_possible_truncation)]
                let reactivity = f64::from_le_bytes(bytes) as Reactivity;
                ReactivityWithPlaceholder::from(reactivity)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if reactivity.len() != sequence_len {
            return Err(E::UnexpectedEof);
        }

        let offset = self.reader.stream_position().map_err(E::StreamPosition)?;
        if offset > self.end_offset {
            return Err(E::SurpassedEofMarker);
        }
        self.offset = offset;

        Ok(Entry {
            id: sequence_id,
            sequence,
            reactivity,
        })
    }
}

#[derive(Debug)]
pub enum NextEntryError {
    SeekStart(io::Error),
    ReadIdLen(io::Error),
    ReadSequenceId(io::Error),
    MissingSequenceIdNul,
    InvalidSequenceId(FromUtf8Error),
    ReadSequenceLen(io::Error),
    ReadSequence(io::Error),
    InvalidEncodedBase(InvalidBasePair),
    ReadReactivity(io::Error),
    UnexpectedEof,
    SurpassedEofMarker,
    StreamPosition(io::Error),
}

impl Display for NextEntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            NextEntryError::SeekStart(_) => "unable to seek to the start of the file",
            NextEntryError::ReadIdLen(_) => "unable to read the length of the sequence id",
            NextEntryError::ReadSequenceId(_) => "unable to read sequence id",
            NextEntryError::MissingSequenceIdNul => {
                "sequence id does not have a nul termination character"
            }
            NextEntryError::InvalidSequenceId(_) => "sequence id is not valid",
            NextEntryError::ReadSequenceLen(_) => "unable to read sequence length",
            NextEntryError::ReadSequence(_) => "unable to read sequence content",
            NextEntryError::InvalidEncodedBase(_) => "invalid encoded base",
            NextEntryError::ReadReactivity(_) => "unable to read rectivity",
            NextEntryError::UnexpectedEof => "unexpected end of file",
            NextEntryError::SurpassedEofMarker => "end of file marker is being surpassed",
            NextEntryError::StreamPosition(_) => "unable to get stream position",
        };

        f.write_str(s)
    }
}

impl StdError for NextEntryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            NextEntryError::SeekStart(source)
            | NextEntryError::ReadIdLen(source)
            | NextEntryError::ReadSequenceId(source)
            | NextEntryError::ReadSequenceLen(source)
            | NextEntryError::ReadSequence(source)
            | NextEntryError::ReadReactivity(source)
            | NextEntryError::StreamPosition(source) => Some(source),
            NextEntryError::MissingSequenceIdNul
            | NextEntryError::UnexpectedEof
            | NextEntryError::SurpassedEofMarker => None,
            NextEntryError::InvalidSequenceId(source) => Some(source),
            NextEntryError::InvalidEncodedBase(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum ReaderError {
    TooSmall,
    InvalidMarker,
}

impl Display for ReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ReaderError::TooSmall => "DB file is too small",
            ReaderError::InvalidMarker => "DB file contains and invalid EOF marker",
        };

        f.write_str(s)
    }
}

impl StdError for ReaderError {}

#[derive(Debug)]
pub enum EntryError {
    InvalidSequenceId(FromUtf8Error),
    InvalidBase,
    UnexpectedEof,
    SurpassedEofMarker,
}

impl Display for EntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            EntryError::InvalidSequenceId(_) => "Invalid sequence ID string",
            EntryError::InvalidBase => "Invalid encoded nucleobase",
            EntryError::UnexpectedEof => "Unexpected end of file",
            EntryError::SurpassedEofMarker => "End of file marked has been surpassed",
        };

        f.write_str(s)
    }
}

impl StdError for EntryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            EntryError::InvalidSequenceId(source) => Some(source),
            EntryError::InvalidBase
            | EntryError::UnexpectedEof
            | EntryError::SurpassedEofMarker => None,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    OpenFile(io::Error),
    NewReader(NewReaderError),
    Entry(NextEntryError),
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Error::OpenFile(_) => "unable to open file",
            Error::NewReader(_) => "unable to create new reader",
            Error::Entry(_) => "unable to get the next entry",
        };

        f.write_str(s)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::OpenFile(source) => Some(source),
            Error::NewReader(source) => Some(source),
            Error::Entry(source) => Some(source),
        }
    }
}

pub fn read_file(path: &Path) -> Result<Vec<Entry>, Error> {
    let file = File::open(path).map_err(Error::OpenFile)?;
    let mut reader = Reader::new(BufReader::new(file)).map_err(Error::NewReader)?;
    let entries = reader
        .entries()
        .collect::<Result<_, _>>()
        .map_err(Error::Entry)?;
    Ok(entries)
}

pub fn write_entries<W: io::Write>(entries: &[Entry], mut writer: W) -> io::Result<()> {
    entries.iter().try_for_each(|entry| {
        let name = entry.name();
        let sequence = entry.sequence();
        let name_len_buf = u32::try_from(name.len().checked_add(1).unwrap())
            .unwrap()
            .to_le_bytes();
        let seq_len_buf = u32::try_from(sequence.len()).unwrap().to_le_bytes();

        writer.write_all(name_len_buf.as_slice())?;
        writer.write_all(name.as_bytes())?;
        writer.write_all(&[0])?;
        writer.write_all(seq_len_buf.as_slice())?;
        sequence.chunks_exact(2).try_for_each(|pair| {
            writer.write_all(&[Base::pair_to_nibble(pair.try_into().unwrap())])
        })?;
        if let Some(base) = sequence.chunks_exact(2).remainder().first().copied() {
            writer.write_all(&[Base::pair_to_nibble([base, Base::A])])?;
        }

        entry.reactivity().iter().try_for_each(|reactivity| {
            let reactivity = f64::from(reactivity.inner()).to_le_bytes();
            writer.write_all(reactivity.as_slice())
        })?;

        Ok::<_, io::Error>(())
    })?;

    let n_entries = u64::try_from(entries.len()).unwrap().to_le_bytes();
    writer.write_all(n_entries.as_slice())?;
    writer.write_all(VERSION.to_le_bytes().as_slice())?;
    writer.write_all(END_MARKER)?;
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    const TEST_DB: &[u8] = include_bytes!("../test_data/test.db");

    #[test]
    fn valid_reader() {
        let reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        #[allow(clippy::used_underscore_binding)]
        let len = reader._db_len;

        #[allow(clippy::used_underscore_binding)]
        let version = reader._version;

        assert_eq!(len, 0x1181);
        assert_eq!(version, 1);
    }

    #[test]
    fn read_all_db() {
        let mut reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        let db_len = reader
            .entries()
            .map_ok(|entry| entry.sequence.len())
            .try_fold(0, |acc, seq_len| seq_len.map(|seq_len| acc + seq_len))
            .unwrap();

        #[allow(clippy::used_underscore_binding)]
        let reader_len = usize::try_from(reader._db_len).unwrap();
        assert_eq!(db_len, reader_len);
    }

    #[test]
    fn transform_pseudo_nans() {
        let mut reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        let entry = reader.entries().next().unwrap().unwrap();

        // The first 13 reactivities are -999 in the file
        assert!(entry.reactivity[..13]
            .iter()
            .copied()
            .all(ReactivityWithPlaceholder::is_nan));
    }
}
