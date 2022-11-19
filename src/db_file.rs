use std::{
    convert::TryInto,
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize, Serializer};

use crate::{Base, Molecule, Reactivity, SequenceEntry};

const END_SIZE: u8 = 17;
const END_MARKER: &[u8] = b"[eofdb]";

#[derive(Debug)]
pub struct Reader<R> {
    reader: R,
    _db_len: u64,
    _version: u16,
    end_offset: u64,
}

impl<R> Reader<R>
where
    R: Read + Seek,
{
    pub fn new(mut reader: R) -> Result<Self, Error> {
        let end_offset = reader
            .seek(SeekFrom::End(-i64::from(END_SIZE)))
            .map_err(|_| ReaderError::TooSmall)?;
        let mut end_buf = [0; END_SIZE as usize];
        reader.read_exact(&mut end_buf)?;

        if &end_buf[10..17] != END_MARKER {
            return Err(ReaderError::InvalidMarker.into());
        }

        let _db_len = u64::from_le_bytes(end_buf[0..8].try_into().unwrap());
        let _version = u16::from_le_bytes(end_buf[8..10].try_into().unwrap());
        Ok(Self {
            reader,
            _db_len,
            _version,
            end_offset,
        })
    }

    pub fn entries(&mut self) -> EntryIter<R> {
        let &mut Self {
            ref mut reader,
            end_offset,
            ..
        } = self;

        EntryIter {
            reader,
            end_offset,
            offset: 0,
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

#[derive(Debug, Clone, Copy, Deserialize)]
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
}

impl ReactivityLike for Reactivity {
    #[inline]
    fn is_nan(self) -> bool {
        Reactivity::is_nan(self)
    }
}

impl ReactivityLike for ReactivityWithPlaceholder {
    #[inline]
    fn is_nan(self) -> bool {
        ReactivityWithPlaceholder::is_nan(self)
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
    type Item = Result<Entry, EntryIoError>;

    fn next(&mut self) -> Option<Self::Item> {
        macro_rules! ok {
            ($expr:expr) => {
                match $expr {
                    Ok(ok) => ok,
                    Err(err) => return Some(Err(err.into())),
                }
            };
        }

        if self.offset == self.end_offset {
            return None;
        }

        if self.offset == 0 {
            ok!(self.reader.seek(SeekFrom::Start(0)));
        }

        let mut id_len_with_nul_buf = [0; 4];
        ok!(self.reader.read_exact(&mut id_len_with_nul_buf));
        let id_len_with_nul: usize = u32::from_le_bytes(id_len_with_nul_buf)
            .try_into()
            .expect("cannot represent id length as usize for the current architecture");
        let mut sequence_id = vec![0; id_len_with_nul];
        ok!(self.reader.read_exact(&mut sequence_id));
        if sequence_id.pop().filter(|&b| b == 0).is_none() {
            return Some(Err(EntryError::InvalidSequenceId.into()));
        }
        let sequence_id =
            ok!(String::from_utf8(sequence_id).map_err(|_| { EntryError::InvalidSequenceId }));
        let mut sequence_len_buf = [0; 4];
        ok!(self.reader.read_exact(&mut sequence_len_buf));
        let sequence_len: usize = u32::from_le_bytes(sequence_len_buf)
            .try_into()
            .expect("cannot represent sequence length as usize for the current architecture");

        let sequence_bytes = sequence_len / 2 + sequence_len % 2;
        let mut sequence = ok!(self
            .reader
            .bytes()
            .take(sequence_bytes)
            .map(
                |result| result.map_err(EntryIoError::from).and_then(|byte| {
                    Base::try_pair_from_byte(byte)
                        .map(|[first, second]| [first, second])
                        .map_err(|_| EntryError::InvalidBase.into())
                })
            )
            .flatten_ok()
            .collect::<Result<Vec<_>, _>>());

        if sequence_len > 0 && sequence_len % 2 == 1 {
            sequence.pop().unwrap();
        }

        if sequence.len() != sequence_len {
            return Some(Err(EntryError::UnexpectedEof.into()));
        }

        let reactivity = ok!((0..sequence_len)
            .map(|_| {
                let mut reactivity_buffer = [0; 8];
                self.reader
                    .read_exact(&mut reactivity_buffer)
                    .map(|()| reactivity_buffer)
            })
            // Reactivity is an alias to either f32 or f64
            .map_ok(|bytes| {
                let reactivity = f64::from_le_bytes(bytes) as Reactivity;
                ReactivityWithPlaceholder::from(reactivity)
            })
            .collect::<Result<Vec<_>, _>>());

        if reactivity.len() != sequence_len {
            return Some(Err(EntryError::UnexpectedEof.into()));
        }

        let offset = ok!(self.reader.stream_position());
        if offset > self.end_offset {
            return Some(Err(EntryError::SurpassedEofMarker.into()));
        }
        self.offset = offset;

        Some(Ok(Entry {
            id: sequence_id,
            sequence,
            reactivity,
        }))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[error("DB file is too small")]
    TooSmall,

    #[error("DB file contains and invalid EOF marker")]
    InvalidMarker,
}

#[derive(Debug, thiserror::Error)]
pub enum EntryIoError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Entry error: {0}")]
    Entry(#[from] EntryError),
}

#[derive(Debug, thiserror::Error)]
pub enum EntryError {
    #[error("Invalid sequence ID string")]
    InvalidSequenceId,

    #[error("Invalid encoded nucleobase")]
    InvalidBase,

    #[error("Unexpected end of file")]
    UnexpectedEof,

    #[error("End of file marked has been surpassed")]
    SurpassedEofMarker,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("DB reader error: {0}")]
    Reader(#[from] ReaderError),

    #[error("Entry error: {0}")]
    Entry(#[from] EntryError),
}

impl From<EntryIoError> for Error {
    fn from(error: EntryIoError) -> Self {
        match error {
            EntryIoError::Io(e) => Error::Io(e),
            EntryIoError::Entry(e) => Error::Entry(e),
        }
    }
}

pub fn read_file(path: &Path) -> Result<Vec<Entry>, Error> {
    let mut reader = Reader::new(BufReader::new(File::open(path)?))?;
    let entries = reader.entries().collect::<Result<_, _>>()?;
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    const TEST_DB: &[u8] = include_bytes!("../test_data/test.db");

    #[test]
    fn valid_reader() {
        let reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        assert_eq!(reader._db_len, 0x1181);
        assert_eq!(reader._version, 1);
    }

    #[test]
    fn read_all_db() {
        let mut reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        let db_len = reader
            .entries()
            .map_ok(|entry| entry.sequence.len())
            .try_fold(0, |acc, seq_len| seq_len.map(|seq_len| acc + seq_len))
            .unwrap();
        assert_eq!(db_len, usize::try_from(reader._db_len).unwrap());
    }

    #[test]
    fn transform_pseudo_nans() {
        let mut reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        let entry = reader.entries().next().unwrap().unwrap();

        // The first 13 reactivities are -999 in the file
        assert!(entry.reactivity[..13]
            .iter()
            .copied()
            .all(|reactivity| reactivity.is_nan()));
    }
}
