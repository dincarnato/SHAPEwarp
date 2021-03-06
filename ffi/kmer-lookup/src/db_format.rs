use std::{
    convert::TryInto,
    ffi::CString,
    io::{self, Read, Seek, SeekFrom},
};

use itertools::Itertools;

const END_SIZE: u8 = 17;
const END_MARKER: &[u8] = b"[eofdb]";

#[derive(Debug)]
pub struct Reader<R> {
    reader: R,
    db_len: u64,
    version: u16,
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

        let db_len = u64::from_le_bytes(end_buf[0..8].try_into().unwrap());
        let version = u16::from_le_bytes(end_buf[8..10].try_into().unwrap());
        Ok(Self {
            reader,
            db_len,
            version,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Base {
    A,
    C,
    G,
    T,
    N,
}

impl Base {
    pub fn try_from_nibble(nibble: u8) -> Result<Self, InvalidEncodedBase> {
        use Base::*;
        Ok(match nibble {
            0 => A,
            1 => C,
            2 => G,
            3 => T,
            4 => N,
            _ => return Err(InvalidEncodedBase),
        })
    }

    pub fn try_pair_from_byte(byte: u8) -> Result<[Self; 2], InvalidEncodedBase> {
        let first = Base::try_from_nibble(byte >> 4)?;
        let second = Base::try_from_nibble(byte & 0x0F)?;

        Ok([first, second])
    }

    pub fn to_byte(self) -> u8 {
        match self {
            Self::A => b'A',
            Self::C => b'C',
            Self::G => b'G',
            Self::T => b'T',
            Self::N => b'N',
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InvalidEncodedBase;

#[derive(Debug)]
pub struct Entry {
    pub id: CString,
    pub sequence: Vec<u8>,
    pub reactivity: Vec<f64>,
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
            ok!(CString::new(sequence_id).map_err(|_| { EntryError::InvalidSequenceId }));
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
                        .map(|[first, second]| [first.to_byte(), second.to_byte()])
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
            .map_ok(f64::from_le_bytes)
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
    #[error("Invalid sequence ID, incoherent length")]
    InvalidSequenceIdLength,

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
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    const TEST_DB: &[u8] = include_bytes!("../data/test.db");

    #[test]
    fn valid_reader() {
        let reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        assert_eq!(reader.db_len, 0x1181);
        assert_eq!(reader.version, 1);
    }

    #[test]
    fn read_all_db() {
        let mut reader = Reader::new(Cursor::new(TEST_DB)).unwrap();
        let db_len = reader
            .entries()
            .map_ok(|entry| entry.sequence.len())
            .try_fold(0, |acc, seq_len| seq_len.map(|seq_len| acc + seq_len))
            .unwrap();
        assert_eq!(db_len, reader.db_len.try_into().unwrap());
    }
}
