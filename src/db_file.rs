pub mod native;

use std::{
    convert::TryInto,
    error::Error as StdError,
    ffi::OsString,
    fmt::{self, Display},
    io,
    path::Path,
    ptr,
    string::FromUtf8Error,
};

use serde::{Serialize, Serializer};

use crate::{Base, Molecule, Reactivity, SequenceEntry};

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

pub fn read_db(path: &Path) -> Result<Vec<Entry>, Error> {
    let extension = path.extension().ok_or(Error::NoExtension)?;
    if extension.eq_ignore_ascii_case("db") {
        native::read_file(path).map_err(Error::Native)
    } else {
        Err(Error::InvalidExtension(extension.to_os_string()))
    }
}

#[derive(Debug)]
pub enum Error {
    NoExtension,
    InvalidExtension(OsString),
    Native(native::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoExtension => f.write_str("db file does not have an extension"),
            Error::InvalidExtension(extension) => {
                write!(
                    f,
                    "extension \"{}\" is not valid for a db",
                    extension.to_string_lossy()
                )
            }
            Error::Native(_) => f.write_str("cannot read native db file"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::NoExtension | Error::InvalidExtension(_) => None,
            Error::Native(source) => Some(source),
        }
    }
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
    writer.write_all(native::VERSION.to_le_bytes().as_slice())?;
    writer.write_all(native::END_MARKER)?;
    writer.flush()?;

    Ok(())
}
