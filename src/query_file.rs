use std::{
    error::Error as StdError,
    fmt::{self, Display},
    fs::File,
    io::{self, BufRead, BufReader},
    ops::Not,
    path::Path,
    sync::Arc,
};

use crate::{db_file::ReactivityWithPlaceholder, Base, Molecule, Reactivity, SequenceEntry};

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: Arc<str>,
    sequence: Vec<Base>,
    reactivities: Vec<ReactivityWithPlaceholder>,
    pub(crate) molecule: Molecule,
}

impl Entry {
    #[cfg(test)]
    pub(crate) fn new_unchecked(
        name: impl Into<Arc<str>>,
        sequence: Vec<Base>,
        reactivities: Vec<ReactivityWithPlaceholder>,
        molecule: Molecule,
    ) -> Self {
        let name = name.into();
        Self {
            name,
            sequence,
            reactivities,
            molecule,
        }
    }

    pub fn cap_reactivities(&mut self, max_reactivity: Reactivity) {
        self.reactivities.iter_mut().for_each(|reactivity| {
            if let Some(x) = reactivity.get_non_nan() {
                *reactivity = x.min(max_reactivity).into();
            }
        });
    }
}

impl SequenceEntry for Entry {
    type Reactivity = ReactivityWithPlaceholder;

    fn name(&self) -> &str {
        &self.name
    }

    fn sequence(&self) -> &[Base] {
        &self.sequence
    }

    fn reactivity(&self) -> &[Self::Reactivity] {
        &self.reactivities
    }

    fn molecule(&self) -> Molecule {
        self.molecule
    }
}

#[derive(Debug)]
pub enum Error {
    TruncatedExpectedSequence,
    TruncatedExpectedReactivities,
    InvalidSequenceBase(Box<RowColumn>),
    InvalidReactivity(Box<RowColumn>),
    EmptySequence(usize),
    UnmatchedLengths(Box<UnmatchedLengths>),
    IO(Box<io::Error>),
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TruncatedExpectedSequence => f.write_str("file truncated, expected sequence"),
            Error::TruncatedExpectedReactivities => {
                f.write_str("file truncated, expected reactivities")
            }
            Error::InvalidSequenceBase(row_column) => {
                write!(
                    f,
                    "invalid sequence base at line {} and column {}",
                    row_column.row, row_column.column,
                )
            }
            Error::InvalidReactivity(row_column) => write!(
                f,
                "invalid reactivity at line {} and column {}",
                row_column.row, row_column.column,
            ),
            Error::EmptySequence(row) => write!(f, "unexpected empty sequence at line {row}"),
            Error::UnmatchedLengths(lengths) => {
                write!(
                    f,
                    "unmatching lengths between sequence ({}) and reactivities ({}) for query \
                    starting at line {}",
                    lengths.sequence, lengths.reactivities, lengths.line,
                )
            }
            Error::IO(_) => f.write_str("I/O error"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::TruncatedExpectedSequence
            | Error::TruncatedExpectedReactivities
            | Error::InvalidSequenceBase(_)
            | Error::InvalidReactivity(_)
            | Error::EmptySequence(_)
            | Error::UnmatchedLengths(_) => None,
            Error::IO(source) => Some(source),
        }
    }
}

impl From<Box<io::Error>> for Error {
    fn from(value: Box<io::Error>) -> Self {
        Self::IO(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowColumn {
    pub row: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnmatchedLengths {
    sequence: usize,
    reactivities: usize,
    line: usize,
}

#[inline]
pub fn read_file(path: &Path) -> Result<Vec<Entry>, Error> {
    let reader = BufReader::new(File::open(path).map_err(Box::new)?);
    read_file_content(reader)
}

pub fn read_file_content<R>(mut reader: R) -> Result<Vec<Entry>, Error>
where
    R: BufRead,
{
    let mut line = String::new();
    let mut entries = Vec::new();

    let mut file_row = 0;
    loop {
        line.clear();
        file_row += 1;
        if reader.read_line(&mut line).map_err(Box::new)? == 0 {
            break;
        }

        if line.as_bytes().iter().all(u8::is_ascii_whitespace) {
            continue;
        }

        let name = Arc::from(line.trim().to_string());

        file_row += 1;
        line.clear();
        if reader.read_line(&mut line).map_err(Box::new)? == 0 {
            return Err(Error::TruncatedExpectedSequence);
        }

        let (sequence, molecule) = parse_sequence(&line, file_row)?;

        if sequence.is_empty() {
            return Err(Error::EmptySequence(file_row));
        }

        file_row += 1;
        line.clear();
        if reader.read_line(&mut line).map_err(Box::new)? == 0 {
            return Err(Error::TruncatedExpectedReactivities);
        }

        let mut column = 1;
        let reactivities = line
            .trim_end()
            .split(',')
            .map(|raw_reactivity| {
                if column != 1 {
                    column += 1;
                }

                let reactivity = if raw_reactivity.eq_ignore_ascii_case("NaN") {
                    ReactivityWithPlaceholder::from(Reactivity::NAN)
                } else {
                    raw_reactivity
                        .parse::<Reactivity>()
                        .map(ReactivityWithPlaceholder::from)
                        .map_err(|_| {
                            Error::InvalidReactivity(Box::new(RowColumn {
                                row: file_row,
                                column,
                            }))
                        })?
                };

                column += raw_reactivity.len();
                Ok::<_, Error>(reactivity)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if sequence.len() != reactivities.len() {
            return Err(Error::UnmatchedLengths(Box::new(UnmatchedLengths {
                sequence: sequence.len(),
                reactivities: reactivities.len(),
                line: file_row - 2,
            })));
        }

        entries.push(Entry {
            name,
            sequence,
            reactivities,
            molecule,
        });
    }

    Ok(entries)
}

fn parse_sequence(raw_line: &str, row: usize) -> Result<(Vec<Base>, Molecule), Error> {
    let mut molecule = Molecule::default();
    raw_line
        .as_bytes()
        .iter()
        .copied()
        .enumerate()
        .skip_while(|(_, c)| c.is_ascii_whitespace())
        .take_while(|(_, c)| c.is_ascii_whitespace().not())
        .map(|(index, c)| {
            match (c, molecule) {
                (b'T', Molecule::Unknown) => molecule = Molecule::Dna,
                (b'U', Molecule::Unknown) => molecule = Molecule::Rna,
                (b'T', Molecule::Rna) | (b'U', Molecule::Dna) => {
                    return Err(Error::InvalidSequenceBase(Box::new(RowColumn {
                        row,
                        column: index + 1,
                    })));
                }
                _ => {}
            }

            Base::try_from(c).map_err(|_| {
                Error::InvalidSequenceBase(Box::new(RowColumn {
                    row,
                    column: index + 1,
                }))
            })
        })
        .collect::<Result<_, _>>()
        .map(|sequence| (sequence, molecule))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    macro_rules! base {
        (A) => {
            crate::Base::A
        };

        (C) => {
            crate::Base::C
        };

        (G) => {
            crate::Base::G
        };

        (T) => {
            crate::Base::T
        };

        (N) => {
            crate::Base::N
        };
    }

    macro_rules! seq {
        ([$($bases:expr),*] $(,)?) => {
            &[$($bases),*]
        };

        ([$($bases:expr),* $(,)?] $base:ident $($rest:ident)*) => {
            seq!([$($bases,)* base!($base)] $($rest)*)
        };

        ($($bases:ident)*) => {
            seq!([] $($bases)*)
        };
    }

    fn reactivities_eq<I1, I2>(a: I1, b: I2) -> bool
    where
        I1: IntoIterator<Item = ReactivityWithPlaceholder>,
        I2: IntoIterator<Item = Reactivity>,
    {
        a.into_iter().zip(b).all(|(a, b)| {
            if b.is_nan() {
                a.is_nan()
            } else {
                (a.to_maybe_placeholder() - b).abs() < 10e-5
            }
        })
    }

    #[test]
    fn read_valid_file() {
        const CONTENT: &str = include_str!("../test_data/valid_query.txt");
        let entries = read_file_content(Cursor::new(CONTENT)).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(&*entries[0].name, "test1");
        assert_eq!(entries[0].sequence, seq!(A C G T N));
        assert!(reactivities_eq(
            entries[0].reactivities.iter().copied(),
            [0.123, 0.456, 0.789, 1.234, Reactivity::NAN]
        ));

        assert_eq!(&*entries[1].name, "test2");
        assert_eq!(entries[1].sequence, seq!(N A C G T));
        assert!(reactivities_eq(
            entries[1].reactivities.iter().copied(),
            [Reactivity::NAN, 12., 0.456, 0.789, 0.012]
        ));
    }

    #[test]
    fn empty_sequence() {
        const CONTENT: &str = include_str!("../test_data/query_empty_sequence.txt");
        let err = read_file_content(Cursor::new(CONTENT)).unwrap_err();

        assert!(matches!(err, Error::EmptySequence(6)));
    }

    #[test]
    fn truncated_sequence() {
        const CONTENT: &str = include_str!("../test_data/query_truncated_sequence.txt");
        let err = read_file_content(Cursor::new(CONTENT)).unwrap_err();

        assert!(matches!(err, Error::TruncatedExpectedSequence));
    }

    #[test]
    fn truncated_reactivities() {
        const CONTENT: &str = include_str!("../test_data/query_truncated_reactivities.txt");
        let err = read_file_content(Cursor::new(CONTENT)).unwrap_err();

        assert!(matches!(err, Error::TruncatedExpectedReactivities));
    }

    #[test]
    fn invalid_sequence_base() {
        const CONTENT: &str = include_str!("../test_data/query_invalid_base.txt");
        let err = read_file_content(Cursor::new(CONTENT)).unwrap_err();

        if let Error::InvalidSequenceBase(err) = err {
            assert_eq!(*err, RowColumn { row: 6, column: 3 });
        } else {
            panic!()
        }
    }

    #[test]
    fn invalid_sequence_reactivity() {
        const CONTENT: &str = include_str!("../test_data/query_invalid_reactivity.txt");
        let err = read_file_content(Cursor::new(CONTENT)).unwrap_err();

        if let Error::InvalidReactivity(err) = err {
            assert_eq!(*err, RowColumn { row: 7, column: 11 });
        } else {
            panic!()
        }
    }

    #[test]
    fn invalid_lengths() {
        const CONTENT: &str = include_str!("../test_data/query_invalid_lengths.txt");
        let err = read_file_content(Cursor::new(CONTENT)).unwrap_err();

        if let Error::UnmatchedLengths(err) = err {
            assert_eq!(
                *err,
                UnmatchedLengths {
                    sequence: 6,
                    reactivities: 5,
                    line: 5,
                }
            );
        } else {
            panic!()
        }
    }

    #[test]
    fn cap_reactivities() {
        const CONTENT: &str = include_str!("../test_data/valid_query.txt");
        let mut entries = read_file_content(Cursor::new(CONTENT)).unwrap();
        entries
            .iter_mut()
            .for_each(|entry| entry.cap_reactivities(1.));

        dbg!(&entries);
        assert!(reactivities_eq(
            entries[0].reactivities.iter().copied(),
            [0.123, 0.456, 0.789, 1., Reactivity::NAN]
        ));

        assert!(reactivities_eq(
            entries[1].reactivities.iter().copied(),
            [Reactivity::NAN, 1., 0.456, 0.789, 0.012]
        ));
    }
}
