use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    ops::Not,
    path::Path,
};

use crate::{Base, Reactivity};

#[derive(Debug)]
pub struct Entry {
    pub name: String,
    sequence: Vec<Base>,
    reactivities: Vec<Reactivity>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("file truncated, expected sequence")]
    TruncatedExpectedSequence,

    #[error("file truncated, expected reactivities")]
    TruncatedExpectedReactivities,

    #[error("invalid sequence base at line {} and column {}", .0.row, .0.column)]
    InvalidSequenceBase(Box<RowColumn>),

    #[error("invalid reactivity at line {} and column {}", .0.row, .0.column)]
    InvalidReactivity(Box<RowColumn>),

    #[error("unexpected empty sequence at line {0}")]
    EmptySequence(usize),

    #[error(
        "unmatching lengths between sequence ({}) and reactivities ({}) for query starting at line {}",
        .0.sequence,
        .0.reactivities,
        .0.line
    )]
    UnmatchedLengths(Box<UnmatchedLengths>),

    #[error("I/O error: {0}")]
    IO(#[from] Box<io::Error>),
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
pub fn read_file(path: &Path, max_reactivity: Reactivity) -> Result<Vec<Entry>, Error> {
    let reader = BufReader::new(File::open(path).map_err(Box::new)?);
    read_file_content(reader, max_reactivity)
}

fn read_file_content<R>(mut reader: R, max_reactivity: Reactivity) -> Result<Vec<Entry>, Error>
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

        if line.as_bytes().iter().all(|c| c.is_ascii_whitespace()) {
            continue;
        }

        let name = line.trim().to_string();

        file_row += 1;
        line.clear();
        if reader.read_line(&mut line).map_err(Box::new)? == 0 {
            return Err(Error::TruncatedExpectedSequence);
        }

        let sequence = line
            .as_bytes()
            .iter()
            .copied()
            .enumerate()
            .skip_while(|(_, c)| c.is_ascii_whitespace())
            .take_while(|(_, c)| c.is_ascii_whitespace().not())
            .map(|(index, c)| {
                Base::try_from(c).map_err(|_| {
                    Error::InvalidSequenceBase(Box::new(RowColumn {
                        row: file_row,
                        column: index + 1,
                    }))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

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
                    Reactivity::NAN
                } else {
                    raw_reactivity
                        .parse::<Reactivity>()
                        .map_err(|_| {
                            Error::InvalidReactivity(Box::new(RowColumn {
                                row: file_row,
                                column,
                            }))
                        })?
                        .min(max_reactivity)
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
        })
    }

    Ok(entries)
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
        I1: IntoIterator<Item = Reactivity>,
        I2: IntoIterator<Item = Reactivity>,
    {
        a.into_iter().zip(b).all(|(a, b)| {
            if b.is_nan() {
                a.is_nan()
            } else {
                (a - b).abs() < 10e-5
            }
        })
    }

    #[test]
    fn read_valid_file() {
        const CONTENT: &str = include_str!("../test_data/valid_query.txt");
        let entries = read_file_content(Cursor::new(CONTENT), 1.).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "test1");
        assert_eq!(entries[0].sequence, seq!(A C G T N));
        assert!(reactivities_eq(
            entries[0].reactivities.iter().copied(),
            [0.123, 0.456, 0.789, 1., Reactivity::NAN]
        ));

        assert_eq!(entries[1].name, "test2");
        assert_eq!(entries[1].sequence, seq!(N A C G T));
        assert!(reactivities_eq(
            entries[1].reactivities.iter().copied(),
            [Reactivity::NAN, 1., 0.456, 0.789, 0.012]
        ));
    }

    #[test]
    fn empty_sequence() {
        const CONTENT: &str = include_str!("../test_data/query_empty_sequence.txt");
        let err = read_file_content(Cursor::new(CONTENT), 1.).unwrap_err();

        assert!(matches!(err, Error::EmptySequence(6)));
    }

    #[test]
    fn truncated_sequence() {
        const CONTENT: &str = include_str!("../test_data/query_truncated_sequence.txt");
        let err = read_file_content(Cursor::new(CONTENT), 1.).unwrap_err();

        assert!(matches!(err, Error::TruncatedExpectedSequence));
    }

    #[test]
    fn truncated_reactivities() {
        const CONTENT: &str = include_str!("../test_data/query_truncated_reactivities.txt");
        let err = read_file_content(Cursor::new(CONTENT), 1.).unwrap_err();

        assert!(matches!(err, Error::TruncatedExpectedReactivities));
    }

    #[test]
    fn invalid_sequence_base() {
        const CONTENT: &str = include_str!("../test_data/query_invalid_base.txt");
        let err = read_file_content(Cursor::new(CONTENT), 1.).unwrap_err();

        match err {
            Error::InvalidSequenceBase(err) => assert_eq!(*err, RowColumn { row: 6, column: 3 }),
            _ => panic!(),
        }
    }

    #[test]
    fn invalid_sequence_reactivity() {
        const CONTENT: &str = include_str!("../test_data/query_invalid_reactivity.txt");
        let err = read_file_content(Cursor::new(CONTENT), 1.).unwrap_err();

        match err {
            Error::InvalidReactivity(err) => assert_eq!(*err, RowColumn { row: 7, column: 11 }),
            _ => panic!(),
        }
    }

    #[test]
    fn invalid_lengths() {
        const CONTENT: &str = include_str!("../test_data/query_invalid_lengths.txt");
        let err = read_file_content(Cursor::new(CONTENT), 1.).unwrap_err();

        match err {
            Error::UnmatchedLengths(err) => assert_eq!(
                *err,
                UnmatchedLengths {
                    sequence: 6,
                    reactivities: 5,
                    line: 5,
                }
            ),
            _ => panic!(),
        }
    }
}
