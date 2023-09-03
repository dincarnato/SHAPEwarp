use std::{
    cmp::Ordering,
    fmt::{self, Display},
    ops::{Not, Range},
    str::FromStr,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DotBracket<C, const SORTED: bool> {
    paired_blocks: C,
    len: usize,
}

impl<C> DotBracket<C, false>
where
    C: AsMut<Vec<PairedBlock>>,
{
    #[inline]
    pub fn from_str(dot_bracket: &str, paired_blocks_buffer: C) -> Result<Self, InvalidDotBracket> {
        Self::from_str_with_buffer(dot_bracket, paired_blocks_buffer, &mut Vec::new())
    }

    #[inline]
    pub fn from_str_with_buffer(
        dot_bracket: &str,
        paired_blocks_buffer: C,
        working_buffer: &mut Vec<PartialPairedBlock>,
    ) -> Result<Self, InvalidDotBracket> {
        Self::from_bytes_with_buffer(dot_bracket.as_bytes(), paired_blocks_buffer, working_buffer)
    }

    pub fn from_bytes_with_buffer(
        dot_bracket: &[u8],
        mut paired_blocks_buffer: C,
        working_buffer: &mut Vec<PartialPairedBlock>,
    ) -> Result<Self, InvalidDotBracket> {
        let len = dot_bracket.len();

        let paired_blocks_buffer_ref = paired_blocks_buffer.as_mut();
        paired_blocks_buffer_ref.clear();
        working_buffer.clear();
        let state = dot_bracket
            .iter()
            .enumerate()
            .try_fold(None, |partial, (index, &c)| {
                try_fold_from_bytes(partial, index, c, paired_blocks_buffer_ref, working_buffer)
            })?;

        if working_buffer.is_empty().not() {
            return Err(InvalidDotBracket);
        }

        if let Some(state) = state {
            let PartialPairedBlockUnstored {
                left_start,
                other:
                    Some(PartialPairedBlockOther {
                        left_end,
                        right_start,
                    }),
            } = state
            else {
                return Err(InvalidDotBracket);
            };

            let left = left_start..left_end;
            let right = right_start..dot_bracket.len();
            if left.len() != right.len() {
                return Err(InvalidDotBracket);
            }

            paired_blocks_buffer_ref.push(PairedBlock { left, right });
        }

        Ok(DotBracket {
            paired_blocks: paired_blocks_buffer,
            len,
        })
    }

    #[inline]
    pub fn into_sorted(self) -> DotBracket<C, true> {
        let Self {
            mut paired_blocks,
            len,
        } = self;
        paired_blocks
            .as_mut()
            .sort_unstable_by_key(|block| block.left.start);

        DotBracket { paired_blocks, len }
    }
}

fn try_fold_from_bytes(
    partial: Option<PartialPairedBlockUnstored>,
    index: usize,
    c: u8,
    paired_blocks_buffer: &mut Vec<PairedBlock>,
    working_buffer: &mut Vec<PartialPairedBlock>,
) -> Result<Option<PartialPairedBlockUnstored>, InvalidDotBracket> {
    match (c, partial) {
        (b'(', None) => Ok(Some(PartialPairedBlockUnstored {
            left_start: index,
            other: None,
        })),

        (b'(', partial @ Some(_)) | (b'.', partial @ None) => Ok(partial),

        (b'.', Some(PartialPairedBlockUnstored { left_start, other })) => {
            match other {
                Some(PartialPairedBlockOther {
                    left_end,
                    right_start,
                }) => paired_blocks_buffer.push(handle_lr_paired_block(
                    index,
                    right_start,
                    left_start,
                    left_end,
                    working_buffer,
                )),
                None => working_buffer.push(PartialPairedBlock {
                    left: left_start..index,
                }),
            }

            Ok(None)
        }

        (b')', None) => {
            let PartialPairedBlock { left } = working_buffer.pop().ok_or(InvalidDotBracket)?;
            Ok(Some(PartialPairedBlockUnstored {
                left_start: left.start,
                other: Some(PartialPairedBlockOther {
                    left_end: left.end,
                    right_start: index,
                }),
            }))
        }

        (
            b')',
            Some(PartialPairedBlockUnstored {
                left_start,
                other:
                    Some(PartialPairedBlockOther {
                        left_end,
                        right_start,
                    }),
            }),
        ) if left_end - left_start > index + 1 - right_start => {
            Ok(Some(PartialPairedBlockUnstored {
                left_start,
                other: Some(PartialPairedBlockOther {
                    left_end,
                    right_start,
                }),
            }))
        }

        (
            b')',
            Some(PartialPairedBlockUnstored {
                left_start,
                other:
                    Some(PartialPairedBlockOther {
                        left_end,
                        right_start,
                    }),
            }),
        ) if left_end - left_start == index + 1 - right_start => {
            let left = left_start..left_end;
            // Cannot use InclusiveRange here
            #[allow(clippy::range_plus_one)]
            let right = right_start..(index + 1);
            paired_blocks_buffer.push(PairedBlock { left, right });

            Ok(None)
        }

        (
            b')',
            Some(PartialPairedBlockUnstored {
                left_start,
                other: None,
            }),
        ) => Ok(Some(PartialPairedBlockUnstored {
            left_start,
            other: Some(PartialPairedBlockOther {
                left_end: index,
                right_start: index,
            }),
        })),

        (
            b')',
            Some(PartialPairedBlockUnstored {
                left_start,
                other:
                    Some(PartialPairedBlockOther {
                        left_end,
                        right_start,
                    }),
            }),
        ) if left_end - left_start < index + 1 - right_start => {
            panic!("invalid partial paired blocks status")
        }

        _ => Err(InvalidDotBracket),
    }
}

fn handle_lr_paired_block(
    index: usize,
    right_start: usize,
    left_start: usize,
    left_end: usize,
    working_buffer: &mut Vec<PartialPairedBlock>,
) -> PairedBlock {
    let right = right_start..index;

    let left_len = left_end - left_start;
    let right_len = index - right_start;
    let left = match left_len.cmp(&right_len) {
        Ordering::Greater => {
            let new_left_start = left_end - right_len;
            working_buffer.push(PartialPairedBlock {
                left: left_start..new_left_start,
            });

            new_left_start..left_end
        }
        Ordering::Equal => left_start..left_end,
        Ordering::Less => unreachable!("invalid paired blocks"),
    };

    PairedBlock { left, right }
}

impl<C, const SORTED: bool> DotBracket<C, SORTED>
where
    C: AsRef<[PairedBlock]>,
{
    #[inline]
    pub fn paired_blocks(&self) -> &[PairedBlock] {
        self.paired_blocks.as_ref()
    }

    pub fn to_owned(&self) -> DotBracket<Vec<PairedBlock>, SORTED> {
        let &Self {
            ref paired_blocks,
            len,
        } = self;
        let paired_blocks = paired_blocks.as_ref().to_owned();
        DotBracket { paired_blocks, len }
    }
}

impl<C, const SORTED: bool> Display for DotBracket<C, SORTED>
where
    C: AsRef<[PairedBlock]>,
{
    // TODO: find a better implementation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self {
            ref paired_blocks,
            len,
        } = self;

        let mut buf = vec![b'.'; len];
        for block in paired_blocks.as_ref() {
            buf[block.left().clone()].fill(b'(');
            buf[block.right().clone()].fill(b')');
        }

        f.write_str(std::str::from_utf8(&buf).unwrap())
    }
}

pub type DotBracketOwned = DotBracket<Vec<PairedBlock>, false>;
pub type DotBracketOwnedSorted = DotBracket<Vec<PairedBlock>, true>;

pub type DotBracketBuffered<'a> = DotBracket<&'a mut Vec<PairedBlock>, false>;

impl FromStr for DotBracketOwned {
    type Err = InvalidDotBracket;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DotBracket::from_str(s, Vec::new())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InvalidDotBracket;

impl Display for InvalidDotBracket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid dot-bracket notation string")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PairedBlock {
    left: Range<usize>,
    right: Range<usize>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct PartialPairedBlock {
    left: Range<usize>,
}

#[derive(Debug)]
struct PartialPairedBlockUnstored {
    left_start: usize,
    other: Option<PartialPairedBlockOther>,
}

#[derive(Debug)]
struct PartialPairedBlockOther {
    left_end: usize,
    right_start: usize,
}

impl PairedBlock {
    #[inline]
    pub fn left(&self) -> &Range<usize> {
        &self.left
    }

    #[inline]
    pub fn right(&self) -> &Range<usize> {
        &self.right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEM_LOOP_DB: &str = "...(((((((((....)))))))))";
    fn test_stem_loop<C>(db: &DotBracket<C, false>)
    where
        C: AsRef<[PairedBlock]> + fmt::Debug,
    {
        assert_eq!(db.len, 25);
        assert_eq!(
            db.paired_blocks.as_ref(),
            [PairedBlock {
                left: 3..12,
                right: 16..25,
            }],
        );
    }

    #[test]
    fn simple_stem_loop_owned() {
        let db: DotBracketOwned = STEM_LOOP_DB.parse().unwrap();
        test_stem_loop(&db);
    }

    #[test]
    fn simple_stem_loop_buffered() {
        let mut buffer = vec![];
        let db = DotBracketBuffered::from_str(STEM_LOOP_DB, &mut buffer).unwrap();
        test_stem_loop(&db);
    }

    #[test]
    fn multiple_stem_loop() {
        let db: DotBracketOwned = "...((((....))))..(((....))).....((....)).."
            .parse()
            .unwrap();
        assert_eq!(db.len, 42);
        assert_eq!(
            db.paired_blocks,
            [
                PairedBlock {
                    left: 3..7,
                    right: 11..15,
                },
                PairedBlock {
                    left: 17..20,
                    right: 24..27,
                },
                PairedBlock {
                    left: 32..34,
                    right: 38..40,
                },
            ],
        );
    }

    #[test]
    fn tight_loop() {
        let db: DotBracketOwned = "(((())))".parse().unwrap();
        assert_eq!(db.len, 8);
        assert_eq!(
            db.paired_blocks,
            [PairedBlock {
                left: 0..4,
                right: 4..8,
            }],
        );
    }

    #[test]
    fn nested_stem_loop_left() {
        let db: DotBracketOwned = "((((.(((((...)))..))...))))..".parse().unwrap();
        assert_eq!(db.len, 29);
        assert_eq!(
            db.paired_blocks,
            [
                PairedBlock {
                    left: 7..10,
                    right: 13..16,
                },
                PairedBlock {
                    left: 5..7,
                    right: 18..20,
                },
                PairedBlock {
                    left: 0..4,
                    right: 23..27,
                },
            ],
        );

        assert_eq!(
            db.into_sorted(),
            DotBracket::<_, true> {
                len: 29,
                paired_blocks: vec![
                    PairedBlock {
                        left: 0..4,
                        right: 23..27,
                    },
                    PairedBlock {
                        left: 5..7,
                        right: 18..20,
                    },
                    PairedBlock {
                        left: 7..10,
                        right: 13..16,
                    },
                ],
            }
        );
    }

    #[test]
    fn nested_stem_loop_right() {
        let db: DotBracketOwned = "((((.((..(((...)))))...))))..".parse().unwrap();
        assert_eq!(db.len, 29);
        assert_eq!(
            db.paired_blocks,
            [
                PairedBlock {
                    left: 9..12,
                    right: 15..18,
                },
                PairedBlock {
                    left: 5..7,
                    right: 18..20,
                },
                PairedBlock {
                    left: 0..4,
                    right: 23..27,
                },
            ],
        );

        assert_eq!(
            db.into_sorted(),
            DotBracket::<_, true> {
                len: 29,
                paired_blocks: vec![
                    PairedBlock {
                        left: 0..4,
                        right: 23..27,
                    },
                    PairedBlock {
                        left: 5..7,
                        right: 18..20,
                    },
                    PairedBlock {
                        left: 9..12,
                        right: 15..18,
                    },
                ],
            },
        );
    }

    #[test]
    fn ending_with_state() {
        let db: DotBracketOwned = "(.((..)))".parse().unwrap();
        assert_eq!(db.len, 9);
        assert_eq!(
            db.paired_blocks,
            [
                PairedBlock {
                    left: 2..4,
                    right: 6..8,
                },
                PairedBlock {
                    left: 0..1,
                    right: 8..9,
                },
            ],
        );

        assert_eq!(
            db.into_sorted(),
            DotBracket::<_, true> {
                len: 9,
                paired_blocks: vec![
                    PairedBlock {
                        left: 0..1,
                        right: 8..9,
                    },
                    PairedBlock {
                        left: 2..4,
                        right: 6..8,
                    },
                ],
            },
        );
    }
}
