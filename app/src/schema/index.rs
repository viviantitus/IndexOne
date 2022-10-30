use std::ops::{Range, RangeFull, RangeFrom, RangeTo};


#[derive(Debug, Clone)]
pub enum Indexer {
    Range(Range<usize>),
    RangeFrom(RangeFrom<usize>),
    RangeTo(RangeTo<usize>),
    RangeFull(RangeFull),
    Number(usize)
}