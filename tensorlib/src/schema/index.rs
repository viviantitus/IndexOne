use std::ops::{Range, RangeFull, RangeFrom, RangeTo};

use super::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum Indexer {
    Range(Range<usize>),
    RangeFrom(RangeFrom<usize>),
    RangeTo(RangeTo<usize>),
    RangeFull(RangeFull),
    Number(usize),
    BoolArray(Tensor<bool>)
}