use std::ops::Range;


#[derive(Debug, Clone)]
pub enum Indexer {
    SliceRange(Range<usize>),
    Number(usize)
}