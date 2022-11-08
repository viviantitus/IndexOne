 

#[macro_export]
macro_rules! t {
    ($($x:expr),*) => {{
        use crate::schema::index::Indexer;
        use crate::schema::tensor::Tensor;
        use std::ops::{Range, RangeFull, RangeFrom, RangeTo};


        trait IndexEnumConverter<'a> {
            fn as_enum_variable(self)->Indexer<'a>;
        }
        impl<'a> IndexEnumConverter<'a> for RangeFrom<usize>{
            fn as_enum_variable(self)->Indexer<'a> {
                Indexer::RangeFrom(self)
            }
        }
        impl<'a> IndexEnumConverter<'a> for RangeTo<usize>{
            fn as_enum_variable(self)->Indexer<'a> {
                Indexer::RangeTo(self)
            }
        }
        impl<'a> IndexEnumConverter<'a> for RangeFull{
            fn as_enum_variable(self)->Indexer<'a> {
                Indexer::RangeFull(self)
            }
        }
        impl<'a> IndexEnumConverter<'a> for Range<usize>{
            fn as_enum_variable(self)->Indexer<'a> {
                Indexer::Range(self)
            }
        }
        impl<'a> IndexEnumConverter<'a> for usize{
            fn as_enum_variable(self)->Indexer<'a> {
                Indexer::Number(self)
            }
        }
        impl<'a> IndexEnumConverter<'a> for Tensor<'a, bool>{
            fn as_enum_variable(self)->Indexer<'a> {
                Indexer::BoolArray(self)
            }
        }
        vec![$(($x).as_enum_variable()),*] as Vec<Indexer>
        
    }};
}