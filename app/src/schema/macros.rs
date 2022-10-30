 

#[macro_export]
macro_rules! t {
    ($($x:expr),*) => {{
        use crate::schema::index::Indexer;
        use std::ops::{Range, RangeFull, RangeFrom, RangeTo};


        trait IndexEnumConverter {
            fn as_enum_variable(self)->Indexer;
        }
        impl IndexEnumConverter for RangeFrom<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::RangeFrom(self)
            }
        }
        impl IndexEnumConverter for RangeTo<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::RangeTo(self)
            }
        }
        impl IndexEnumConverter for RangeFull{
            fn as_enum_variable(self)->Indexer {
                Indexer::RangeFull(self)
            }
        }
        impl IndexEnumConverter for Range<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::Range(self)
            }
        }
        impl IndexEnumConverter for usize{
            fn as_enum_variable(self)->Indexer {
                Indexer::Number(self)
            }
        }
        vec![$(($x).as_enum_variable()),*] as Vec<Indexer>
        
    }};
}