 

#[macro_export]
macro_rules! t {
    ($($x:expr),*) => {{
        use crate::tensor_ops::index::Indexer;
        use std::ops::Range;

        trait IndexEnumConverter {
            fn as_enum_variable(self)->Indexer;
        }
        impl IndexEnumConverter for Range<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::SliceRange(self)
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