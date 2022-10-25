mod tensor_ops;
use crate::tensor_ops::tensor::{Tensor, Indexer};
use std::ops::Range;

macro_rules! tindex {
    ($($x:expr),*) => {{
        trait IndexEnumConverter {
            fn as_enum_variable(self)->Indexer;
        }
        impl IndexEnumConverter for Range<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::range(self)
            }
        }
        impl IndexEnumConverter for usize{
            fn as_enum_variable(self)->Indexer {
                Indexer::number(self)
            }
        }
        
            // println!("{:?}", ($x).as_enum_variable());
            vec![$(($x).as_enum_variable()),*] as Vec<Indexer>
        
    }};
}

fn main() {
    let tensor1 = Tensor::<f32>::new(vec![1000, 512], true, None);
    let tensor2 = Tensor::<f32>::new(vec![1000, 512], true, None);

    let _ = tensor1.euclidean_distance(tensor2);

    // TODO: convert this into macro
    let a: Vec<Indexer> = tindex![1, 1..2, 2, 5, 3..8];
    println!("{:?}", a);

}