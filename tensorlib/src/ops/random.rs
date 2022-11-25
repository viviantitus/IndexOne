use std::ops::Range;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Standard, Distribution};
use rand::{thread_rng, Rng};

use crate::schema::size::TensorSize;
use crate::schema::tensor::Tensor;



pub trait PartialOrdwithSampling: SampleUniform + PartialOrd + Copy{
    
}

impl<T> PartialOrdwithSampling for T where T: SampleUniform + PartialOrd + Copy {

}

pub trait Random<T: PartialOrdwithSampling>{
    type Output;
    fn create_random(size: Vec<usize>, range: Option<Range<T>>) -> Self::Output where Standard: Distribution<T>;
    fn create_random_with_tensorsize(size: TensorSize, range: Option<Range<T>>) -> Self::Output where Standard: Distribution<T>;
}

impl<T: PartialOrdwithSampling> Random<T> for Tensor<T> {
    type Output = Self;

    fn create_random(size: Vec<usize>, range: Option<Range<T>>) -> Self::Output   where Standard: Distribution<T>
    { 
        let tensor_size = TensorSize::new(size);
        Self::create_random_with_tensorsize(tensor_size, range)
    }

    fn create_random_with_tensorsize(size: TensorSize, range: Option<Range<T>>) -> Self::Output  where Standard: Distribution<T>
    {
        let mut tensor = Self::create_with_tensorsize(size);
        let mut rng = thread_rng();

        let total_size = tensor.data.capacity();
        match range{
            Some(x) => for _ in 0..total_size{
                tensor.data.push(rng.gen_range(x.clone()))
            },
            None => for _ in 0..total_size{
                tensor.data.push(rng.gen())
            }
        }
        tensor
    }
}