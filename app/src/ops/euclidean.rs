use rand::distributions::uniform::SampleUniform;
use crate::schema::tensor::Tensor;

pub fn euclidean<'a, T: SampleUniform + PartialOrd + Copy>(t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<'a, T>{
    if t1.size() != t2.size(){
        panic!("Euclidean: Dimensions do not match");
    }

    Tensor::create_with_tensorsize(t1.size().clone())
}