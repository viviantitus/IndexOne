use rand::distributions::uniform::SampleUniform;
use crate::schema::tensor::Tensor;
use crate::ops::subtract::Subtract;
use crate::openblas_wrapper::norm2::Norm2;


impl<'a, T: SampleUniform + PartialOrd + Copy + std::ops::Sub<Output=T> + Norm2<T>> Tensor<'a, T>{

    pub fn euclidean(&mut self, other: &'a mut Tensor<T>) -> Tensor<'a, T>{
        if self.size() != other.size(){
            panic!("Euclidean: Dimensions do not match");
        }
        let new_tensor = self.sub(other);
        // new_tensor.norm2()
        new_tensor
    }
}