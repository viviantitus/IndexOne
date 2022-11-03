use crate::schema::tensor::Tensor;


pub trait Equals<'a, T: PartialEq>{
    type Output;
    fn equals(&self, other: T) -> Self::Output;
}


impl<'a, T: PartialEq> Equals<'a, T> for Tensor<'a, T> {
    type Output = Tensor<'a, bool>;
    fn equals(&self, other: T) -> Self::Output{
        let tensor = Tensor::<bool>::new(vec![self.size.total_elements()]);
        for i in 0..tensor.size.total_elements(){
            tensor.data[i] = self.data[i] == other;
        }
        tensor
    }
}