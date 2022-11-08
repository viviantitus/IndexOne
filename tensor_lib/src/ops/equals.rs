use crate::schema::tensor::Tensor;


pub trait Equals<T: PartialEq>{
    type Output;
    fn equals(&self, other: T) -> Self::Output;
}


impl<'a, T: PartialEq> Equals<T> for Tensor<'a, T> {
    type Output = Tensor<'a, bool>;
    fn equals(&self, other: T) -> Self::Output{
        let tensor = Tensor::<bool>::create_with_tensorsize(self.size.clone());
        for i in 0..tensor.size.total_elements(){
            tensor.data[i] = self.data[i] == other;
        }
        tensor
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schema::size::TensorSize};

    #[test]
    fn test_equals() {
        let mut data = [10.0, 3.0, 2.0, 10.0, 3.0];
        let data_len = data.len();
        let tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![data_len]));

        let result = tensor.equals(3.0);
        assert!(!result[0] && result[1])
    }
}