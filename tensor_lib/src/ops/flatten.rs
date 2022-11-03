use crate::schema::{tensor::Tensor, size::TensorSize};


pub trait Flatten<'a, T>{
    fn flatten(&'a mut self) -> Self;
}


impl<'a, T> Flatten<'a, T> for Tensor<'a, T> {
    fn flatten(&'a mut self) -> Self{
        let tensor = Tensor::create_with_data_copy(self.data, TensorSize::new(vec![self.size.total_elements()]));
        tensor
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schema::{size::TensorSize, tensor}};

    #[test]
    fn test_flatten() {
        let mut tensor = Tensor::<f32>::new(vec![5, 10]);

        let result = tensor.flatten();
        assert!(result.dim() == 1);
    }
}