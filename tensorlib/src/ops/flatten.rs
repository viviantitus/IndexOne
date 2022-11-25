use crate::schema::{tensor::Tensor, size::TensorSize};


pub trait Flatten<'a, T>{
    fn flatten(self) -> Self;
}


impl<'a, T: Copy> Flatten<'a, T> for Tensor<T> {
    fn flatten(self) -> Self{
        let mut tensor = self.clone();
        tensor.size = TensorSize::new(vec![self.size.total_elements()]);
        tensor.dim = 1;
        tensor
    }
}


#[cfg(test)]
mod tests {
    use crate::ops::assign::Assign;

    use super::*;

    #[test]
    fn test_flatten() {
        let tensor = Tensor::<f32>::create_zeros(vec![5, 10]);

        let result = tensor.flatten();
        assert!(result.dim() == 1);
    }
}