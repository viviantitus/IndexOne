use crate::{schema::{tensor::Tensor, size::TensorSize}};

pub trait Convert<'a>{
    type Output;
    fn convert_to_tensor(&'a mut self) -> Tensor<'a, Self::Output>;
}

macro_rules! convert_impl {
    ($($t:ty)*) => ($(
        impl<'a> Convert<'a> for Vec<$t> {
            type Output=$t;
            fn convert_to_tensor(&'a mut self) -> Tensor<'a, Self::Output>{
                let len = self.len();
                Tensor::create_with_data_copy(self.as_mut_slice(), TensorSize::new(vec![len]))
            }
        }
    )*)
}

convert_impl! { f32 f64 }




#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schema::size::TensorSize};

    #[test]
    fn test_convert() {
        let mut vec = vec![];
        vec.push(10.0);
        vec.push(20.0);

        let new_tensor = vec.convert_to_tensor();

        assert!(new_tensor[0] == 10.0 && new_tensor[1] == 20.0 && new_tensor.size == TensorSize::new(vec![2]));
    }

}