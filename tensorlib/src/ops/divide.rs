use crate::schema::tensor::Tensor;

pub trait Divide{
    type Output;
    fn div(self, rhs: Self::Output) -> Self;
}


macro_rules! div_impl {
    ($($t:ty)*) => ($(
        impl Divide for Tensor<'_, $t> {
            type Output = $t;

            fn div(self, rhs: Self::Output) -> Self {
                if rhs == 0.0 {
                    panic!("Cannot divide by zero-valued `Tensor`!");
                }

                for i in 0..self.data.len(){
                    self.data[i] /= rhs;
                }
                self
            }
        }
    )*)
}

div_impl! { f32 f64 }


#[cfg(test)]
mod divide_tests {
    use crate::schema::size::TensorSize;

    use super::*;

    #[test]
    fn snorm() {
        let mut data = [10.0, 15.0, 10.0, 10.0, 10.0];
        let data_len = data.len();
        let tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![data_len]));

        let result = tensor.div(5.0);
        assert!(result[0]==2.0 && result[1] == 3.0)
    }
}