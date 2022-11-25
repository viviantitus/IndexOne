use crate::{schema::tensor::Tensor};

pub trait Concat{
    type Output;
    fn concat(self) -> Tensor<Self::Output>;
}


macro_rules! concat_impl {
    ($($t:ty)*) => ($(
        impl Concat for Vec<Tensor<$t>> {
            type Output = $t;

            fn concat(self) -> Tensor<$t>{
                assert!(self.len() >= 1);

                let mut compare_len = self[0].size.clone();
                for i in 1..self.len(){
                    assert!(compare_len == self[i].size);
                }

                let first_dim = compare_len[0] * self.len();
                let new_size = compare_len.remove_dim(0).push_front(first_dim);
                let mut new_data = vec![];
                
                for i in 0..self.len(){
                    new_data.extend_from_slice(&self[i].data);
                }
                
                Tensor::create_with_data_copy(new_data, new_size)
            }
        }
    )*)
}   

concat_impl! { i32 i64 f32 f64 usize bool}


// TODO: test and place it in main.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schema::size::TensorSize, ops::assign::Assign};

    #[test]
    fn test_concat() {
        let mut vec = vec![];
        vec.push(Tensor::<f32>::create_zeros(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::create_zeros(vec![500, 30, 10]));

        let new_tensor = vec.concat();

        assert!(new_tensor.size() == &TensorSize::new(vec![1000, 30, 10]));
    }
}