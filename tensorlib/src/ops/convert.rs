use crate::{schema::tensor::Tensor};
use crate::schema::size::TensorSize;
use crate::ops::assign::Assign;

pub trait Convert{
    type Output;
    fn convert_to_tensor(self) -> Tensor<Self::Output>;
}


macro_rules! convert_impl {
    ($($t:ty)*) => ($(
        impl Convert for Vec<Tensor<$t>> {
            type Output = $t;

            fn convert_to_tensor(self) -> Tensor<$t>{
                assert!(self.len() >= 1);

                let mut compare_len = self[0].size.clone();
                for i in 1..self.len(){
                    assert!(compare_len == self[i].size);
                }

                let new_size = compare_len.push_front(self.len());
                let mut new_data = vec![];
                
                for i in 0..self.len(){
                    new_data.extend_from_slice(&self[i].data);
                }
                
                Tensor::create_with_data_copy(new_data, new_size)
            }
        }

        impl Convert for Vec<$t> {
            type Output = $t;

            fn convert_to_tensor(self) -> Tensor<$t>{
                assert!(self.len() >= 1);
                
                let len = self.len();
                Tensor::create_with_data_copy(self, TensorSize::new(vec![len]))
            }
        }

        impl Convert for $t {
            type Output = $t;

            fn convert_to_tensor(self) -> Tensor<$t>{
                
                let mut tensor: Tensor<$t> = Tensor::<$t>::create_zeros(vec![1]);
                tensor.data[0]= self;
                tensor
            }
        }
    )*)
}

convert_impl! { u8 i32 i64 f32 f64 usize bool}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schema::size::TensorSize, ops::assign::Assign};

    #[test]
    fn test_convert() {
        let mut vec = vec![];
        vec.push(Tensor::<f32>::create_zeros(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::create_zeros(vec![500, 30, 10]));

        let new_tensor = vec.convert_to_tensor();

        assert!(new_tensor.size() == &TensorSize::new(vec![2, 500, 30, 10]));
    }

    #[test]
    #[should_panic]
    fn test_convert2() {
        let mut vec = vec![];
        vec.push(Tensor::<f32>::create_zeros(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::create_zeros(vec![50, 30, 10]));

        let new_tensor = vec.convert_to_tensor();

        assert!(new_tensor.size() == &TensorSize::new(vec![2, 500, 30, 10]));
    }

}