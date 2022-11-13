use crate::{schema::tensor::Tensor, ops::memalloc::MemAlloc};

pub trait Concat<'a>{
    type Output;
    fn concat(self) -> Tensor<'a, Self::Output>;
}


macro_rules! concat_impl {
    ($($t:ty)*) => ($(
        impl<'a> Concat<'a> for Vec<Tensor<'a, $t>> {
            type Output = $t;

            fn concat(self) -> Tensor<'a, $t>{
                assert!(self.len() >= 1);

                let mut compare_len = self[0].size.clone();
                for i in 1..self.len(){
                    assert!(compare_len == self[i].size);
                }

                let first_dim = compare_len[0] * self.len();
                let new_size = compare_len.remove_dim(0).push_front(first_dim);
                let new_data = new_size.mem_alloc();
                
                for i in 0..self.len(){
                    new_data[i*compare_len.total_elements()..(i+1)*compare_len.total_elements()].copy_from_slice(self[i].data);
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
    use crate::{schema::size::TensorSize};

    #[test]
    fn test_concat() {
        let mut vec = vec![];
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));

        let new_tensor = vec.concat();

        assert!(new_tensor.size() == &TensorSize::new(vec![1000, 30, 10]));
    }
}