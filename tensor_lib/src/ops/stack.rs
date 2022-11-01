use crate::{schema::tensor::Tensor, ops::memalloc::MemAlloc};

pub trait Stack{
    type Output;
    fn stack(&self) -> Tensor<'_, Self::Output>;
}


macro_rules! stack_impl {
    ($($t:ty)*) => ($(
        impl Stack for Vec<Tensor<'_, $t>> {
            type Output = $t;

            fn stack(&self) -> Tensor<'_, $t>{
                assert!(self.len() >= 1);

                let mut compare_len = self[0].size.clone();
                for i in 1..self.len(){
                    assert!(compare_len == self[i].size);
                }

                let new_size = compare_len.push_front(self.len());
                let new_data = new_size.mem_alloc();
                
                for i in 0..self.len(){
                    new_data[i*compare_len.total_elements()..(i+1)*compare_len.total_elements()].copy_from_slice(self[i].data);
                }
                
                Tensor::create_with_data_copy(new_data, new_size)
            }
        }
    )*)
}

stack_impl! { f32 f64 }



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{schema::size::TensorSize};

    #[test]
    fn test_stack() {
        let mut vec = vec![];
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));

        let new_tensor = vec.stack();

        assert!(new_tensor.size() == &TensorSize::new(vec![2, 500, 30, 10]));
    }

    #[test]
    #[should_panic]
    fn test_stack2() {
        let mut vec = vec![];
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::new(vec![50, 30, 10]));

        let new_tensor = vec.stack();

        assert!(new_tensor.size() == &TensorSize::new(vec![2, 500, 30, 10]));
    }

}