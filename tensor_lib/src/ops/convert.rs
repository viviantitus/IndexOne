use crate::{schema::tensor::Tensor, ops::memalloc::MemAlloc};

pub trait Convert<'a>{
    type Output;
    fn convert_to_tensor(container: Self) -> Tensor<'a, Self::Output>;
}


macro_rules! convert_impl {
    ($($t:ty)*) => ($(
        impl<'a> Convert<'a> for Vec<Tensor<'a, $t>> {
            type Output = $t;

            fn convert_to_tensor(container: Self) -> Tensor<'a, $t>{
                assert!(container.len() >= 1);

                let mut compare_len = container[0].size.clone();
                for i in 1..container.len(){
                    assert!(compare_len == container[i].size);
                }

                let new_size = compare_len.push_front(container.len());
                let new_data = new_size.mem_alloc();
                
                for i in 0..container.len(){
                    new_data[i*compare_len.total_elements()..(i+1)*compare_len.total_elements()].copy_from_slice(container[i].data);
                }
                
                Tensor::create_with_data_copy(new_data, new_size)
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
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));

        let new_tensor = Vec::convert_to_tensor(vec);

        assert!(new_tensor.size() == &TensorSize::new(vec![2, 500, 30, 10]));
    }

    #[test]
    #[should_panic]
    fn test_convert2() {
        let mut vec = vec![];
        vec.push(Tensor::<f32>::new(vec![500, 30, 10]));
        vec.push(Tensor::<f32>::new(vec![50, 30, 10]));

        let new_tensor = Vec::convert_to_tensor(vec);

        assert!(new_tensor.size() == &TensorSize::new(vec![2, 500, 30, 10]));
    }

}