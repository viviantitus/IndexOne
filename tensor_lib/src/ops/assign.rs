use crate::schema::size::TensorSize;
use crate::schema::tensor::Tensor;

use super::memalloc::MemAlloc;


pub trait Assign{
    type Output;
    fn create_zeros(size: Vec<usize>) -> Self::Output;
    fn create_zeros_with_tensorsize(size: TensorSize) -> Self::Output;
}

macro_rules! assign_impl {
    ($($t:ty)*) => ($(

        impl Assign for Tensor<'_, $t> {
            type Output = Self;

            fn create_zeros(size: Vec<usize>) -> Self::Output
            { 
                let tensor_size = TensorSize::new(size);
                Self::create_zeros_with_tensorsize(tensor_size)
            }

            fn create_zeros_with_tensorsize(tensor_size: TensorSize) -> Self::Output
            { 
                let data = tensor_size.mem_alloc();
                data.fill(0.0);
                Self::create_with_data_copy(data, tensor_size)
            }
        }
)*)
}


assign_impl! { f32 f64 }