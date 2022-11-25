use crate::schema::size::TensorSize;
use crate::schema::tensor::Tensor;

pub trait Assign{
    type Output;
    fn create_zeros(size: Vec<usize>) -> Self::Output;
    fn create_zeros_with_tensorsize(size: TensorSize) -> Self::Output;
}

macro_rules! assign_impl {
    ($($t:ty)*) => ($(

        impl Assign for Tensor<$t> {
            type Output = Self;

            fn create_zeros(size: Vec<usize>) -> Self::Output
            { 
                let tensor_size = TensorSize::new(size);
                Self::create_zeros_with_tensorsize(tensor_size)
            }

            fn create_zeros_with_tensorsize(tensor_size: TensorSize) -> Self::Output
            { 
                let data = vec![0.0; tensor_size.total_elements()];
                Self::create_with_data_copy(data, tensor_size)
            }
        }
)*)
}


assign_impl! { f32 f64 }

macro_rules! assign_impl_for_integers {
    ($($t:ty)*) => ($(

        impl Assign for Tensor<$t> {
            type Output = Self;

            fn create_zeros(size: Vec<usize>) -> Self::Output
            { 
                let tensor_size = TensorSize::new(size);
                Self::create_zeros_with_tensorsize(tensor_size)
            }

            fn create_zeros_with_tensorsize(tensor_size: TensorSize) -> Self::Output
            { 
                let data = vec![0; tensor_size.total_elements()];
                Self::create_with_data_copy(data, tensor_size)
            }
        }
)*)
}


assign_impl_for_integers! { u8 usize i32 i64 }

macro_rules! assign_impl_for_bool {
    ($($t:ty)*) => ($(

        impl Assign for Tensor<$t> {
            type Output = Self;

            fn create_zeros(size: Vec<usize>) -> Self::Output
            { 
                let tensor_size = TensorSize::new(size);
                Self::create_zeros_with_tensorsize(tensor_size)
            }

            fn create_zeros_with_tensorsize(tensor_size: TensorSize) -> Self::Output
            { 
                let data = vec![false; tensor_size.total_elements()];
                Self::create_with_data_copy(data, tensor_size)
            }
        }
)*)
}


assign_impl_for_bool! { bool }