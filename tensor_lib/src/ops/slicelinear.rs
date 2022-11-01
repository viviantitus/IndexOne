use std::ops::Range;

use crate::schema::{tensor::Tensor, size::TensorSize};


pub trait SliceLinear<T> {
    type Output;
    fn slice_linear(&mut self, slice: Range<usize>) -> Self::Output;
}

impl<T> SliceLinear<T> for Tensor<'_, T> {
    type Output = Self;
    fn slice_linear(&mut self, slice: Range<usize>) -> Self{
        let total_elements = slice.end-slice.start;
        let new_size = TensorSize::new(vec![slice.end-slice.start]);

        let data: &mut [T];
        unsafe{
            let start = slice.start;
            data = std::slice::from_raw_parts_mut(self.data.as_mut_ptr().add(start), total_elements);
        }
        Self::create_with_data_copy(data, new_size)
    }
}