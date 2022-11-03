use std::ops::Range;

use rand::{thread_rng, Rng};

use crate::schema::{tensor::Tensor, size::TensorSize};


pub trait SliceLinear<T> {
    type Output;
    fn slice_linear(&mut self, slice: Range<usize>) -> Self::Output;
    fn slice_linear_last(&mut self, index: usize) -> Self::Output;
    fn slice_linear_random_last(&mut self) -> Self::Output;
    fn slice_linear_random_last_with_ignore(&mut self, ignore: &mut Vec<usize>) -> Self::Output;
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

    fn slice_linear_last(&mut self, index: usize) -> Self::Output{
        let len_of_dim = self.size[self.dim()-1];
        let range = index*len_of_dim..(index+1)*len_of_dim;
        self.slice_linear(range)
    }

    fn slice_linear_random_last(&mut self) -> Self::Output{
        let size = self.size.remove_dim(self.dim()-1);

        let random_num = thread_rng().gen_range(0..size.total_elements());
        self.slice_linear_last(random_num)
    }


    fn slice_linear_random_last_with_ignore(&mut self, ignore: &mut Vec<usize>) -> Self::Output{
        let size = self.size.remove_dim(self.dim()-1);

        let random_num = thread_rng().gen_range(0..size.total_elements());

        let output: Self;
        loop {
            if !ignore.contains(&random_num){
                ignore.push(random_num);
                output = self.slice_linear_last(random_num);
                break;
            }
            else {
                continue;
            }
        }
        output
    }
}