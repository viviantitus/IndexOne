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

impl<T:Copy> SliceLinear<T> for Tensor<T> {
    type Output = Self;
    fn slice_linear(&mut self, slice: Range<usize>) -> Self{
        let new_size = TensorSize::new(vec![slice.end-slice.start]);
        let mut data = vec![];
        data.extend_from_slice(&self.data[slice.start..slice.end]);
        Self::create_with_data_copy(data, new_size)
    }

    fn slice_linear_last(&mut self, index: usize) -> Self::Output{
        let len_of_dim = self.size[self.dim-1];
        let range = index*len_of_dim..(index+1)*len_of_dim;
        self.slice_linear(range)
    }

    fn slice_linear_random_last(&mut self) -> Self::Output{
        let size = self.size.remove_dim(self.dim-1);

        let random_num = thread_rng().gen_range(0..size.total_elements());
        self.slice_linear_last(random_num)
    }


    fn slice_linear_random_last_with_ignore(&mut self, ignore: &mut Vec<usize>) -> Self::Output{
        let size = self.size.remove_dim(self.dim-1);
        let output: Self;
        loop {
            let random_num = thread_rng().gen_range(0..size.total_elements());
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