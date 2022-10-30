extern crate libc;
use crate::schema::size::TensorSize;
use crate::schema::index::Indexer;
use crate::schema::traits::PartialOrdwithSampling;
use std::fmt::Debug;
use std::mem;
use std::ops::{Range, Index, IndexMut};
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution, Standard};

#[derive(Debug)]
pub struct Tensor<'a, T>{
    pub data: &'a mut [T],
    size: TensorSize,
    dim: usize
}

impl<'a, T: PartialOrdwithSampling> Tensor<'a, T>{
    pub fn new(size: Vec<usize>, random: bool, range: Option<Range<T>>) -> Self where Standard: Distribution<T>{
        let tensor_size = TensorSize::new(size);
        let mut tensor = Self::create_with_tensorsize(tensor_size);
        if random{
            tensor.assign_random_values(range);
        }
        tensor
    }

    fn alloc_mem_for_size(tensor_size: &TensorSize) -> &'a mut [T]{
        let data: &mut [T] = match tensor_size.dim(){
            0 => panic!("Size has to be greater than zero"),
            _ => unsafe {
                    let raw_ptr: *mut T = libc::malloc(mem::size_of::<T>() * tensor_size.total_elements()) as *mut T;
                    let slice = std::slice::from_raw_parts_mut(raw_ptr, tensor_size.total_elements());
                    slice
                }
        };
        data
    }

    pub fn create_with_tensorsize(tensor_size: TensorSize) -> Self{
        let data = Self::alloc_mem_for_size(&tensor_size);
        Self::create_with_data_copy(data, tensor_size)
    }

    fn create_with_data_copy(data: &'a mut [T], tensor_size: TensorSize) -> Self{
        let dim = tensor_size.dim();        
        Tensor{ data: data, size: tensor_size, dim: dim}
    }

    fn assign_random_values(&mut self, range: Option<Range<T>>) where Standard: Distribution<T>{
        let mut rng = thread_rng();

        match range{
            Some(x) => for i in 0..self.data.len(){
                self.data[i] = rng.gen_range(x.clone())
            },
            None => for i in 0..self.data.len(){
                self.data[i] = rng.gen()
            }
        }
    }

    pub fn size(&self) -> &TensorSize{
        &self.size
    }

    pub fn dim(&self) -> usize{
        self.dim
    }

    pub fn slice(&mut self, slice_vec: Vec<Indexer>) -> Self{
        let new_size = self.size().slice(&slice_vec);
        let data = Self::alloc_mem_for_size(&new_size);

        let mut data_iter: usize = 0;
        for indx_ in 0..self.data.len(){
            if self.size.is_seqindex_within_slice(slice_vec.clone(), indx_){
                data[data_iter] = self.data[indx_];
                data_iter += 1;
            }
        }

        Self::create_with_data_copy(data, new_size)
    }

    pub fn slice_at(&mut self, slice: Indexer, dim: usize) -> Self{
        let current_size = self.size();
        let slice_vec = current_size.create_slicevec(&slice, dim);
        let new_size = current_size.slice(&slice_vec);
        let data = Self::alloc_mem_for_size(&new_size);

        let mut data_iter: usize = 0;
        for indx_ in 0..self.data.len(){
            if self.size.is_seqindex_within_slice(slice_vec.clone(), indx_){
                data[data_iter] = self.data[indx_];
                data_iter += 1;
            }
        }

        Self::create_with_data_copy(data, new_size)
    }

    pub fn linear_slice(&mut self, slice: Range<usize>) -> Self{
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

impl<'a, T: PartialOrdwithSampling> Index<Vec<Indexer>> for Tensor<'a, T> {
    type Output = T;
    fn index(&self, index: Vec<Indexer>) -> &Self::Output {
        let data_index = self.size.calc_seq_index(index);
        &self.data[data_index]
    }
}

impl<'a, T: PartialOrdwithSampling> Index<usize> for Tensor<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T: PartialOrdwithSampling> IndexMut<usize> for Tensor<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

impl<'a, T: PartialOrdwithSampling> Clone for Tensor<'a, T>  {
    fn clone(&self) -> Self {
        let clone = Self::create_with_tensorsize(self.size.clone());
        clone.data.copy_from_slice(self.data);
        clone
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::t;

    #[test]
    fn test_slice_size() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10], false, None);
        let sliced_tensor = tensor1.slice(t![33..400, 5..10, 3]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![367, 5, 1]));
    }

    #[test]
    fn test_slice_size2() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10], false, None);
        let sliced_tensor = tensor1.slice(t![450.., ..25, ..]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![50, 25, 10]));
    }

    #[test]
    fn test_slice_size3() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10], false, None);
        let sliced_tensor = tensor1.slice(t![497..498, 5..30, 3]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![1, 25, 1]));
    }

    #[test]
    #[should_panic]
    fn test_slice_size4() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10], false, None);
        let _sliced_tensor = tensor1.slice(t![500.., ..30, ..]);
    }
}