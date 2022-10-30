extern crate libc;
use crate::schema::size::TensorSize;
use crate::schema::index::Indexer;
use std::fmt::Debug;
use std::mem;
use std::ops::{Range, Index, IndexMut};

#[derive(Debug)]
pub struct Tensor<'a, T>{
    pub data: &'a mut [T],
    pub size: TensorSize,
    dim: usize
}

impl<'a, T> Tensor<'a, T>{
    pub fn new(size: Vec<usize>) -> Self{
        let tensor_size = TensorSize::new(size);
        let tensor = Self::create_with_tensorsize(tensor_size);
        tensor
    }

    pub fn alloc_mem_for_size(tensor_size: &TensorSize) -> &'a mut [T]{
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

    pub fn create_with_data_copy(data: &'a mut [T], tensor_size: TensorSize) -> Self{
        let dim = tensor_size.dim();        
        Tensor{ data: data, size: tensor_size, dim: dim}
    }

    pub fn size(&self) -> &TensorSize{
        &self.size
    }

    pub fn dim(&self) -> usize{
        self.dim
    }
}

impl<'a, T> Index<Vec<Indexer>> for Tensor<'a, T> {
    type Output = T;
    fn index(&self, index: Vec<Indexer>) -> &Self::Output {
        let data_index = self.size.calc_seq_index(index);
        &self.data[data_index]
    }
}

impl<'a, T> Index<usize> for Tensor<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T> IndexMut<usize> for Tensor<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

impl<'a, T: Copy> Clone for Tensor<'a, T>  {
    fn clone(&self) -> Self {
        let clone = Self::create_with_tensorsize(self.size.clone());
        clone.data.copy_from_slice(self.data);
        clone
    }
}
