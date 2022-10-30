extern crate libc;
use crate::ops::memalloc::MemAlloc;
use crate::schema::size::TensorSize;
use crate::schema::index::Indexer;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

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

    pub fn create_with_tensorsize(tensor_size: TensorSize) -> Self{
        let data = tensor_size.mem_alloc();
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
