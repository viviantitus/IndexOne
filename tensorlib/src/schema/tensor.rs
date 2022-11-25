extern crate libc;
use crate::ops::memalloc::MemAlloc;
use crate::schema::size::TensorSize;
use crate::schema::index::Indexer;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Tensor<T>{
    pub data: Vec<T>,
    pub size: TensorSize,
    pub dim: usize
}

impl<T: Clone> Tensor<T>{

    pub fn create_with_tensorsize(tensor_size: TensorSize) -> Self{
        let data = tensor_size.mem_alloc();
        Self::create_with_data_copy(data, tensor_size)
    }

    pub fn create_with_data_copy(data: Vec<T>, tensor_size: TensorSize) -> Self{
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

impl<T> Index<Vec<Indexer>> for Tensor<T> {
    type Output = T;
    fn index(&self, index: Vec<Indexer>) -> &Self::Output {
        let data_index = self.size.calc_seq_index(index);
        &self.data[data_index]
    }
}

impl<T> IndexMut<Vec<Indexer>> for Tensor<T> {
    fn index_mut(&mut self, index: Vec<Indexer>) -> &mut T {
        let data_index = self.size.calc_seq_index(index);
        &mut self.data[data_index]
    }
}

impl<T> Index<usize> for Tensor<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Tensor<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}
