extern crate libc;
use crate::tensor_ops::size::TensorSize;
use crate::tensor_ops::index::Indexer;
use std::fmt::Debug;
use std::mem;
use std::ops::{Index, Range};
use rand::{thread_rng, Rng};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard};

#[derive(Debug)]
pub struct Tensor<'a, T>{
    data: &'a mut [T],
    size: TensorSize,
    dim: usize
}

impl<'a, T: SampleUniform + PartialOrd + Copy> Tensor<'a, T>{
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

    fn create_with_tensorsize(tensor_size: TensorSize) -> Self{
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

    pub fn slice(&mut self, sliceindex: Vec<Indexer>) -> Self{
        let new_size = TensorSize::create_with_sliceindex(&sliceindex);
        let data = Self::alloc_mem_for_size(&new_size);

        let mut data_iter: usize = 0;
        for indx_ in 0..self.data.len(){
            if self.size.is_within_sliceindex(&sliceindex, indx_){
                data[data_iter] = self.data[indx_];
                data_iter += 1;
            }
        }

        Self::create_with_data_copy(data, new_size)
    }

    pub fn euclidean_distance(&self, other: &Tensor<T>) -> Tensor<'a, T>  where Standard: Distribution<T>{
        if self.size != other.size{
            panic!("Euclidean: Dimensions do not match");
        }
        let mut new_size = self.size.clone();
        new_size.remove_dim(self.size.dim()-1);
        // TODO: Dimension for euclidean distance is set to last
        let ret = Tensor::create_with_tensorsize(new_size);

        return ret;
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::t;

    #[test]
    fn test_slice_size() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10], true, None);
        let sliced_tensor = tensor1.slice(t![33..400, 5..10, 3]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![367, 5, 1]));
    }
}