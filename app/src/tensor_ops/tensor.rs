extern crate libc;
use crate::tensor_ops::size::TensorSize;
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

impl<'a, T: SampleUniform + PartialOrd + Clone> Tensor<'a, T>{
    pub fn new(size: Vec<usize>, random: bool, range: Option<Range<T>>) -> Self where Standard: Distribution<T>{
        let tensor_size = TensorSize::new(size);
        Self::create_with_tensorsize(tensor_size, random, range)
    }

    fn create_with_tensorsize(tensor_size: TensorSize, random: bool, range: Option<Range<T>>) -> Self where Standard: Distribution<T>{
        let data = match tensor_size.len(){
            0 => panic!("Size has to be greater than zero"),
            _ => unsafe {
                    let raw_ptr: *mut T = libc::malloc(mem::size_of::<T>() * tensor_size.total_elements()) as *mut T;
                    let slice = std::slice::from_raw_parts_mut(raw_ptr, tensor_size.total_elements());
                    slice
                }
        };
        let dim = tensor_size.len();
        let mut tensor: Tensor<T> = Tensor{ data: data, size: tensor_size, dim: dim};
        if random{
            tensor.assign_random_values(range);
        }
        tensor
    }

    pub fn assign_random_values(&mut self, range: Option<Range<T>>) where Standard: Distribution<T>{
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

    pub fn euclidean_distance(&self, other: Tensor<T>) -> Tensor<'a, T>  where Standard: Distribution<T>{
        if self.size != other.size{
            panic!("Euclidean: Dimensions do not match");
        }
        // TODO: Dimension for euclidean distance is set to last
        let ret = Tensor::create_with_tensorsize(self.size.copy(), false, None);
        return ret;
    }
}

#[derive(Debug)]
pub enum Indexer {
    range(Range<usize>),
    number(usize)
}

impl<'a, T: SampleUniform + PartialOrd + Clone> Index<Vec<usize>> for Tensor<'a, T> {
    type Output = T;
    fn index(&self, index: Vec<usize>) -> &Self::Output {
        self.size.assert_index(&index);
        let mut data_index = 0;

        for i in 0..self.dim(){
            if i == self.dim() -1 {
                data_index += index[i];
            }
            else{
                data_index += self.size.cumulative_size(i+1) * index[i];
            }
        }
        &self.data[data_index]
    }
}