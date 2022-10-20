extern crate libc;
use crate::tensor_ops::size::TensorSize;
use std::fmt::Debug;
use std::mem;
use std::ops::{Index, Range};
use rand::{thread_rng, Rng};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard};


#[derive(Debug)]
pub struct Tensor<'a, 'b, T>{
    data: &'a mut [T],
    size: TensorSize<'b>,
    dim: usize
}

impl<'a, 'b, T: SampleUniform + PartialOrd + Clone> Tensor<'a, 'b, T>{
    pub fn new(size: &'b [usize], assign_type: &str, range: Option<Range<T>>) -> Self where Standard: Distribution<T>{
        let mut size_of_ndarray: usize = 1;
        for i in 0..size.len(){
            size_of_ndarray *= size[i];
        }

        let data = match size.len(){
            0 => panic!("Size has to be greater than zero"),
            _ => unsafe {
                    let raw_ptr: *mut T = libc::malloc(mem::size_of::<T>() * size_of_ndarray) as *mut T;
                    let slice = std::slice::from_raw_parts_mut(raw_ptr, size_of_ndarray);
                    slice
                }
        };
        let tensor_size = TensorSize::new(size);
        let mut tensor: Tensor<T> = Tensor{ data: data, size: tensor_size, dim: size.len()};
        match assign_type{
            "random" => tensor.assign_random_values(range),
            _ => panic!("assignment type not implemented")
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

    pub fn size(&self) -> TensorSize{
        self.size
    }

    pub fn dim(&self) -> usize{
        self.dim
    }

    pub fn euclidean_distance(&self, other: Tensor<T>, dim: usize) -> Tensor<'a, 'b, T>  where Standard: Distribution<T>{
        if self.size() != other.size(){
            panic!("Euclidean: Dimensions do not match");
        }
        let ret = Tensor::new(&[2,2,2], "random", None);
        return ret;
    }
}


impl<'a, 'b, T: SampleUniform + PartialOrd + Clone> Index<&[usize]> for Tensor<'a, 'b, T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        self.size.assert_index(index);
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