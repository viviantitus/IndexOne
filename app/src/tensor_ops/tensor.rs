extern crate libc;
use std::mem;
use std::ops::Index;
use rand::{thread_rng, Rng};
use rand::distributions::{Distribution, Standard};

#[derive(Debug)]
pub struct Tensor<'a, T>{
    data: *mut T,
    size: &'a [usize],
    dim: usize
}

impl<'a, T> Tensor<'a, T>{
    pub fn new(size: &'a [usize], assign_type: &str) -> Self where Standard: Distribution<T>{
        let mut size_of_ndarray: usize = 1;
        for i in 0..size.len(){
            size_of_ndarray *= size[i];
        }

        let data = match size.len(){
            0 => panic!("Size has to be greater than zero"),
            _ => unsafe {
                    let my_num: *mut T = libc::malloc(mem::size_of::<T>() * size_of_ndarray) as *mut T;
                    my_num
                }
        };
        let tensor: Tensor<T> = Tensor{ data: data, size: &size, dim: size.len()};
        match assign_type{
            "random" => tensor.assign_random_values(),
            _ => panic!("assignment type not implemented")
        }
        tensor
    }

    pub fn assign_random_values(&self) where Standard: Distribution<T>{
        let mut rng = thread_rng();
        let size_of_ndarray = self.get_total_len();
        unsafe{
            for i in 0..size_of_ndarray{
                *self.data.add(i) = rng.gen();
            }
        }
    }

    pub fn size(&self) -> &[usize]{
        self.size
    }

    pub fn dim(&self) -> usize{
        self.dim
    }

    fn get_total_len(&self) -> usize{
        let mut size_of_ndarray: usize = 1;
        let size = self.size();
        for i in 0..size.len(){
            size_of_ndarray *= size[i];
        }
        size_of_ndarray
    }

    fn cumulative_size(&self, indx: usize) -> usize{
        let mut ret_val:usize = 1;
        for i in indx..self.size.len(){
            ret_val *= self.size[i];
        }
        ret_val
    }
}


impl<'a, T> Index<&[usize]> for Tensor<'a, T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        if index.len() != self.dim(){
            panic!("Index size does not match tensor dimension size");
        }
        let mut data_index = 0;

        for i in 0..self.dim(){
            if i == self.dim() -1 {
                data_index += index[i];
            }
            else{
                data_index += self.cumulative_size(i+1) * index[i];
            }
        }
        unsafe { &*self.data.add(data_index) }
    }
}