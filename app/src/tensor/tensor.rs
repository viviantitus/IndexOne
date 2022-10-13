extern crate libc;
use std::mem;

pub struct Tensor{
    data: *mut i32,
    size: *const usize,
    dim: usize
}

impl Tensor{
    pub fn new(size: &[usize]) -> Self{
        let mut size_of_ndarray: usize = 1;
        for i in 0..size.len(){
            size_of_ndarray *= size[i];
        }

        let data = match size.len(){
            0 => panic!("Size has to be greater than zero"),
            _ => unsafe {
                    let my_num = libc::malloc(mem::size_of::<i32>() * size_of_ndarray) as *mut i32;
                    my_num
                }
        };
        Tensor{ data: data, size: size.as_ptr(), dim: size.len()}
    }

    pub fn size(&self) -> &[usize]{
        unsafe { std::slice::from_raw_parts(self.size, self.dim) }
    }

    pub fn dim(&self) -> usize{
        self.dim
    }

    pub fn debug(&self){
        println!("tensor size = {:?}  dim = {}",  self.size(), self.dim());
    }
}