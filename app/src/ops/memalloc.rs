use std::mem;

use crate::schema::size::TensorSize;


pub trait MemAlloc<'a, T>{
    fn mem_alloc(&self) -> &'a mut [T];
}


impl<'a, T> MemAlloc<'a, T> for TensorSize {

    fn mem_alloc(&self) -> &'a mut [T]{
        let data: &mut [T] = match self.dim(){
            0 => panic!("Size has to be greater than zero"),
            _ => unsafe {
                    let raw_ptr: *mut T = libc::malloc(mem::size_of::<T>() * self.total_elements()) as *mut T;
                    let slice = std::slice::from_raw_parts_mut(raw_ptr, self.total_elements());
                    slice
                }
        };
        data
    }
}