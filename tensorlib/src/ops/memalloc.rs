
use crate::schema::size::TensorSize;


pub trait MemAlloc<T: Clone>{
    fn mem_alloc(&self) -> Vec<T>;
}


impl<T: Clone> MemAlloc<T> for TensorSize {

    fn mem_alloc(&self) -> Vec<T>{
        let data = Vec::<T>::with_capacity(self.total_elements());
        data
    }
}