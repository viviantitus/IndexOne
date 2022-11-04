use crate::schema::tensor;
use crate::schema::{tensor::Tensor, index::Indexer};
use crate::ops::memalloc::MemAlloc;


pub trait Slice<'a, T: Copy> {
    type Output;
    fn slice(&mut self, slice_vec: Vec<Indexer>) -> Self::Output;
    fn slice_at(&mut self, slice: Indexer, dim: usize) -> Self::Output;
    // fn gather(&mut self, array: Tensor<'a, bool>) -> Self::Output;
}

impl<'a, T: Copy> Slice<'a, T> for Tensor<'a, T> {
    type Output = Self;

    fn slice(&mut self, slice_vec: Vec<Indexer>) -> Self::Output{
        let new_size = self.size().slice(&slice_vec);
        let data = new_size.mem_alloc();

        let mut data_iter: usize = 0;
        for indx_ in 0..self.data.len(){
            if self.size.is_seqindex_within_slice(slice_vec.clone(), indx_){
                data[data_iter] = self.data[indx_];
                data_iter += 1;
            }
        }

        Self::create_with_data_copy(data, new_size)
    }

    fn slice_at(&mut self, slice: Indexer, dim: usize) -> Self::Output{
        let current_size = self.size();
        let slice_vec = current_size.create_slicevec(&slice, dim);
        let new_size = current_size.slice(&slice_vec);
        let data = new_size.mem_alloc();

        let mut data_iter: usize = 0;
        for indx_ in 0..self.data.len(){
            if self.size.is_seqindex_within_slice(slice_vec.clone(), indx_){
                data[data_iter] = self.data[indx_];
                data_iter += 1;
            }
        }

        Self::create_with_data_copy(data, new_size)
    }

    // fn gather(&mut self, array: Tensor<'a, bool>) -> Self::Output{
    //     assert!(self.size.total_elements() == array.size.total_elements());
    //     let tensor = Tensor::<T>::create_with_tensorsize(self.size.clone());
    //     for indx_ in 0..self.size.total_elements(){
    //         if array[indx_]{
    //             tensor.data[indx_] = self.data[indx_];
    //         }
    //     }
    //     tensor
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{t, schema::size::TensorSize};

    #[test]
    fn test_slice_size() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10]);
        let sliced_tensor = tensor1.slice(t![33..400, 5..10, 3]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![367, 5, 1]));
    }

    #[test]
    fn test_slice_size2() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10]);
        let sliced_tensor = tensor1.slice(t![450.., ..25, ..]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![50, 25, 10]));
    }

    #[test]
    fn test_slice_size3() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10]);
        let sliced_tensor = tensor1.slice(t![497..498, 5..30, 3]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![1, 25, 1]));
    }

    #[test]
    #[should_panic]
    fn test_slice_size4() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10]);
        let _sliced_tensor = tensor1.slice(t![500.., ..30, ..]);
    }

    // #[test]
    // fn test_gather() {
    //     let mut tensor1 = Tensor::<f32>::new(vec![5]);
    //     let sliced_tensor = tensor1.gather(vec![true, false, true, false, false].convert_to_tensor());
    //     assert!(sliced_tensor.size() == &TensorSize::new(vec![1, 25, 1]));
    // }
}
