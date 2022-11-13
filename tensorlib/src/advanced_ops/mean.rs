use crate::schema::tensor::Tensor;
use crate::openblas_wrapper::norm::Norm;
use crate::ops::equals::Equals;
use crate::ops::divide::Divide;
use crate::ops::slice::Slice;
use crate::ops::convert::Convert;
use crate::schema::index::Indexer;


pub trait Mean<T, Rhs=Self> {
    type Assignments;
    fn mean(&mut self) -> Self;
    fn mean_with_assignments(&mut self, assignments: &Self::Assignments, num_centroids: usize) -> Self;
}

macro_rules! mean_impl {
    ($($t:ty)*) => ($(

        impl<'a> Mean<$t> for Tensor<'a, $t> {
            type Assignments = Tensor<'a, usize>;

            fn mean(&mut self) -> Self {
                if self.dim() != 1{
                    panic!("Mean: Samples dimension is not eq to 1");
                }

                let normed_tensor = self.norm(None);
                normed_tensor.div(self.size[0] as $t)
            }

            fn mean_with_assignments(&mut self, assignments: &Self::Assignments, num_centroids: usize) -> Self{
                if self.dim() != 2{
                    panic!("Mean: Samples dimension is not eq to 2");
                }
                if assignments.data.len() != self.size[0]{
                    panic!("Mean: Assignments not equal tot samples size");
                }
                
                //mean for last dimension

                let mut mean_tensor = vec![];
                for i in 0..num_centroids{
                    let bool_tensor = assignments.equals(i);

                    let mut slice = self.slice_at(Indexer::BoolArray(bool_tensor), 0);
                    let mean = slice.norm(Some(1)).div(slice.size[0] as $t);
                    mean_tensor.push(mean);
                } 
                mean_tensor.convert_to_tensor()
            }
        }

    )*)
}

mean_impl! { f32 f64 }



#[cfg(test)]
mod tests {
    use crate::schema::size::TensorSize;

    use super::*;

    #[test]
    fn test_mean() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut samples = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![5]));

        let result = samples.mean();
        assert!(result[0]==3.0)
    }

    #[test]
    fn test_mean_with_assignments() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0, 3.0, 6.0, 9.0, 12.0, 15.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let mut samples = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![4, 5]));

        let mut assign_data: [usize; 4] = [0, 1, 2, 2];
        let assignments = Tensor::create_with_data_copy(assign_data.as_mut_slice(), TensorSize::new(vec![4]));

        let result = samples.mean_with_assignments(&assignments, 3);
        assert!(result[10]==2.5 && result[11]==5.0 && result[12]==7.5)
    }

}
