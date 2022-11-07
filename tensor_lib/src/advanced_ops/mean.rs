use crate::schema::tensor::Tensor;
use crate::openblas_wrapper::norm::Norm;
use crate::ops::assign::Assign;
use crate::ops::equals::Equals;
use crate::ops::divide::Divide;
use crate::ops::slice::Slice;
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
                
                //mean for last dimension

                let mut mean: Tensor<'_, $t> = Tensor::<$t>::create_zeros(vec![self.size[1]]);
                for i in 0..num_centroids{
                    let mut slice = self.slice_at(Indexer::BoolArray(assignments.equals(i)), 0);
                    mean = slice.norm(Some(1)).div(self.size[0] as $t);
                } 
                mean
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

}
