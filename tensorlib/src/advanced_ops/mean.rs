use crate::schema::tensor::Tensor;
use crate::openblas_wrapper::norm::Norm;

pub trait Mean<T, Rhs=Self> {
    type Assignments;
    type Assignmentsu8;
    
    fn mean(&mut self) -> Self;
    fn mean_with_assignments(&mut self, assignments: &Self::Assignments, num_centorids: usize) -> Self;
    fn mean_with_assignments_for_u8(&mut self, assignments: &Self::Assignmentsu8, centroids: &mut Self);

}

macro_rules! mean_impl {
    ($($t:ty)*) => ($(

        impl Mean<$t> for Tensor<$t> {
            type Assignments = Tensor<usize>;
            type Assignmentsu8 = Tensor<u8>;


            fn mean(&mut self) -> Self {
                if self.dim() != 1{
                    panic!("Mean: Samples dimension is not eq to 1");
                }

                let mut normed_tensor = self.norm(None);
                for index in 0..normed_tensor.data.len(){
                    normed_tensor.data[index] = normed_tensor.data[index] / self.size[0] as $t;
                } 

                normed_tensor
            }

            fn mean_with_assignments(&mut self, _assignments: &Self::Assignments, _num_centorids: usize) -> Self{
                todo!("Implement the same as implemented for u8")
            }

            fn mean_with_assignments_for_u8(&mut self, assignments: &Self::Assignmentsu8, centroid_tensor: &mut Self){
                if self.dim() != 2{
                    panic!("Mean: Samples dimension is not eq to 2");
                }
                if assignments.data.len() != self.size[0]{
                    panic!("Mean: Assignments not equal tot samples size");
                }
                
                //mean for last dimension
                let mut assign_count_for_dim: [$t; 3] = [0.0, 0.0, 0.0];
                centroid_tensor.data.fill(0.0);

                for i in 0..self.size[0]{
                    assign_count_for_dim[usize::from(assignments[i])] += 1.0;
                }

                for sample_indx in 0..self.size[0]{
                    for dim in 0..self.size[1]{
                        centroid_tensor[(usize::from(assignments[sample_indx]) * self.size[1]) + dim] += self[(sample_indx * self.size[1]) + dim];
                    }  
                }

                for centroid_indx in 0..centroid_tensor.size[0]{
                    for dim in 0..self.size[1]{
                        centroid_tensor[(centroid_indx * self.size[1]) + dim] /= assign_count_for_dim[centroid_indx];
                    }
                }

            }
        }

    )*)
}

mean_impl! { f32 f64 }



#[cfg(test)]
mod tests {
    use crate::{schema::size::TensorSize};

    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut samples = Tensor::create_with_data_copy(data, TensorSize::new(vec![5]));

        let result = samples.mean();
        assert!(result[0]==3.0)
    }

    // #[test]
    // fn test_mean_with_assignments() {
    //     let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0, 3.0, 6.0, 9.0, 12.0, 15.0, 2.0, 4.0, 6.0, 8.0, 10.0];
    //     let mut samples = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![4, 5]));

    //     let mut assign_data: [u8; 4] = [0, 1, 2, 2];
    //     let assignments = Tensor::create_with_data_copy(assign_data.as_mut_slice(), TensorSize::new(vec![4]));

    //     let mut centroid_tensor = Tensor::<f32>::create_zeros(vec![3, 5]);

    //     samples.mean_with_assignments_for_u8(&assignments, &mut centroid_tensor);
    //     assert!(centroid_tensor[10]==2.5 && centroid_tensor[11]==5.0 && centroid_tensor[12]==7.5)
    // }

}
