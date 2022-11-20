use crate::{schema::tensor::Tensor, ops::slicelinear::SliceLinear};
use crate::ops::convert::Convert;
use super::euclidean::Euclidean;
use super::variance::Variance;
use crate::openblas_wrapper::min::Min;
use crate::advanced_ops::mean::Mean;


pub trait KMeans<'a>{
    type Output;
    fn compute_distance(&self, dataset: &mut Tensor<'a, Self::Output>) -> Tensor<'a, Self::Output>;
    fn train(&'a mut self, num_centorids: usize, num_init: usize, num_iter: usize, tolerance: Self::Output) -> (Self::Output, Tensor<'a, Self::Output>, Tensor<'a, u8>);
}

macro_rules! kmeans_impl {
    ($($t:ty)*) => ($(

        impl<'a> KMeans<'a> for Tensor<'_, $t> {
            type Output = $t;

            fn compute_distance(&self, dataset: &mut Tensor<'a, $t>) -> Tensor<'a, $t> {
                let mut distances = Tensor::new(vec![dataset.size[0]]);
                for i in 0..dataset.size[0]{
                    distances[i] = self.euclidean(&dataset.slice_linear_last(i));
                }
                distances
            }

            fn train(&'a mut self, num_centorids: usize, num_init: usize, num_iter: usize, tolerance: Self::Output) -> ($t, Tensor<'a, $t>, Tensor<'a, u8>){
                assert!(self.dim() == 2);
                
                let mut final_centroids: Tensor<'a, $t> = Tensor::new(vec![num_centorids, self.size[1]]);
                let mut final_assignments: Tensor<'a, u8> = Tensor::new(vec![self.size[0]]);
                let mut final_variance = <$t>::MAX;

                for _ in 0..num_init{
                    let mut iter_variance = <$t>::MAX;

                    let mut centroids = vec![];
                    let mut ignore = vec![];

                    for _ in 0..num_centorids{
                        centroids.push(self.slice_linear_random_last_with_ignore(&mut ignore));
                    }
                    let mut centroid_tensor  = centroids.convert_to_tensor();

                    let mut loop_iter = 0;
                    loop {
                        loop_iter += 1;
                        let mut distances = vec![];

                        for i in 0..num_centorids{
                            distances.push(centroid_tensor.slice_linear_last(i).compute_distance(self));
                        }
                        
                        let distance_tensor  = distances.convert_to_tensor();
                        let assignments = distance_tensor.min_for_u8(Some(1));
                        
                        let variance = centroid_tensor.variance_with_assignments_for_u8(self, &assignments);
                        
                        if (variance - iter_variance).abs() < tolerance{
                            if variance < final_variance{
                                final_variance = variance;
                                final_centroids = centroid_tensor;
                                final_assignments = assignments;
                            }
                            break;
                        }
                        if variance < iter_variance{
                                iter_variance = variance;
                        }
                        if loop_iter >= num_iter{
                            break;
                        }
                        self.mean_with_assignments_for_u8(&assignments, &mut centroid_tensor);
                    }
                }
                                
                (final_variance, final_centroids, final_assignments)
            }
        }

    )*)
}


kmeans_impl! { f32 f64 }



#[cfg(test)]
mod tests {
    use crate::schema::size::TensorSize;

    use super::*;


    #[test]
    fn test_kmeans() {
        let mut data = [10.0, 2.0, 0.0, 1.0, 10.1];
        let mut samples = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![5, 1]));

        let _ = Tensor::train(&mut samples, 3, 10, 300, 1e-4);
    }

}
