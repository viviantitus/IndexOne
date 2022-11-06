use crate::{schema::tensor::Tensor, ops::slicelinear::SliceLinear};
use crate::ops::convert::Convert;
use super::euclidean::Euclidean;
use super::variance::Variance;
use crate::openblas_wrapper::min::Min;

pub trait KMeans<'a>{
    type Output;
    fn compute_distance(&self, dataset: &mut Tensor<'a, Self::Output>) -> Tensor<'a, Self::Output>;
    fn train(&mut self, num_centorids: usize, num_iter: usize) -> Tensor<'a, Self::Output>;
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

            fn train(&mut self, num_centorids: usize, num_iter: usize) -> Tensor<'a, $t>{
                assert!(self.dim() == 2);
            
                let mut centroids = vec![];
                let mut distances = vec![];

                let mut ignore = vec![];
                for i in 0..num_centorids{
                    centroids.push(self.slice_linear_random_last_with_ignore(&mut ignore));
                    distances.push(centroids[i].compute_distance(self));
                }
                
                let distance_tensor  = distances.convert_to_tensor();
                let assignments = distance_tensor.min(Some(1));
                
                let mut centroid_tensor = centroids.convert_to_tensor();
                let variances = centroid_tensor.variance_with_assignments(self, assignments);
                println!("variances {:?}", variances);

                
                Tensor::new(vec![2])
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
    fn test_variance() {
        let mut data = [10.0, 2.0, 0.0, 1.0, 10.1];
        let mut samples = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![5, 1]));

        let _ = Tensor::train(&mut samples, 3, 3);
        assert!(0==0)
    }

}
