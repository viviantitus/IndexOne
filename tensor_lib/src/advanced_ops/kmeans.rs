use crate::{schema::tensor::Tensor, ops::slicelinear::SliceLinear};
use crate::ops::convert::Convert;
use super::euclidean::Euclidean;
use crate::openblas_wrapper::min::Min;

pub trait KMeans<'a>{
    type Output;
    fn compute_distance(&self, dataset: &mut Tensor<'a, Self::Output>) -> Tensor<'a, Self::Output>;
    fn train(&mut self, num_centorids: i32, num_iter: i32) -> Tensor<'a, Self::Output>;
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

            fn train(&mut self, num_centorids: i32, num_iter: i32) -> Tensor<'a, $t>{
                assert!(self.dim() == 2);
            
                let mut distances = vec![];

                let mut ignore = vec![];
                for _ in 0..num_centorids{
                    let query = self.slice_linear_random_last_with_ignore(&mut ignore);
                    distances.push(query.compute_distance(self));
                }
            
                let mut assignments  = distances.convert_to_tensor();
                let min_indices= &mut assignments.min(Some(0));
                
                println!("{:?}", min_indices);

                
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
        let mut data = [10.0, 2.0, 0.0, 1.0, 10.0];
        let mut samples = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![5, 1]));

        let result = Tensor::train(&mut samples, 3, 3);
        assert!(1==0)
    }

}
