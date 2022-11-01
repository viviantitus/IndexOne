use crate::{schema::tensor::Tensor, ops::slicelinear::SliceLinear};
use crate::ops::euclidean::Euclidean;



pub trait KMeans<T> {
    type Output;
    fn compute_distance(&mut self, samples: Self) -> Vec<T>;
    fn train(&mut self, num_centorids: i32, num_iter: i32) -> &Self::Output;
}

macro_rules! kmeans_impl {
    ($($t:ty)*) => ($(

        impl<'a> KMeans<$t> for Tensor<'a, $t> {
            type Output = Self;

            fn compute_distance(&mut self, sample: Tensor<'a, $t>) -> Vec<$t>{
                let mut distances = vec![];
                for i in 0..self.size[0]{
                    let distance = sample.euclidean(&self.slice_linear_last(i));
                    distances.push(distance);
                }
                distances
            }

            fn train(&mut self, num_centorids: i32, num_iter: i32) -> &Self::Output {
                assert!(self.dim() == 2);

                for _ in 0..num_centorids{
                    let rand_sample = self.slice_linear_random_last();
                    self.compute_distance(rand_sample);
                }

                self
            }
        }

    )*)
}


kmeans_impl! { f32 f64 }