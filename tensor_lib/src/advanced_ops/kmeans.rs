use crate::{schema::tensor::Tensor, ops::slicelinear::SliceLinear};
use crate::ops::convert::Convert;
use super::euclidean::Euclidean;

pub trait KMeans {
    type Output;
    fn compute_distance<'a>(dataset: &mut Tensor<'a, Self::Output>, query: &Tensor<'a, Self::Output>) -> Tensor<'a, Self::Output>;
    fn train<'a>(dataset: &mut Tensor<'a, Self::Output>, num_centorids: i32, num_iter: i32) -> Tensor<'a, Self::Output>;
}

macro_rules! kmeans_impl {
    ($($t:ty)*) => ($(

        impl KMeans for Tensor<'_, $t> {
            type Output = $t;

            fn compute_distance<'a>(dataset: &mut Tensor<'a, $t>, query: &Tensor<'a, $t>) -> Tensor<'a, $t> {
                let mut distances = Tensor::new(vec![dataset.size[0]]);
                for i in 0..dataset.size[0]{
                    distances[i] = query.euclidean(&dataset.slice_linear_last(i));
                }
                distances
            }

            fn train<'a>(dataset: &mut Tensor<'a, $t>, num_centorids: i32, num_iter: i32) -> Tensor<'a, $t>{
                assert!(dataset.dim() == 2);
            
                let mut distances = vec![];
            
                for _ in 0..num_centorids{
                    let query = dataset.slice_linear_random_last();
                    distances.push(Self::compute_distance(dataset, &query));
                }
            
                distances.convert_to_tensor()
            }
        }

    )*)
}


kmeans_impl! { f32 f64 }