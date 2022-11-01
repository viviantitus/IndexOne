// use crate::{schema::tensor::Tensor, ops::slicelinear::SliceLinear};
// use crate::ops::convert::Convert;
// use crate::ops::stack::Stack;
// use super::euclidean::Euclidean;



// pub trait KMeans {
//     type Output;
//     fn compute_distance(&mut self, query: &Self) -> Tensor<'_, Self::Output>;
//     fn train(dataset: &mut Self, num_centorids: i32, num_iter: i32) -> Tensor<'_, Self::Output>;
// }

// macro_rules! kmeans_impl {
//     ($($t:ty)*) => ($(

//         impl KMeans for Tensor<'_, $t> {
//             type Output = $t;

//             fn compute_distance(&mut self, query: &Self) -> Tensor<'_, Self::Output> {
//                 let mut distances = Tensor::new(vec![self.size[0]]);
//                 for i in 0..self.size[0]{
//                     distances[i] = query.euclidean(&self.slice_linear_last(i));
//                 }
//                 distances
//             }

//             fn train(dataset: &mut Self, num_centorids: i32, num_iter: i32) -> Tensor<'_, Self::Output> {
//                 assert!(dataset.dim() == 2);

//                 let mut queries: Vec<Tensor<$t>> = vec![];

//                 for _ in 0..num_centorids{
//                     queries.push(dataset.slice_linear_random_last());
//                 }

//                 queries.stack()

//                 // centroid_distances.push(dataset.compute_distance(&query));

//                 // // let query = dataset.slice_linear_random_last();

//                 // let a= centroid_distances.stack();
//                 // a

//             }
//         }

//     )*)
// }


// kmeans_impl! { f32 f64 }