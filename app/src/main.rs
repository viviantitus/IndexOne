use tensorlib::{schema::tensor::Tensor, advanced_ops::kmeans::KMeans};
use std::time::Instant;

fn main() {
    let start = Instant::now();

    let mut dataset = Tensor::<f32>::new(vec![10000, 512]);
    dataset.train(3, 10);

    let duration = start.elapsed();
    println!("Total time taken to run is {:?}", duration);
}

