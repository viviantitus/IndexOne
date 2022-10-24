mod tensor_ops;
use crate::tensor_ops::tensor::Tensor;
use crate::tensor_ops::tensor::Indexer;



fn main() {
    let tensor1 = Tensor::<f32>::new(vec![1000, 512], true, None);
    let tensor2 = Tensor::<f32>::new(vec![1000, 512], true, None);

    let a = tensor1.euclidean_distance(tensor2);
    println!("{:?}", a);

    // TODO: convert this into macro
    let ina:Vec<Indexer> = vec![Indexer::number(1), Indexer::range(1..2)];
}