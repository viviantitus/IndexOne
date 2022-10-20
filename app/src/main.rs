mod tensor_ops;
use crate::tensor_ops::tensor::Tensor;


fn main() {
    let size = [1000, 512];
    let tensor1 = Tensor::<f32>::new(&size, "random", None);
    let tensor2 = Tensor::<f32>::new(&size, "random", None);

    let a = tensor1.euclidean_distance(tensor2, 2);
    println!("{:?}", a);
}