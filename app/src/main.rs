mod tensor_ops;
use crate::tensor_ops::tensor::Tensor;


fn main() {
    let tensor1 = Tensor::<f32>::new(vec![1000, 512], true, None);
    let mut tensor2 = Tensor::<f32>::new(vec![1000, 512], true, None);

    let _ = tensor1.euclidean_distance(&tensor2);

    println!("{:?}", &tensor2[vec![35, 5]]);
    println!("{:?}", tensor2.slice(t![35, 5]));

}