mod tensor_ops;
use crate::tensor_ops::tensor as tensor;


fn main() {
    let size = [5,3,3];
    let tensor = tensor::Tensor::<f32>::new(&size, "random");

    let a = &tensor[&[4,1,0]];
    println!("{:?}", a);
}