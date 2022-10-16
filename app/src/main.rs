mod tensor_ops;
use crate::tensor_ops::tensor;

fn main() {
    let size = [5, 3, 3];
    let tensor = tensor::Tensor::<f32>::new(&size, "random");

    let a = &tensor[&[0, 0, 0]];
    println!("{:?}", a);
}