mod tensor;


fn main() {
    let size: [usize; 3] = [3,4,2];
    let tensor = tensor::tensor::Tensor::new(&size);

    // println!("{}",  unsafe { *(tensor.add(2))});
    tensor.debug();

}