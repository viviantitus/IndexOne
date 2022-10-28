
#[cfg(test)]
mod tensor_tests {
    use crate::tensor_ops::tensor::Tensor;
    use crate::tensor_ops::size::TensorSize;
    use crate::t;

    #[test]
    fn test_slice_size() {
        let mut tensor1 = Tensor::<f32>::new(vec![500, 30, 10], true, None);
        let sliced_tensor = tensor1.slice(t![33..400, 5..10, 3]);
        assert!(sliced_tensor.size() == &TensorSize::new(vec![367, 5, 1]));
    }
}