
#[cfg(test)]
mod tensor_tests {
    use crate::tensor_ops::size::TensorSize;
    use crate::t;

    #[test]
    fn test_is_within_range() {
        let size = TensorSize::new(vec![5, 10, 2, 5]);
        let cond = size.is_within_sliceindex(&t![3, 3, 1, 4], 339);
        assert!(cond);
    }
}