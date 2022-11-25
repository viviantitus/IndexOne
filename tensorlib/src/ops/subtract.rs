use crate::schema::tensor::Tensor;


pub trait Subtract<Rhs=Self> {
    type Output;
    fn sub(&self, other: &Rhs) -> Self::Output;
}

impl<T: std::ops::Sub<Output=T> + Copy> Subtract for Tensor<T> {
    type Output = Self;

    fn sub(&self, other: &Tensor<T>) -> Self::Output
    { 
        if self.size() != other.size(){
            panic!("Subtract func: Size do not match")
        }

        let mut tensor = Tensor::create_with_tensorsize(self.size().clone());
        for i in 0..self.data.len(){
            tensor.data.push(self.data[i] - other.data[i]);
        }
        tensor
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{t, ops::random::Random};

    #[test]
    fn test_subtract() {
        let t1 = Tensor::<f32>::create_random(vec![10, 30, 10], None);
        let t2 = t1.clone();

        let result = t1.sub(&t2);
        assert!(result[t![0, 0, 0]] == 0.0);
    }
}