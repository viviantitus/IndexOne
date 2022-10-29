use rand::distributions::uniform::SampleUniform;
use crate::schema::tensor::Tensor;


pub trait Subtract<Rhs=Self> {
    type Output;
    fn sub(&mut self, other: &Rhs) -> Self::Output;
}

impl<T: SampleUniform + PartialOrd + Copy  + std::ops::Sub<Output=T>> Subtract for Tensor<'_, T> {
    type Output = Self;

    fn sub(&mut self, other: &Tensor<'_, T>) -> Self::Output
    { 
        if self.size() != other.size(){
            panic!("Subtract func: Size do not match")
        }

        let tensor = Tensor::create_with_tensorsize(self.size().clone());
        for i in 0..self.size().total_elements(){
            tensor.data[i] = self.data[i] - other.data[i];
        }
        tensor
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::t;

    #[test]
    fn test_subtract() {
        let mut t1 = Tensor::<f32>::new(vec![10, 30, 10], true, None);
        let t2 = t1.clone();

        let result = t1.sub(&t2);
        assert!(result[t![0, 0, 0]] == 0.0);
    }
}