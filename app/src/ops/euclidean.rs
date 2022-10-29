use crate::schema::tensor::Tensor;
use crate::ops::subtract::Subtract;
use crate::openblas_wrapper::norm2::Norm2;


pub trait Euclidean<Rhs=Self> {
    type Output;
    fn euclidean(&self, other: &Rhs) -> Self::Output;
}

impl Euclidean for Tensor<'_, f32> {
    type Output = f32;
    fn euclidean(&self, other: &Tensor<'_, f32>) -> Self::Output
    { 
        if self.size() != other.size(){
            panic!("Euclidean: Dimensions do not match");
        }
        let mut new_tensor = self.sub(other);
        new_tensor.norm2()
    }
}

impl Euclidean for Tensor<'_, f64> {
    type Output = f64;
    fn euclidean(&self, other: &Tensor<'_, f64>) -> Self::Output
    { 
        if self.size() != other.size(){
            panic!("Euclidean: Dimensions do not match");
        }
        let mut new_tensor = self.sub(other);
        new_tensor.norm2()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let mut t1 = Tensor::<f32>::new(vec![10, 30, 10], true, None);
        let t2 = t1.clone();

        let result = t1.euclidean(&t2);
        assert!(result == 0.0);
    }
}
