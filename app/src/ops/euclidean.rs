use crate::schema::tensor::Tensor;
use crate::ops::subtract::Subtract;
use crate::openblas_wrapper::norm2::Norm2;


pub trait Euclidean<Rhs=Self> {
    type Output;
    fn euclidean(&self, other: &Rhs) -> Self::Output;
}

macro_rules! euclidean_impl {
    ($($t:ty)*) => ($(

        impl Euclidean for Tensor<'_, $t> {
            type Output = $t;

            fn euclidean(&self, other: &Tensor<'_, $t>) -> Self::Output {
                if self.size() != other.size(){
                    panic!("Euclidean: Dimensions do not match");
                }
                let len_of_dim = self.size()[self.dim()-1];
                let mut new_tensor = self.sub(other);
                new_tensor.norm2()
            }
        }

    )*)
}

euclidean_impl! { f32 f64 }



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let t1 = Tensor::<f64>::new(vec![10, 30, 10], true, None);
        let t2 = t1.clone();

        let result = t1.euclidean(&t2);
        assert!(result == 0.0);
    }

    #[test]
    fn test_euclidean2() {
        let t1 = Tensor::<f32>::new(vec![10, 30, 10], true, None);
        let t2 = Tensor::<f32>::new(vec![10, 30, 10], true, None);

        let result = t1.euclidean(&t2);
        assert!(result != 0.0);
    }
}
