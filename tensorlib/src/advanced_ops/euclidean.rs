use crate::schema::tensor::Tensor;
use crate::ops::subtract::Subtract;
use crate::openblas_wrapper::norm2::Norm2;


pub trait Euclidean<T, Rhs=Self> {
    type Output;
    fn euclidean(&self, other: &Rhs) -> T;
}

macro_rules! euclidean_impl {
    ($($t:ty)*) => ($(

        impl<'a> Euclidean<$t> for Tensor<'a, $t> {
            type Output = Tensor<'a, $t>;

            fn euclidean(&self, other: &Tensor<'_, $t>) -> $t {
                if self.size() != other.size(){
                    panic!("Euclidean: Dimensions do not match");
                }
                let mut sub_tensor = self.sub(&other);
                sub_tensor.norm2()
            }
        }

    )*)
}

euclidean_impl! { f32 f64 }



#[cfg(test)]
mod tests {
    use crate::ops::random::Random;

    use super::*;

    #[test]
    fn test_euclidean() {
        let t1 = Tensor::<f64>::create_random(vec![300], None);
        let t2 = t1.clone();

        let result = t1.euclidean(&t2);
        assert!(result == 0.0);
    }

    #[test]
    fn test_euclidean2() {
        let t1 = Tensor::<f32>::create_random(vec![20], Some(0.0..0.00001));
        let t2 = Tensor::<f32>::create_random(vec![20], Some(10.0..10.00001));

        let result = t2.euclidean(&t1);
        assert!(result != 0.0);
    }

    #[test]
    #[should_panic]
    fn test_euclidean3() {
        let t1 = Tensor::<f64>::create_random(vec![10, 30, 10, 3], None);
        let t2 = t1.clone();

        let result = t1.euclidean(&t2);
        assert!(result == 0.0);
    }

}
