use crate::schema::tensor::Tensor;


pub trait Euclidean<T, Rhs=Self> {
    type Output;
    fn euclidean(&self, other: &Rhs) -> T;
}

macro_rules! euclidean_impl {
    ($($t:ty)*) => ($(

        impl<'a> Euclidean<$t> for Tensor<$t> {
            type Output = Tensor<$t>;

            fn euclidean(&self, other: &Tensor<$t>) -> $t {
                if self.size() != other.size() || self.dim() != 1{
                    panic!("Euclidean: Dimensions do not match");
                }
                let mut norm: $t = 0.0;
                for i in 0..self.size[0]{
                    norm += (self.data[i] - other.data[i]).powi(2);
                }
                norm.sqrt()
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
