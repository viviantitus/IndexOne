use crate::schema::tensor::Tensor;
use crate::ops::subtract::Subtract;
use crate::ops::slicelinear::SliceLinear;
use crate::openblas_wrapper::norm2::Norm2;


pub trait Euclidean<Rhs=Self> {
    type Output;
    fn euclidean(&mut self, other: &mut Rhs) -> Self::Output;
}

macro_rules! euclidean_impl {
    ($($t:ty)*) => ($(

        impl<'a> Euclidean for Tensor<'a, $t> {
            type Output = Tensor<'a, $t>;

            fn euclidean(&mut self, other: &mut Tensor<'_, $t>) -> Self::Output {
                if self.size() != other.size(){
                    panic!("Euclidean: Dimensions do not match");
                }
                let mut size_copy = self.size().clone();
                size_copy.remove_dim(self.dim()-1);

                let mut sub_tensor = self.sub(&other);
                let total_elements = size_copy.total_elements();
                let mut result_tensor: Tensor<'_, $t> = Tensor::create_with_tensorsize(size_copy);

                let len_of_dim = self.size()[self.dim()-1];

                for i in 0..total_elements{
                    result_tensor[i] = sub_tensor.slice_linear(i*len_of_dim..(i+1)*len_of_dim).norm2();
                }
                result_tensor
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
        let mut t1 = Tensor::<f64>::create_random(vec![10, 30, 10, 3], None);
        let mut t2 = t1.clone();

        let result = t1.euclidean(&mut t2);
        assert!(result[0] == 0.0);
    }

    #[test]
    fn test_euclidean2() {
        let mut t1 = Tensor::<f32>::create_random(vec![10, 30, 10], Some(0.0..0.1));
        let mut t2 = Tensor::<f32>::create_random(vec![10, 30, 10], Some(10.0..10.1));

        let result = t1.euclidean(&mut t2);
        assert!(result[0] != 0.0);
    }
}
