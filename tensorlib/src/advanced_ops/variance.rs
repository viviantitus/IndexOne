use crate::schema::tensor::Tensor;
use crate::ops::subtract::Subtract;
use crate::openblas_wrapper::norm2::Norm2;
use crate::ops::slicelinear::SliceLinear;


pub trait Variance<T, Rhs=Self> {
    type Output;
    type Assignments;
    type Assignmentsu8;

    fn variance(&self, samples: &mut Rhs) -> T;
    fn variance_with_assignments(&mut self, samples: &mut Rhs, assignments: &Self::Assignments) -> T;
    fn variance_with_assignments_for_u8(&mut self, samples: &mut Rhs, assignments: &Self::Assignmentsu8) -> T;

}

macro_rules! variance_impl {
    ($($t:ty)*) => ($(

        impl Variance<$t> for Tensor<$t> {
            type Output = Tensor<$t>;
            type Assignments = Tensor<usize>;
            type Assignmentsu8 = Tensor<u8>;


            fn variance(&self, samples: &mut Tensor<$t>) -> $t {
                if self.dim() != 1{
                    panic!("Variance: Mean dimension is more than one");
                }
                if samples.dim() != 2{
                    panic!("Variance: Samples dimension is not eq to 2");
                }
                assert!(self.size[self.dim()-1] == samples.size[samples.dim()-1]);

                let mut variance: $t = 0.0;
                for i in 0..samples.size[0]{
                    let mut sub_tensor = self.sub(&samples.slice_linear_last(i));
                    variance += sub_tensor.norm2();
                }
                variance
            }

            fn variance_with_assignments(&mut self, samples: &mut Tensor<$t>, assignments: &Self::Assignments) -> $t {
                if self.dim() > 2{
                    panic!("Variance: Mean dimension is more than one");
                }
                if samples.dim() != 2{
                    panic!("Variance: Samples dimension is not eq to 2");
                }
                assert!(self.size[self.dim()-1] == samples.size[samples.dim()-1]);

                let mut variance = 0.0;
                for i in 0..samples.size[0]{
                    let mut sub_tensor = self.slice_linear_last(assignments[i]).sub(&samples.slice_linear_last(i));
                    variance += sub_tensor.norm2();
                }
                variance
            }

            fn variance_with_assignments_for_u8(&mut self, samples: &mut Tensor<$t>, assignments: &Self::Assignmentsu8) -> $t {
                if self.dim() > 2{
                    panic!("Variance: Mean dimension is more than one");
                }
                if samples.dim() != 2{
                    panic!("Variance: Samples dimension is not eq to 2");
                }
                assert!(self.size[self.dim()-1] == samples.size[samples.dim()-1]);

                let mut variance = 0.0;
                for i in 0..samples.size[0]{
                    let first = self.slice_linear_last(usize::from(assignments[i]));
                    let second = samples.slice_linear_last(i);
                    for i in 0..first.size[0]{
                        variance += (first.data[i] - second.data[i]).powi(2);
                    }
                }
                variance
            }
        }

    )*)
}

variance_impl! { f32 f64 }



#[cfg(test)]
mod tests {
    use crate::schema::size::TensorSize;

    use super::*;

    #[test]
    fn test_variance() {
        let data = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let mut samples = Tensor::create_with_data_copy(data, TensorSize::new(vec![5, 1]));
        let mean  = samples.slice_linear_last(2);

        let result: f32 = mean.variance(&mut samples);
        assert!(result==0.0)
    }

}
