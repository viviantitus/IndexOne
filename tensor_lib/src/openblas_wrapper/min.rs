extern crate libc;
extern crate openblas_src;
extern crate num_traits;
use libc::{c_float, c_double, c_int, size_t};
use crate::schema::tensor::Tensor;
use crate::ops::convert::Convert;


extern "C" {
    pub fn isamin_(x: *mut c_int, N: *mut c_float, incx: *mut c_int) -> size_t;
    pub fn idamin_(x: *mut c_int, N: *mut c_double, incx: *mut c_int) -> size_t;
}

pub trait Min<T>{
    fn min(&mut self, dim: Option<usize>) -> Tensor<'_, usize>;
    fn min_blas(array: &mut [T]) -> usize;
}

macro_rules! min_impl {
    ( $( $t:ty ),* ; $( $j:ident ),* ) => ($(
        impl Min<$t> for Tensor<'_, $t> {
            fn min(&mut self, dim: Option<usize>) -> Tensor<'_, usize>
            { 
                if dim.is_some() && dim.unwrap() >= self.dim(){
                    panic!("Min: Dimensions of tensor should be one");
                }

                match dim{
                    Some(x) => match x{
                        0 => {
                            let mut min_values = vec![];
                            let size_of_slice: usize = self.size.total_elements() / self.size[0];
                            for i in 0..self.size[0]{
                                min_values.push(Self::min_blas(&mut self.data[i*size_of_slice..(i+1)*size_of_slice]));
                            }
                            min_values.convert_to_tensor()
                        },
                        _ => panic!("not implemented")
                    },
                    None => Self::min_blas(self.data).convert_to_tensor()
                }
            }

            fn min_blas(array: &mut [$t]) -> usize
            {

                unsafe {
                    let mut n: i32 = i32::try_from(array.len()).unwrap();
                    let mut incx: i32 = 1;
                    let ret = $j(
                        &mut n as *mut _,
                        array.as_mut_ptr(),
                        &mut incx as *mut _,
                    );
                    ret-1
                }
            }
        }
    )*)
}

min_impl! { f32, f64; isamin_, idamin_ }


#[cfg(test)]
mod blas_tests {
    use crate::schema::size::TensorSize;

    use super::*;

    #[test]
    fn smin() {
        let mut data = [8.0, 10.0, 3.0, 1.0, 10.0];
        let data_len = data.len();
        let mut tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![data_len]));

        let result = tensor.min(None);
        assert!(result[0]==3)
    }

    #[test]
    fn dmin() {
        unsafe {
            let mut n: i32 = 2;
            let mut dat: [f64; 2] = [3.0, 0.0];
            let x: &mut [f64] = dat.as_mut_slice();
            let mut incx: i32 = 1;
            let ret = idamin_(
                &mut n as *mut _,
                x.as_mut_ptr(),
                &mut incx as *mut _,
            );
            assert!(ret == 2);
        }
    }

    #[test]
    fn smin2() {
        let mut data = [8.0, 10.0, 3.0, 1.0, 10.0, 5.0];
        let mut tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![2, 3]));

        let result = tensor.min(Some(0));
        assert!(result[0]==2 && result[1] == 0)
    }
}