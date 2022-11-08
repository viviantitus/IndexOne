extern crate libc;
extern crate openblas_src;
extern crate num_traits;
use libc::{c_float, c_double, c_int};
use crate::schema::tensor::Tensor;
use crate::ops::convert::Convert;


extern "C" {
    pub fn sasum_(x: *mut c_int, N: *mut c_float, incx: *mut c_int) -> c_float;
    pub fn dasum_(x: *mut c_int, N: *mut c_double, incx: *mut c_int) -> c_double;
}


pub trait Norm<'a, T>{
    fn norm(&mut self, dim: Option<usize>) -> Self;
    fn norm_blas(array: &mut [T], n: usize, incx: usize, offset: usize) -> T;
}

macro_rules! norm_impl {
    ( $( $t:ty ),* ; $( $j:ident ),* ) => ($(
        impl<'a> Norm<'a, $t> for Tensor<'a, $t> {
            fn norm(&mut self, dim: Option<usize>) -> Self
            { 
                if dim.is_some() && dim.unwrap() >= self.dim(){
                    panic!("Min: Dimensions of tensor should be equal");
                }
                assert!(self.dim() <= 2);

                match dim{
                    Some(x) => match x{
                        0 => {
                            let mut min_values = vec![];
                            let size_of_slice: usize = self.size.total_elements() / self.size[0];
                            for i in 0..self.size[0]{
                                min_values.push(Self::norm_blas(self.data, size_of_slice, 1, i*size_of_slice));
                            }
                            min_values.convert_to_tensor()
                        },
                        1 => {
                            let mut min_values = vec![];
                            let size_of_slice: usize = self.size.total_elements() / self.size[0];
                            for i in 0..self.size[1]{
                                min_values.push(Self::norm_blas(self.data, self.size[0], size_of_slice, i));
                            }
                            min_values.convert_to_tensor()
                        }
                        _ => panic!("not implemented")
                    },
                    None => Self::norm_blas(self.data, self.size.total_elements(), 1, 0).convert_to_tensor()
                }
            }

            fn norm_blas(array: &mut [$t], n: usize, incx: usize, offset: usize) -> $t
            {

                unsafe {
                    let mut n: i32 = i32::try_from(n).unwrap();
                    let mut incx: i32 = i32::try_from(incx).unwrap();
                    let ret = $j(
                        &mut n as *mut _,
                        array.as_mut_ptr().add(offset),
                        &mut incx as *mut _,
                    );
                    ret
                }
            }
        }
    )*)
}

norm_impl! { f32, f64; sasum_, dasum_ }


#[cfg(test)]
mod blas_tests {
    use crate::schema::size::TensorSize;

    use super::*;

    #[test]
    fn snorm() {
        let mut data = [10.0, 10.0, 10.0, 10.0, 10.0];
        let data_len = data.len();
        let mut tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![data_len]));

        let result = tensor.norm(None);
        assert!(result[0]==50.0)
    }

    #[test]
    fn dnorm() {
        unsafe {
            let mut n: i32 = 2;
            let mut dat: [f64; 2] = [3.0, 3.0];
            let x: &mut [f64] = dat.as_mut_slice();
            let mut incx: i32 = 1;
            let ret = dasum_(
                &mut n as *mut _,
                x.as_mut_ptr(),
                &mut incx as *mut _,
            );
            assert!(ret == 6.0);
        }
    }
}