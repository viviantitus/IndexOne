extern crate libc;
extern crate openblas_src;
extern crate num_traits;
use libc::{c_float, c_double, c_int};
use crate::schema::tensor::Tensor;

extern "C" {
    pub fn sasum_(x: *mut c_int, N: *mut c_float, incx: *mut c_int) -> c_float;
    pub fn dasum_(x: *mut c_int, N: *mut c_double, incx: *mut c_int) -> c_double;
}

pub trait Norm<T>{
    fn norm(&mut self) -> T;
}

impl Norm<f32> for Tensor<'_, f32> {
    fn norm(&mut self) -> f32
    { 
        if self.dim() != 1{
            panic!("Norm: Dimensions of tensor should be one");
        }

        unsafe {
            let mut n: i32 = i32::try_from(self.data.len()).unwrap();
            let mut incx: i32 = 1;
            let ret = sasum_(
                &mut n as *mut _,
                self.data.as_mut_ptr(),
                &mut incx as *mut _,
            );
            ret
        }
    }
}

impl Norm<f64> for Tensor<'_, f64> {
    fn norm(&mut self) -> f64
    { 
        if self.dim() != 1{
            panic!("Norm: Dimensions of tensor should be one");
        }
        
        unsafe {
            let mut n: i32 = i32::try_from(self.data.len()).unwrap();
            let mut incx: i32 = 1;
            let ret = dasum_(
                &mut n as *mut _,
                self.data.as_mut_ptr(),
                &mut incx as *mut _,
            );
            ret
        }
    }
}


#[cfg(test)]
mod blas_tests {
    use crate::schema::size::TensorSize;

    use super::*;

    #[test]
    fn snorm() {
        let mut data = [10.0, 10.0, 10.0, 10.0, 10.0];
        let data_len = data.len();
        let mut tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![data_len]));

        let result: f32 = tensor.norm();
        assert!(result==50.0)
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