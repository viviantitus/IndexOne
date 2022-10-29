extern crate libc;
extern crate openblas_src;

use libc::{c_float, c_double, c_int};
use crate::schema::tensor::Tensor;


extern "C" {
    pub fn snrm2_(x: *mut c_int, N: *mut c_float, incx: *mut c_int) -> c_float;
    pub fn dnrm2_(x: *mut c_int, N: *mut c_double, incx: *mut c_int) -> c_double;
}

trait Norm2<T> {
    fn norm2(&mut self) -> T;
}

impl Norm2<f32> for Tensor<'_, f32> {
    fn norm2(&mut self) -> f32
    { 
        unsafe {
            let mut N: i32 = i32::try_from(self.data.len()).unwrap();
            let mut incx: i32 = 1;
            let ret = snrm2_(
                &mut N as *mut _,
                self.data.as_mut_ptr(),
                &mut incx as *mut _,
            );
            ret
        }
    }
}

impl Norm2<f64> for Tensor<'_, f64> {
    fn norm2(&mut self) -> f64
    { 
        unsafe {
            let mut N: i32 = i32::try_from(self.data.len()).unwrap();
            let mut incx: i32 = 1;
            let ret = dnrm2_(
                &mut N as *mut _,
                self.data.as_mut_ptr(),
                &mut incx as *mut _,
            );
            ret
        }
    }
}


#[cfg(test)]
mod blas_tests {
    use super::*;

    #[test]
    fn snorm2() {
        unsafe {
            let mut N: i32 = 2;
            let mut dat = [4.0, 3.0];
            let x: &mut [f32] = dat.as_mut_slice();
            let mut incx: i32 = 1;
            let ret = snrm2_(
                &mut N as *mut _,
                x.as_mut_ptr(),
                &mut incx as *mut _,
            );
            assert!(ret == 5.0);
        }
    }

    #[test]
    fn dnorm2() {
        unsafe {
            let mut N: i32 = 2;
            let mut dat: [f64; 2] = [4.0, 3.0];
            let x: &mut [f64] = dat.as_mut_slice();
            let mut incx: i32 = 1;
            let ret = dnrm2_(
                &mut N as *mut _,
                x.as_mut_ptr(),
                &mut incx as *mut _,
            );
            assert!(ret == 5.0);
        }
    }
}