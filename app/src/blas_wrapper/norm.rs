extern crate libc;
extern crate openblas_src;

use libc::{c_float, c_int};


extern "C" {
    pub fn snrm2_(x: *mut c_int, N: *mut c_float, incx: *mut c_int) -> c_float;
}

#[cfg(test)]
mod blas_tests {
    use super::*;

    #[test]
    fn norm2() {
        unsafe {
            let mut N: i32 = 1;
            let mut x: f32 = 4.0;
            let mut incx: i32 = 1;
            let ret = snrm2_(
                &mut N as *mut _,
                &mut x as *mut _,
                &mut incx as *mut _,
            );
            assert!(ret == 4.0);
        }
    }
}