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

pub trait Min<'a, T>{
    fn min(self, dim: Option<usize>) -> Tensor<'a, usize>;
    fn min_blas(array: &mut [T], n: usize, incx: usize, offset: usize) -> usize;
    fn min_for_u8(self, dim: Option<usize>) -> Tensor<'a, u8>;
    fn min_blas_for_u8(array: &mut [T], n: usize, incx: usize, offset: usize) -> u8;
}

macro_rules! min_impl {
    ( $( $t:ty ),* ; $( $j:ident ),* ) => ($(
        impl<'a> Min<'a, $t> for Tensor<'a, $t> {
            fn min(self, dim: Option<usize>) -> Tensor<'a, usize>
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
                                min_values.push(Self::min_blas(self.data, size_of_slice, 1, i*size_of_slice));
                            }
                            min_values.convert_to_tensor()
                        },
                        1 => {
                            let mut min_values = vec![];
                            let size_of_slice: usize = self.size.total_elements() / self.size[0];
                            for i in 0..self.size[1]{
                                min_values.push(Self::min_blas(self.data, self.size[0], size_of_slice, i));
                            }
                            min_values.convert_to_tensor()
                        }
                        _ => panic!("not implemented")
                    },
                    None => Self::min_blas(self.data, self.size.total_elements(), 1, 0).convert_to_tensor()
                }
            }

            fn min_blas(array: &mut [$t], n: usize, incx: usize, offset: usize) -> usize
            {

                unsafe {
                    let mut n: i32 = i32::try_from(n).unwrap();
                    let mut incx: i32 = i32::try_from(incx).unwrap();
                    let ret = $j(
                        &mut n as *mut _,
                        array.as_mut_ptr().add(offset),
                        &mut incx as *mut _,
                    );
                    ret-1
                }
            }

            fn min_for_u8(self, dim: Option<usize>) -> Tensor<'a, u8>
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
                                min_values.push(Self::min_blas_for_u8(self.data, size_of_slice, 1, i*size_of_slice));
                            }
                            min_values.convert_to_tensor()
                        },
                        1 => {
                            let mut min_values = vec![];
                            let size_of_slice: usize = self.size.total_elements() / self.size[0];
                            for i in 0..self.size[1]{
                                min_values.push(Self::min_blas_for_u8(self.data, self.size[0], size_of_slice, i));
                            }
                            min_values.convert_to_tensor()
                        }
                        _ => panic!("not implemented")
                    },
                    None => Self::min_blas_for_u8(self.data, self.size.total_elements(), 1, 0).convert_to_tensor()
                }
            }

            fn min_blas_for_u8(array: &mut [$t], n: usize, incx: usize, offset: usize) -> u8
            {

                unsafe {
                    let mut n: i32 = i32::try_from(n).unwrap();
                    let mut incx: i32 = i32::try_from(incx).unwrap();
                    let ret = $j(
                        &mut n as *mut _,
                        array.as_mut_ptr().add(offset),
                        &mut incx as *mut _,
                    );
                    u8::try_from(ret-1).unwrap()
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
        let tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![data_len]));

        let result = tensor.min(None);
        assert!(result[0]==3)
    }

    #[test]
    fn dmin() {
        unsafe {
            let mut dat: [f32; 10] = [1.0, 1.0, 0.0, 0.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];
            let mut n: i32 = 5;
            let x: &mut [f32] = dat.as_mut_slice();
            let mut incx: i32 = 2;
            let ret = isamin_(
                &mut n as *mut _,
                x.as_mut_ptr().add(1),
                &mut incx as *mut _,
            );
            assert!(ret == 2);
        }
    }

    #[test]
    fn smin_dim0() {
        let mut data = [8.0, 10.0, 3.0, 1.0, 10.0, 5.0];
        let tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![2, 3]));

        let result = tensor.min(Some(0));
        assert!(result[0]==2 && result[1] == 0)
    }

    #[test]
    fn smin_dim1() {
        let mut data = [8.0, 10.0, 3.0, 1.0, 9.0, 2.0];
        let tensor = Tensor::create_with_data_copy(data.as_mut_slice(), TensorSize::new(vec![2, 3]));

        let result = tensor.min(Some(1));
        assert!(result[0]==1 && result[1] == 1 && result[2] == 1)
    }
}