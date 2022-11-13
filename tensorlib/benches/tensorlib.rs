use benchmarks::{blas_benchmark, ops_benchmark};
use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    blas_benchmark::blas,
    ops_benchmark::ops
}