use tensor_lib::{openblas_wrapper::norm2::Norm2, schema::tensor::Tensor, ops::{random::Random}};
use criterion::{
    black_box,
    criterion_group,
    Criterion
};

pub fn norm2_benchmark(c: &mut Criterion) {
    let mut tensor = black_box(
        Tensor::create_random(vec![2], Some(-1.0..1.0))
    );

    c.bench_function(
        "norm2 algorithm", 
        |b| b.iter(|| tensor.norm2())
    );
}


criterion_group!(benches, norm2_benchmark);