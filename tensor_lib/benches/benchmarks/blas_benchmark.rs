use tensor_lib::{openblas_wrapper::norm2::Norm2, schema::tensor::Tensor, ops::{random::Random}};
use criterion::{
    black_box,
    criterion_group,
    Criterion
};

pub fn norm_benchmark(c: &mut Criterion) {

    let mut tensor = black_box(
        Tensor::create_random(vec![512], Some(-1.0..1.0))
    );

    c.bench_function(
        "norm2 algorithm", |b| {
            b.iter_custom(|iters| {
                let mut time = std::time::Duration::new(0, 0);
                for _ in 0..iters {
                    let instant = std::time::Instant::now();
                    let _value = tensor.norm2();
                    let elapsed = instant.elapsed();
                    time += elapsed;
                }
                time
            })
        });

}


criterion_group!(blas, norm_benchmark);