use tensor_lib::{schema::tensor::Tensor, ops::{random::Random, subtract::Subtract}};
use criterion::{
    black_box,
    criterion_group,
    Criterion
};

fn subtract_benchmark(c: &mut Criterion) {
    let t1 = black_box(
        Tensor::<f32>::create_random(vec![10], None)
    );

    let t2 = black_box(
        Tensor::<f32>::create_random(vec![10], None)
    );

    c.bench_function(
        "subtract algorithm", 
        |b| b.iter(|| t1.sub(&t2))
    );
}

fn create_rand_benchmark(c: &mut Criterion) {

    c.bench_function(
        "random tensor creation", 
        |b| b.iter(|| Tensor::create_random(vec![10], Some(-1.0..1.0)))
    );
}


criterion_group!(benches, subtract_benchmark, create_rand_benchmark);