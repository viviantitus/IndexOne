use tensorlib::{schema::tensor::Tensor, ops::{random::Random, subtract::Subtract}};
use criterion::{
    black_box,
    criterion_group,
    Criterion
};

fn subtract_benchmark(c: &mut Criterion) {
    let t1 = black_box(
        Tensor::<f32>::create_random(vec![100], None)
    );

    let t2 = black_box(
        Tensor::<f32>::create_random(vec![100], None)
    );


    c.bench_function(
        "subtract algorithm", |b| {
            b.iter_custom(|iters| {
                let mut time = std::time::Duration::new(0, 0);
                for _ in 0..iters {
                    let instant = std::time::Instant::now();
                    let _value = t1.sub(&t2);
                    let elapsed = instant.elapsed();
                    time += elapsed;
                }
                time
            })
        });
    
}

fn create_rand_benchmark(c: &mut Criterion) {

    c.bench_function(
        "create rand tensor", |b| {
            b.iter_custom(|iters| {
                let mut time = std::time::Duration::new(0, 0);
                for _ in 0..iters {
                    let instant = std::time::Instant::now();
                    let _value = Tensor::create_random(vec![1], Some(-1.0..1.0));
                    let elapsed = instant.elapsed();
                    time += elapsed;
                }
                time
            })
        });
}




criterion_group!(ops, create_rand_benchmark, subtract_benchmark);