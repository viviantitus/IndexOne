use gnuplot::{Figure, Color};
use tensorlib::{schema::tensor::Tensor, advanced_ops::kmeans::KMeans, ops::{slice::Slice, random::Random, concat::Concat}};
use std::time::Instant;


 

#[macro_export]
macro_rules! t {
    ($($x:expr),*) => {{
        use tensorlib::schema::index::Indexer;
        use tensorlib::schema::tensor::Tensor;
        use std::ops::{Range, RangeFull, RangeFrom, RangeTo};


        trait IndexEnumConverter {
            fn as_enum_variable(self)->Indexer;
        }
        impl IndexEnumConverter for RangeFrom<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::RangeFrom(self)
            }
        }
        impl IndexEnumConverter for RangeTo<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::RangeTo(self)
            }
        }
        impl IndexEnumConverter for RangeFull{
            fn as_enum_variable(self)->Indexer {
                Indexer::RangeFull(self)
            }
        }
        impl IndexEnumConverter for Range<usize>{
            fn as_enum_variable(self)->Indexer {
                Indexer::Range(self)
            }
        }
        impl IndexEnumConverter for usize{
            fn as_enum_variable(self)->Indexer {
                Indexer::Number(self)
            }
        }
        impl IndexEnumConverter for Tensor<bool>{
            fn as_enum_variable(self)->Indexer {
                Indexer::BoolArray(self)
            }
        }
        vec![$(($x).as_enum_variable()),*] as Vec<Indexer>
        
    }};
}

fn main() {

    let start = Instant::now();

    let mut first_data =  Tensor::<f32>::create_random(vec![10000, 512], Some(10.0..15.0));

    let _ = first_data.train(3, 10, 5, 1e-4);

    let duration = start.elapsed();
    println!("Total time taken to run is {:?}", duration);

    // -----------------------------   Test with multiple clusters -----------------------------

    let mut dataset = vec![];
    dataset.push(Tensor::<f32>::create_random(vec![20, 2], Some(10.0..15.0)));
    dataset.push(Tensor::<f32>::create_random(vec![20, 2], Some(20.0..25.0)));
    dataset.push(Tensor::<f32>::create_random(vec![20, 2], Some(30.0..35.0)));

    let mut data_tensor = dataset.concat();

    let start = Instant::now();
    let results = data_tensor.train(3, 10, 300, 1e-4);
    let duration = start.elapsed();
    println!("Total time taken to run is {:?}", duration);
    

    println!("{:?}", results.2);


    let x_tensor = data_tensor.slice(t![.., 0]);
    let y_tensor = data_tensor.slice(t![.., 1]);


    let x_slice: &[f32] = x_tensor.data.as_slice();
    let y_slice: &[f32] = y_tensor.data.as_slice();


    let mut fg = Figure::new();
    fg.axes2d().points(x_slice, y_slice, &[Color("red")]);
    fg.show().unwrap();
}

