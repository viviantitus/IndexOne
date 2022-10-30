use rand::distributions::uniform::SampleUniform;



pub trait PartialOrdwithSampling: SampleUniform + PartialOrd + Copy{}

impl<T> PartialOrdwithSampling for T where T: SampleUniform + PartialOrd + Copy {}