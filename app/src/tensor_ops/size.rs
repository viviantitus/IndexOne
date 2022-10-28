use super::index::Indexer;

#[derive(Debug, Clone)]
pub struct TensorSize{
    data: Vec<usize>,
    cumulative: Vec<usize>
}

impl TensorSize{
    pub fn new(val: Vec<usize>) -> Self{
        let cumulative = Self::create_cumulative(&val);
        TensorSize { data: val, cumulative: cumulative }
    }

    pub fn create_with_sliceindex(sliceindex: &Vec<Indexer>) -> Self{

        let mut val : Vec<usize> = vec![];
        for indx_ in 0..sliceindex.len(){
            let data = match sliceindex[indx_].clone(){
                Indexer::Number(_) => 1,
                Indexer::SliceRange(x) => x.end - x.start
            };
            val.push(data)
        }
        let cumulative = Self::create_cumulative(&val);
        TensorSize { data: val, cumulative: cumulative }
    }

    fn create_cumulative(val: &Vec<usize>) -> Vec<usize>{
        let mut cumulative: Vec<usize> = vec![];
        let mut ret_val:usize = 1;
        for i in 1..val.len(){
            ret_val *= val[i];
            cumulative.push(ret_val.clone());
        }
        cumulative
    }

    pub fn dim(&self) -> usize{
        return self.data.len();
    }

    fn assert_index(&self, index: &Vec<usize>){
        if index.len() != self.dim(){
            panic!("Index size does not match tensor dimension size");
        }

        for i in 0..self.dim(){
            if index[i] >= self.data[i]{
                panic!("Index of Dim {} out of bounds", i)
            }
        }
    }

    pub fn total_elements(&self) -> usize{
        let mut size_of_ndarray: usize = 1;
        for i in 0..self.data.len(){
            size_of_ndarray *= self.data[i];
        }
        size_of_ndarray
    }

    pub fn calc_seq_index(&self, index: Vec<usize>) -> usize{
        self.assert_index(&index);

        let mut data_index = 0;

        for i in 0..self.data.len(){
            if i == self.data.len() -1 {
                data_index += index[i];
            }
            else{
                data_index += self.cumulative[i] * index[i];
            }
        }
        data_index
    }

    #[inline(always)]
    pub fn is_within_sliceindex(&self, sliceindex: &Vec<Indexer>, seq_index: usize) -> bool{
        for indx_ in 0..sliceindex.len(){
            let denom = if indx_ == sliceindex.len()-1 {1} else {self.cumulative[indx_]};
            let cond = match sliceindex[indx_].clone() {
                Indexer::Number(x) => usize::from(seq_index/denom) == usize::from(x),
                Indexer::SliceRange(x) => usize::from(seq_index/denom) >= x.start && usize::from(seq_index/denom) < x.end,
            };
            if indx_ > 0{
                println!("{:?} {:?}", denom, sliceindex[indx_]);
            }
            if !cond{
                return false;
            }
        }
        return true
    }

}

impl PartialEq for TensorSize{
    fn eq(&self, other: &Self) -> bool {
        if self.data == other.data{
            return true;
        }
        return false;
    }
}