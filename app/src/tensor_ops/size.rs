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
        TensorSize { data: val, cumulative: cumulative }    }

    fn create_cumulative(val: &Vec<usize>) -> Vec<usize>{
        let mut cumulative: Vec<usize> = vec![];
        let mut ret_val:usize = 1;
        for i in (1..val.len()).rev(){
            ret_val *= val[i];
            cumulative.push(ret_val.clone());
        }
        cumulative.reverse();
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
        let mut offset: usize = 0;
        for indx_ in 0..sliceindex.len(){
            let dim_index: usize = 
                if indx_ == sliceindex.len()-1 {
                    usize::from(seq_index%self.cumulative[indx_-1])
                } else {
                    usize::from((seq_index - offset)/self.cumulative[indx_])
                };
            let cond = match sliceindex[indx_].clone() {
                Indexer::Number(x) => dim_index == x,
                Indexer::SliceRange(x) => dim_index >= x.start && dim_index < x.end,
            };
            if !cond{
                return false;
            }
            offset += if indx_!= sliceindex.len()-1 {dim_index * self.cumulative[indx_]} else {0};
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