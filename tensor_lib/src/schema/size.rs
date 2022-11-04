use std::ops::Index;

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

    pub fn slice(&self, slice_vec: &Vec<Indexer>) -> Self{
        let mut data : Vec<usize> = vec![];
        for i in 0..self.data.len(){
            data.push(self.match_slice(slice_vec[i].clone(), i))
        }
        let cumulative = Self::create_cumulative(&data);
        TensorSize { data: data, cumulative: cumulative }    
    }

    pub fn create_slicevec(&self, slice: &Indexer, dim: usize) -> Vec<Indexer>{
        let mut slice_vec : Vec<Indexer> = vec![];
        for i in 0..self.data.len(){
            slice_vec.push(if dim == i {slice.clone()} else {Indexer::RangeFull(..)});
        }
        slice_vec
    }

    fn match_slice(&self, slice: Indexer, dim: usize) -> usize{
        match slice.clone(){
            Indexer::Number(_) => 1,
            Indexer::Range(x) => x.end - x.start,
            Indexer::RangeFrom(x) => self[dim] - x.start,
            Indexer::RangeTo(x) => x.end,
            Indexer::RangeFull(_) => self[dim]
        }
    }

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

    pub fn remove_dim(&mut self, dim_index: usize) -> TensorSize{
        let mut new_data = self.data.clone();
        new_data.remove(dim_index);
        Self::new(new_data)
    }

    pub fn push_front(&mut self, val: usize) -> TensorSize{
        let mut data = self.data.clone();
        data.insert(0, val);
        Self::new(data)
    }

    fn assert_index(&self, index: &Vec<Indexer>){
        if index.len() != self.dim(){
            panic!("Index size does not match tensor dimension size");
        }

        for i in 0..self.dim(){
            let assert_index = match index[i].clone() {
                Indexer::Number(x) => x >= self.data[i],
                Indexer::Range(x) => x.start >= self.data[i] || x.end > self.data[i] || x.end <= x.start,
                Indexer::RangeFrom(x) => x.start >= self.data[i],
                Indexer::RangeTo(x) => x.end > self.data[i],
                Indexer::RangeFull(_) => false
            };
            if assert_index{
                panic!("Index of Dim {} out of bounds", i)
            }
        }
    }

    // #TODO: Save this data for faster access
    pub fn total_elements(&self) -> usize{
        let mut size_of_ndarray: usize = 1;
        for i in 0..self.data.len(){
            size_of_ndarray *= self.data[i];
        }
        size_of_ndarray
    }

    pub fn calc_seq_index(&self, index: Vec<Indexer>) -> usize{
        self.assert_index(&index);

        let mut data_index = 0;

        for i in 0..self.data.len(){
            match index[i] {
                Indexer::Number(x) => {
                    if i == self.data.len() -1 {
                        data_index += x;
                    }
                    else{
                        data_index += self.cumulative[i] * x;
                    }
                }
                Indexer::Range(_) => panic!("Cannot implement sequence calculation for slices"),
                Indexer::RangeFrom(_) => panic!("Cannot implement sequence calculation for slices"),
                Indexer::RangeTo(_) => panic!("Cannot implement sequence calculation for slices"),
                Indexer::RangeFull(_) => panic!("Cannot implement sequence calculation for slices")
            }
            
        }
        data_index
    }

    #[inline(always)]
    pub fn is_seqindex_within_slice(&self, slice_vec: Vec<Indexer>, seq_index: usize) -> bool{
        self.assert_index(&slice_vec);
        let mut offset: usize = 0;
        for indx_ in 0..self.data.len(){
            let dim_index: usize = 
                if indx_ == self.data.len()-1 {
                    usize::from(seq_index%self.cumulative[indx_-1])
                } else {
                    usize::from((seq_index - offset)/self.cumulative[indx_])
                };
            let cond = match slice_vec[indx_].clone() {
                Indexer::Number(x) => dim_index == x,
                Indexer::Range(x) => dim_index >= x.start && dim_index < x.end,
                Indexer::RangeFrom(x) => dim_index >= x.start,
                Indexer::RangeTo(x) => dim_index < x.end,
                Indexer::RangeFull(_) => true
            };
            if !cond{
                return false;
            }
            offset += if indx_!= self.data.len()-1 {dim_index * self.cumulative[indx_]} else {0};
        }
        return true
    }
}

impl Index<usize> for TensorSize {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
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



#[cfg(test)]
mod size_tests {
    use crate::schema::size::TensorSize;
    use crate::t;

    #[test]
    fn test_is_within_range() {
        let size = TensorSize::new(vec![5, 10, 2, 5]);
        let cond = size.is_seqindex_within_slice(t![3, 3, 1, 4], 339);
        assert!(cond);
    }

    #[test]
    #[should_panic]
    fn test_assert_index() {
        let size = TensorSize::new(vec![5, 10, 2, 5]);
        size.is_seqindex_within_slice(t![3, 3, 1, 6], 339);
    }
}