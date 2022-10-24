#[derive(Debug, Clone)]
pub struct TensorSize{
    data: Vec<usize>,
}

impl TensorSize{
    pub fn new(val: Vec<usize>) -> Self{
        TensorSize { data: val }
    }

    pub fn len(&self) -> usize{
        return self.data.len();
    }

    pub fn assert_index(&self, index: &Vec<usize>){
        if index.len() != self.len(){
            panic!("Index size does not match tensor dimension size");
        }

        for i in 0..self.len(){
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

    pub fn cumulative_size(&self, indx: usize) -> usize{
        let mut ret_val:usize = 1;
        for i in indx..self.len(){
            ret_val *= self.data[i];
        }
        ret_val
    }

    pub fn copy(&self) -> Self{
        TensorSize {data: self.data[..self.data.len()-1].to_vec()}
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