#[derive(Debug, Clone, Copy)]
pub struct TensorSize<'a>{
    data: &'a [usize],
}

impl<'a> TensorSize<'a>{
    pub fn new(val: &'a [usize]) -> Self{
        TensorSize { data: val }
    }

    pub fn len(&self) -> usize{
        return self.data.len();
    }

    pub fn assert_index(&self, index: &[usize]){
        if index.len() != self.len(){
            panic!("Index size does not match tensor dimension size");
        }

        for i in 0..self.len(){
            if index[i] >= self.data[i]{
                panic!("Index of Dim {} out of bounds", i)
            }
        }
    }

    pub fn cumulative_size(&self, indx: usize) -> usize{
        let mut ret_val:usize = 1;
        for i in indx..self.len(){
            ret_val *= self.data[i];
        }
        ret_val
    }
}

impl<'a> PartialEq for TensorSize<'a>{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len(){
            return false;
        }
        for i in 0..self.len(){
            if self.data[i] != other.data[i]{
                return false;
            }
        }
        return true;
    }
}