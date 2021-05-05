use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FeatureId(u32);

impl FeatureId {
    pub fn from_index(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn to_index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct InstanceId(u32);
impl InstanceId {
    pub fn from_index(idx: usize) -> Self {
        Self(idx as u32)
    }
    pub fn to_index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug)]
pub struct Scored<T: Clone> {
    pub score: f64,
    pub item: T,
}
impl<T: Clone> Eq for Scored<T> {}
impl<T: Clone> PartialEq for Scored<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl<T: Clone> PartialOrd for Scored<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl<T: Clone> Ord for Scored<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).expect("NaN found!")
    }
}
impl<T: Clone> Scored<T> {
    pub fn new(score: f64, item: T) -> Self {
        Self { score, item }
    }
    pub fn replace_if_better(&mut self, score: f64, item: T) -> bool {
        if score > self.score {
            self.item = item;
            self.score = score;
            return true;
        }
        false
    }
}
