use ordered_float::NotNan;
use std::cmp::Ordering;

#[macro_use]
extern crate serde_derive;

pub mod randutil;
/// Contains code for feature-at-a-time non-differentiable optimization.
pub mod coordinate_ascent;
pub mod dataset;
pub mod dense_dataset;
pub mod evaluators;
pub mod instance;
/// Contains code for reading compressed files based on their extension.
pub mod io_helper;
/// Contains code for reading ranklib and libsvm input files.
pub mod libsvm;
pub mod model;
pub mod normalizers;
pub mod qrel;
pub mod sampling;

pub mod json_api;

pub mod random_forest;
/// Streaming computation of statistics.
pub mod stats;

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
    pub score: NotNan<f64>,
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
        Some(self.score.cmp(&other.score))
    }
}
impl<T: Clone> Ord for Scored<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}
impl<T: Clone> Scored<T> {
    pub fn new(score: f64, item: T) -> Self {
        Self {
            score: NotNan::new(score).expect("NaN found!"),
            item,
        }
    }
    fn replace_if_better(&mut self, score: f64, item: T) -> bool {
        if let Ok(score) = NotNan::new(score) {
            if score > self.score {
                self.item = item;
                self.score = score;
                return true;
            }
        }
        false
    }
}
