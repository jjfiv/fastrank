use ordered_float::NotNan;
use std::cmp::Ordering;
use std::sync::Arc;

/// Contains code for feature-at-a-time non-differentiable optimization.
pub mod coordinate_ascent;
pub mod dataset;
pub mod evaluators;
pub mod io_helper;
/// Contains code for reading ranklib and libsvm input files.
pub mod libsvm;
pub mod qrel;

pub trait Model: std::fmt::Debug {
    fn score(&self, features: &dataset::Features) -> NotNan<f64>;
}

#[derive(Debug, Clone)]
pub struct WeightedEnsemble(Vec<Scored<Arc<dyn Model>>>);

impl Model for WeightedEnsemble {
    fn score(&self, features: &dataset::Features) -> NotNan<f64> {
        let mut output = 0.0;
        for scored_m in self.0.iter() {
            output += scored_m.score.into_inner() * scored_m.item.score(features).into_inner();
        }
        NotNan::new(output).expect("NaN produced by ensemble.")
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
    fn new(score: f64, item: T) -> Self {
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
