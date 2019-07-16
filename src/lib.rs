use ordered_float::NotNan;

/// Contains code for feature-at-a-time non-differentiable optimization.
pub mod coordinate_ascent;
/// Contains code for reading ranklib and libsvm input files.
pub mod libsvm;
pub mod dataset;
pub mod evaluators;

pub mod io_helper;

pub trait Model : std::fmt::Debug {
    fn score(&self, features: &dataset::Features) -> f64;
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Scored<T: Clone> {
    pub score: NotNan<f64>,
    pub item: T,
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
    fn use_best(&mut self, other: &Scored<T>) {
        if self.score > other.score {
            return;
        } else {
            self.score = other.score;
            self.item = other.item.clone();
        }
    }
}
