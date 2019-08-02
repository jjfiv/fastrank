use crate::instance::Features;
use crate::Scored;

use ordered_float::NotNan;
use std::sync::Arc;

pub trait Model: std::fmt::Debug {
    fn score(&self, features: &Features) -> NotNan<f64>;
}

#[derive(Debug, Clone)]
pub struct WeightedEnsemble {
    weighted_models: Vec<Scored<Arc<dyn Model>>>,
}

impl WeightedEnsemble {
    pub fn new(weighted_models: Vec<Scored<Arc<dyn Model>>>) -> Self {
        Self { weighted_models }
    }
}

impl Model for WeightedEnsemble {
    fn score(&self, features: &Features) -> NotNan<f64> {
        let mut output = 0.0;
        for scored_m in self.weighted_models.iter() {
            output += scored_m.score.into_inner() * scored_m.item.score(features).into_inner();
        }
        NotNan::new(output).expect("NaN produced by ensemble.")
    }
}
