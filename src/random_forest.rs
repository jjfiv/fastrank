use crate::dataset::*;
use crate::evaluators::Evaluator;
use crate::model::{Model, WeightedEnsemble};
use crate::Scored;
use ordered_float::NotNan;
use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct RandomForestParams {
    seed: u64,
    quiet: bool,
    instance_sampling_rate: f64,
    feature_sampling_rate: f64,
}

impl Default for RandomForestParams {
    fn default() -> Self {
        Self {
            seed: thread_rng().next_u64(),
            quiet: false,
            instance_sampling_rate: 0.5,
            feature_sampling_rate: 0.25,
        }
    }
}

pub enum TreeNode {
    FeatureSplit {
        fid: u32,
        split: f64,
        lhs: Box<TreeNode>,
        rhs: Box<TreeNode>,
    },
    LeafNode(f64),
}

impl TreeNode {
    pub fn score(&self, features: &Features) -> f64 {
        match self {
            TreeNode::LeafNode(score) => *score,
            TreeNode::FeatureSplit {
                fid,
                split,
                lhs,
                rhs,
            } => {
                if features.get(*fid).unwrap_or(0.0) < *split {
                    lhs.score(features)
                } else {
                    rhs.score(features)
                }
            }
        }
    }
}
