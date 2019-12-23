use crate::instance::FeatureRead;
use crate::{FeatureId, Scored};

use ordered_float::NotNan;

pub trait Model: std::fmt::Debug {
    fn score(&self, features: &dyn FeatureRead) -> NotNan<f64>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelEnum {
    SingleFeature(SingleFeatureModel),
    Linear(DenseLinearRankingModel),
    DecisionTree(TreeNode),
    Ensemble(WeightedEnsemble),
}

impl Model for ModelEnum {
    fn score(&self, features: &dyn FeatureRead) -> NotNan<f64> {
        match self {
            ModelEnum::SingleFeature(m) => m.score(features),
            ModelEnum::Linear(m) => m.score(features),
            ModelEnum::DecisionTree(m) => m.score(features),
            ModelEnum::Ensemble(m) => m.score(features),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SingleFeatureModel {
    pub fid: FeatureId,
    pub dir: f64,
}

impl Model for SingleFeatureModel {
    fn score(&self, features: &dyn FeatureRead) -> NotNan<f64> {
        let val = features.get(self.fid).unwrap_or(0.0);
        NotNan::new(self.dir * val).unwrap()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DenseLinearRankingModel {
    pub weights: Vec<f64>,
}

impl Model for DenseLinearRankingModel {
    fn score(&self, features: &dyn FeatureRead) -> NotNan<f64> {
        NotNan::new(features.dotp(&self.weights)).expect("Model.predict -> NaN")
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum TreeNode {
    FeatureSplit {
        fid: FeatureId,
        split: NotNan<f64>,
        lhs: Box<TreeNode>,
        rhs: Box<TreeNode>,
    },
    LeafNode(NotNan<f64>),
}

impl Model for TreeNode {
    fn score(&self, features: &dyn FeatureRead) -> NotNan<f64> {
        match self {
            TreeNode::LeafNode(score) => score.clone(),
            TreeNode::FeatureSplit {
                fid,
                split,
                lhs,
                rhs,
            } => {
                let fval =
                    NotNan::new(features.get(*fid).unwrap_or(0.0)).expect("NaN in feature eval...");
                if fval <= *split {
                    lhs.score(features)
                } else {
                    rhs.score(features)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedEnsemble {
    weights: Vec<NotNan<f64>>,
    models: Vec<ModelEnum>,
}

impl WeightedEnsemble {
    pub fn new(weighted_models: Vec<Scored<ModelEnum>>) -> Self {
        let mut weights = Vec::new();
        let mut models = Vec::new();
        for sm in weighted_models {
            weights.push(sm.score);
            models.push(sm.item);
        }
        Self { weights, models }
    }
}

impl Model for WeightedEnsemble {
    fn score(&self, features: &dyn FeatureRead) -> NotNan<f64> {
        let mut output = 0.0;
        for (weight, model) in self.weights.iter().zip(self.models.iter()) {
            output += weight.into_inner() * model.score(features).into_inner();
        }
        NotNan::new(output).expect("NaN produced by ensemble.")
    }
}
