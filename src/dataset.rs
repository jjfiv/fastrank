use std::collections::HashMap;
use std::collections::HashSet;
use ordered_float::NotNan;
use std::f64;
use crate::libsvm;
use crate::Model;
use crate::evaluators::{RankedInstance, Evaluator};

pub enum Features {
    Dense32(Vec<f32>),
    Dense64(Vec<f64>),
    /// Sparse 32-bit representation; must be sorted!
    Sparse32(Vec<(u32, f32)>),
    /// Sparse 64-bit representation; must be sorted!
    Sparse64(Vec<(u32, f64)>),
}

impl Features {
    pub fn get(&self, idx: u32) -> Option<f64> {
        match self {
            Features::Dense32(arr) => Some(f64::from(arr[idx as usize])),
            Features::Dense64(arr) => Some(arr[idx as usize]),
            Features::Sparse32(features) => {
                for (fidx, val) in features.iter() {
                    if *fidx == idx {
                        return Some(f64::from(*val));
                    } else if *fidx > idx {
                        break;
                    }
                }
                None
            }
            Features::Sparse64(features) => {
                for (fidx, val) in features.iter() {
                    if *fidx == idx {
                        return Some(*val);
                    } else if *fidx > idx {
                        break;
                    }
                }
                None
            }
        }
    }
    pub fn ids(&self) -> Vec<u32> {
        let mut features: Vec<u32> = Vec::new();
        match self {
            Features::Dense32(arr) => features.extend(0..(arr.len() as u32)),
            Features::Dense64(arr) => features.extend(0..(arr.len() as u32)),
            Features::Sparse32(arr) => features.extend(arr.iter().map(|(idx, _)| *idx)),
            Features::Sparse64(arr) => features.extend(arr.iter().map(|(idx, _)| *idx)),
        }
        features
    }
}

pub struct TrainingInstance {
    pub gain: NotNan<f32>,
    pub qid: String,
    pub features: Features,
}

impl TrainingInstance {
    pub fn try_new(libsvm: libsvm::Instance) -> Result<TrainingInstance, &'static str> {
        Ok(TrainingInstance {
            gain: libsvm.label,
            qid: libsvm.query.ok_or("Missing qid")?,
            features: Features::Sparse32(
                libsvm
                    .features
                    .into_iter()
                    .map(|f| (f.idx, f.value))
                    .collect(),
            ),
        })
    }
    pub fn is_relevant(&self) -> bool {
        self.gain.into_inner() > 0.0
    }
}

pub struct RankingDataset {
    pub instances: Vec<TrainingInstance>,
    pub features: Vec<u32>,
    pub n_dim: u32,
    pub data_by_query: HashMap<String, Vec<usize>>,
}

impl RankingDataset {
    pub fn evaluate_mean(&self, model: &Model, evaluator: &Evaluator) -> f64 {
        let worst_prediction = NotNan::new(f64::MIN).unwrap();
        let mut sum_score = 0.0;
        let mut num_scores = self.data_by_query.len() as f64;
        for (qid, docs) in self.data_by_query.iter() {
            // Predict for every document:
            let mut ranked_list: Vec<_> = docs.iter().cloned().map(|index| {
                let prediction = NotNan::new(model.score(&self.instances[index].features));
                RankedInstance::new(prediction.unwrap_or(worst_prediction), self.instances[index].gain, index as u32)
            }).collect();
            // Sort largest to smallest:
            ranked_list.sort_unstable();
            sum_score += evaluator.score(&qid, &ranked_list);
        }
        sum_score / num_scores
    }

    pub fn import(data: Vec<libsvm::Instance>) -> Result<Self, &'static str> {
        let instances: Result<Vec<_>, _> = data
            .into_iter()
            .map(|i| TrainingInstance::try_new(i))
            .collect();
        Ok(Self::new(instances?))
    }
    pub fn new(data: Vec<TrainingInstance>) -> Self {
        // Collect features that are actually present.
        let mut features: HashSet<u32> = HashSet::new();
        // Collect training instances by the query.
        let mut data_by_query: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, inst) in data.iter().enumerate() {
            data_by_query
                .entry(inst.qid.clone())
                .or_insert(Vec::new())
                .push(i);
            features.extend(inst.features.ids());
        }
        
        // Get a sorted list of active features in this dataset.
        let mut features: Vec<u32> = features.iter().cloned().collect();
        features.sort_unstable();

        // Calculate the number of features, including any missing ones, so we can have a dense linear model.
        let n_dim = features
            .iter()
            .cloned()
            .max()
            .expect("No features defined!")
            + 1;

        Self { instances: data, features, n_dim, data_by_query }
    }
}