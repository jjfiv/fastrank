use crate::libsvm;
use crate::normalizers::Normalizer;
use crate::stats::StreamingStats;
use crate::FeatureId;
use ordered_float::NotNan;
use std::collections::HashMap;

pub enum Features {
    Dense32(Vec<f32>),
    /// Sparse 32-bit representation; must be sorted!
    Sparse32(Vec<(FeatureId, f32)>),
}

impl Features {
    pub fn get(&self, idx: FeatureId) -> Option<f64> {
        match self {
            Features::Dense32(arr) => arr.get(idx.to_index()).map(|val| f64::from(*val)),
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
        }
    }
    pub fn ids(&self) -> Vec<FeatureId> {
        let mut features: Vec<FeatureId> = Vec::new();
        match self {
            Features::Dense32(arr) => {
                features.extend((0..arr.len()).map(|idx| FeatureId::from_index(idx)))
            }
            Features::Sparse32(arr) => features.extend(arr.iter().map(|(idx, _)| *idx)),
        }
        features
    }
    pub fn update_stats(&self, per_feature_stats: &mut HashMap<FeatureId, StreamingStats>) {
        for (fid, stats) in per_feature_stats.iter_mut() {
            if let Some(x) = self.get(*fid) {
                stats.push(x);
            }
            // Expliticly skip missing; so as not to make it part of normalization.
        }
    }
    pub fn apply_normalization(&mut self, normalizer: &Normalizer) {
        match self {
            Features::Dense32(arr) => {
                for (fid, val) in arr.iter_mut().enumerate() {
                    *val = normalizer.normalize(FeatureId::from_index(fid), *val);
                }
            }
            Features::Sparse32(arr) => {
                for (fid, val) in arr.iter_mut() {
                    *val = normalizer.normalize(*fid, *val);
                }
            }
        }
    }
}

pub struct TrainingInstance {
    pub gain: NotNan<f32>,
    pub qid: String,
    pub features: Features,
}

impl TrainingInstance {
    pub fn new(gain: NotNan<f32>, qid: String, features: Features) -> Self {
        Self {
            gain,
            qid,
            features,
        }
    }
    pub fn try_new(libsvm: libsvm::Instance) -> Result<TrainingInstance, &'static str> {
        // Convert features to dense representation if it's worthwhile.
        let max_feature = libsvm.features.iter().map(|f| f.idx).max().unwrap_or(1);
        let density = (libsvm.features.len() as f64) / (max_feature as f64);
        let features = if density >= 0.5 {
            let mut dense = vec![0_f32; (max_feature + 1) as usize];
            for f in libsvm.features.iter() {
                dense[f.idx as usize] = f.value;
            }
            Features::Dense32(dense)
        } else {
            Features::Sparse32(
                libsvm
                    .features
                    .into_iter()
                    .map(|f| (FeatureId::from_index(f.idx as usize), f.value))
                    .collect(),
            )
        };

        Ok(TrainingInstance {
            gain: libsvm.label,
            qid: libsvm.query.ok_or("Missing qid")?,
            features,
        })
    }
    pub fn is_relevant(&self) -> bool {
        self.gain.into_inner() > 0.0
    }
    pub fn perceptron_label(&self) -> i32 {
        if self.is_relevant() {
            1
        } else {
            -1
        }
    }
}
