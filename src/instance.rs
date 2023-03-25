use std::cmp::Ordering;

use crate::libsvm;
use crate::normalizers::Normalizer;
use crate::FeatureId;
use ordered_float::NotNan;

pub enum Features {
    Dense32(Vec<f32>),
    /// Sparse 32-bit representation; must be sorted!
    Sparse32(Vec<(FeatureId, f32)>),
}

impl Features {
    pub fn ids(&self) -> Vec<FeatureId> {
        let mut features: Vec<FeatureId> = Vec::new();
        match self {
            Features::Dense32(arr) => features.extend((0..arr.len()).map(FeatureId::from_index)),
            Features::Sparse32(arr) => features.extend(arr.iter().map(|(idx, _)| *idx)),
        }
        features
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

pub trait FeatureRead {
    fn get(&self, idx: FeatureId) -> Option<f64>;
    /// Note: assumes zero as missing.
    fn dotp(&self, weights: &[f64]) -> f64;
}

pub struct Instance {
    pub gain: NotNan<f32>,
    pub qid: String,
    pub docid: Option<String>,
    pub features: Features,
}

impl FeatureRead for Instance {
    fn get(&self, idx: FeatureId) -> Option<f64> {
        self.features.get(idx)
    }
    fn dotp(&self, weights: &[f64]) -> f64 {
        self.features.dotp(weights)
    }
}

impl FeatureRead for Features {
    fn get(&self, idx: FeatureId) -> Option<f64> {
        match self {
            Features::Dense32(arr) => arr.get(idx.to_index()).map(|val| f64::from(*val)),
            Features::Sparse32(features) => {
                for (fidx, val) in features.iter() {
                    match fidx.cmp(&idx) {
                        Ordering::Less => continue,
                        Ordering::Equal => return Some(f64::from(*val)),
                        Ordering::Greater => break,
                    }
                }
                None
            }
        }
    }
    fn dotp(&self, weights: &[f64]) -> f64 {
        let mut output = 0.0;
        match self {
            Features::Dense32(arr) => {
                for (feature, weight) in arr.iter().cloned().zip(weights.iter().cloned()) {
                    output += f64::from(feature) * weight;
                }
            }
            Features::Sparse32(arr) => {
                for (idx, feature) in arr.iter().cloned() {
                    output += f64::from(feature) * weights[idx.to_index()];
                }
            }
        };
        output
    }
}

impl Instance {
    pub fn new(gain: NotNan<f32>, qid: String, docid: Option<String>, features: Features) -> Self {
        Self {
            gain,
            qid,
            docid,
            features,
        }
    }
    pub fn try_new(libsvm: libsvm::Instance) -> Result<Instance, &'static str> {
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

        Ok(Instance {
            gain: libsvm.label,
            qid: libsvm.query.ok_or("Missing qid")?,
            docid: libsvm.comment,
            features,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::core::FeatureId;

    use super::{FeatureRead, Features};

    fn assert_opt_flt_eq(lhs: Option<f64>, rhs: Option<f64>) {
        match (lhs, rhs) {
            (Some(lhs), Some(rhs)) => {
                if (lhs - rhs).abs() > 1e-7 {
                    panic!("{lhs} != {rhs}")
                }
            }
            (None, None) => {}
            (x, y) => panic!("{x:?} != {y:?}"),
        }
    }

    #[test]
    fn test_sparse() {
        let alt = Features::Sparse32(vec![
            (FeatureId(1), 1.0),
            (FeatureId(3), 3.0),
            (FeatureId(5), 5.0),
        ]);
        assert_opt_flt_eq(alt.get(FeatureId(1)), Some(1.0));
        assert_opt_flt_eq(alt.get(FeatureId(2)), None);
        assert_opt_flt_eq(alt.get(FeatureId(3)), Some(3.0));
        assert_opt_flt_eq(alt.get(FeatureId(4)), None);
        assert_opt_flt_eq(alt.get(FeatureId(5)), Some(5.0));
        assert_opt_flt_eq(alt.get(FeatureId(6)), None);
    }
}
