use crate::dataset::{FeatureStats, RankingDataset, FeatureId};
use ordered_float::NotNan;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Normalizer {
    MaxMinNormalizer(FeatureStats),
    ZScoreNormalizer(FeatureStats),
    SigmoidNormalizer(),
}

impl Normalizer {
    pub fn new(method: &str, dataset: &dyn RankingDataset) -> Result<Normalizer, String> {
        Ok(match method {
            "zscore" => Normalizer::ZScoreNormalizer(FeatureStats::compute(dataset)),
            "maxmin" | "linear" => Normalizer::MaxMinNormalizer(FeatureStats::compute(dataset)),
            "sigmoid" => Normalizer::SigmoidNormalizer(),
            unkn => Err(format!("Unsupported Normalizer: {}", unkn))?,
        })
    }
    pub fn normalize(&self, fid: FeatureId, val: f32) -> f32 {
        match self {
            Normalizer::MaxMinNormalizer(fs) => {
                if let Some(stats) = fs.feature_stats.get(&fid) {
                    let max = stats.max as f32;
                    let min = stats.min as f32;
                    if max == min {
                        return 0.0;
                    }
                    match NotNan::new((val - min) / (max - min)) {
                        Ok(out) => return out.into_inner(),
                        Err(_) => panic!(
                            "Normalization.maxmin NaN: {} {} {:?}",
                            val,
                            max - min,
                            stats
                        ),
                    }
                }
            }
            Normalizer::ZScoreNormalizer(fs) => {
                if let Some(stats) = fs.feature_stats.get(&fid) {
                    let mean = stats.mean as f32;
                    let stddev = stats.variance.sqrt() as f32;
                    if stddev == 0.0 {
                        return 0.0;
                    }
                    match NotNan::new((val - mean) / stddev) {
                        Ok(out) => return out.into_inner(),
                        Err(_) => panic!(
                            "Normalization.zscore NaN: {} {} {} {:?}",
                            stats.mean,
                            stats.variance,
                            stats.variance.sqrt(),
                            stats
                        ),
                    };
                }
            }
            Normalizer::SigmoidNormalizer() => match NotNan::new(sigmoid(val)) {
                Ok(out) => return out.into_inner() as f32,
                Err(_) => panic!(
                    "Normalization.sigmoid NaN: {} {} {} {} {}",
                    val,
                    val.exp(),
                    -(val).exp(),
                    sigmoid(val),
                    fid.to_index()
                ),
            },
        }
        // if no stats or match, original value.
        val
    }
}

/// [Numerically stable sigmoid](https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)
fn sigmoid(x: f32) -> f32 {
    if x > 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
