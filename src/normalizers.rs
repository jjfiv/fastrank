use crate::dataset::RankingDataset;
use crate::stats::{ComputedStats, StreamingStats};
use crate::FeatureId;
use ordered_float::NotNan;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_stats: HashMap<FeatureId, ComputedStats>,
}

impl FeatureStats {
    pub fn compute(dataset: &dyn RankingDataset) -> FeatureStats {
        let mut stats_builders: HashMap<FeatureId, StreamingStats> = dataset
            .features()
            .iter()
            .cloned()
            .map(|fid| (fid, StreamingStats::new()))
            .collect();

        for inst in dataset.instances().iter().cloned() {
            for (fid, stats) in stats_builders.iter_mut() {
                if let Some(fval) = dataset.get_feature_value(inst, *fid) {
                    stats.push(fval)
                }
                // Explicitly skip missing; so as not to make it part of normalization.
            }
        }

        FeatureStats {
            feature_stats: stats_builders
                .into_iter()
                .flat_map(|(fid, stats)| stats.finish().map(|cs| (fid, cs)))
                .collect(),
        }
    }
}

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
