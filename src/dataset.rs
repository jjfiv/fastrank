use crate::evaluators::*;
use crate::io_helper;
use crate::libsvm;
use crate::model::Model;
use crate::qrel::QuerySetJudgments;
use crate::stats::{ComputedStats, StreamingStats};
use ordered_float::NotNan;
use std::collections::HashMap;
use std::collections::HashSet;
use std::f64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_stats: HashMap<u32, ComputedStats>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Normalizer {
    MaxMinNormalizer(FeatureStats),
    ZScoreNormalizer(FeatureStats),
    SigmoidNormalizer(),
}

impl Normalizer {
    pub fn new(method: &str, dataset: &RankingDataset) -> Result<Normalizer, String> {
        Ok(match method {
            "zscore" => Normalizer::ZScoreNormalizer(dataset.compute_feature_stats()),
            "maxmin" | "linear" => Normalizer::MaxMinNormalizer(dataset.compute_feature_stats()),
            "sigmoid" => Normalizer::SigmoidNormalizer(),
            unkn => Err(format!("Unsupported Normalizer: {}", unkn))?,
        })
    }
    fn normalize(&self, fid: u32, val: f32) -> f32 {
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
                    fid
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

pub enum Features {
    Dense32(Vec<f32>),
    /// Sparse 32-bit representation; must be sorted!
    Sparse32(Vec<(u32, f32)>),
}

impl Features {
    pub fn get(&self, idx: u32) -> Option<f64> {
        match self {
            Features::Dense32(arr) => arr.get(idx as usize).map(|val| f64::from(*val)),
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
    pub fn ids(&self) -> Vec<u32> {
        let mut features: Vec<u32> = Vec::new();
        match self {
            Features::Dense32(arr) => features.extend(0..(arr.len() as u32)),
            Features::Sparse32(arr) => features.extend(arr.iter().map(|(idx, _)| *idx)),
        }
        features
    }
    pub fn update_stats(&self, per_feature_stats: &mut HashMap<u32, StreamingStats>) {
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
                    let fid = fid as u32;
                    *val = normalizer.normalize(fid, *val);
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
                    .map(|f| (f.idx, f.value))
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

pub fn load_feature_names_json(path: &str) -> Result<HashMap<u32, String>, Box<std::error::Error>> {
    let reader = io_helper::open_reader(path)?;
    let data: HashMap<String, String> = serde_json::from_reader(reader)?;
    let data: Result<HashMap<u32, String>, _> = data
        .into_iter()
        .map(|(k, v)| k.parse::<u32>().map(|num| (num, v)))
        .collect();
    Ok(data?)
}

pub struct RankingDataset {
    pub instances: Vec<TrainingInstance>,
    pub features: Vec<u32>,
    pub n_dim: u32,
    pub normalization: Option<Normalizer>,
    pub data_by_query: HashMap<String, Vec<usize>>,
    pub feature_names: HashMap<u32, String>,
}

impl RankingDataset {
    pub fn compute_feature_subsets(&self, features: &[u32], instances: &[u32]) -> FeatureStats {
        let mut stats_builders: HashMap<u32, StreamingStats> = features
            .iter()
            .cloned()
            .map(|fid| (fid, StreamingStats::new()))
            .collect();

        for index in instances.iter().cloned() {
            self.instances[index as usize]
                .features
                .update_stats(&mut stats_builders);
        }

        FeatureStats {
            feature_stats: stats_builders
                .into_iter()
                .flat_map(|(fid, stats)| stats.finish().map(|cs| (fid, cs)))
                .collect(),
        }
    }
    pub fn label_stats(&self, instances: &[u32]) -> Option<ComputedStats> {
        let mut label_stats = StreamingStats::new();
        for index in instances.iter().cloned() {
            label_stats.push(self.instances[index as usize].gain.into_inner() as f64);
        }
        label_stats.finish()
    }
    pub fn compute_feature_stats(&self) -> FeatureStats {
        let mut stats_builders: HashMap<u32, StreamingStats> = self
            .features
            .iter()
            .cloned()
            .map(|fid| (fid, StreamingStats::new()))
            .collect();
        for inst in self.instances.iter() {
            inst.features.update_stats(&mut stats_builders);
        }

        for (fid, stats) in stats_builders.iter() {
            println!("fid={}\t{:?}", fid, stats);
        }

        FeatureStats {
            feature_stats: stats_builders
                .into_iter()
                .flat_map(|(fid, stats)| stats.finish().map(|cs| (fid, cs)))
                .collect(),
        }
    }

    pub fn apply_normalization(&mut self, normalizer: &Normalizer) {
        if self.normalization.is_some() {
            panic!("Cannot apply normalization twice!");
        }
        for inst in self.instances.iter_mut() {
            inst.features.apply_normalization(&normalizer);
        }
        self.normalization = Some(normalizer.clone());
    }

    /// Remove a feature or return "not-found".
    pub fn try_remove_feature(&mut self, name_or_num: &str) -> Result<(), String> {
        if let Some((num, _)) = self
            .feature_names
            .iter()
            .find(|(_, v)| v.as_str() == name_or_num)
        {
            if let Some(idx) = self.features.iter().position(|n| n == num) {
                self.features.swap_remove(idx);
                return Ok(());
            } else {
                return Err(format!(
                    "Named feature not present in actual dataset! {}",
                    name_or_num
                ))?;
            }
        }

        let num = name_or_num.parse::<u32>().map_err(|_| {
            format!(
                "Could not turn {} into a name or number in this dataset.",
                name_or_num
            )
        })?;
        if let Some(idx) = self.features.iter().position(|n| *n == num) {
            self.features.swap_remove(idx);
            return Ok(());
        } else {
            return Err(format!(
                "Feature #{} not present in actual dataset!",
                name_or_num
            ))?;
        }
    }

    pub fn make_evaluator(
        &self,
        orig_name: &str,
        judgments: Option<QuerySetJudgments>,
    ) -> Result<Box<Evaluator>, Box<std::error::Error>> {
        let (name, depth) = if let Some(at_point) = orig_name.find("@") {
            let (lhs, rhs) = orig_name.split_at(at_point);
            let depth = rhs[1..]
                .parse::<usize>()
                .map_err(|_| format!("Couldn't parse after the @ in \"{}\": {}", orig_name, rhs))?;
            (lhs.to_lowercase(), Some(depth))
        } else {
            (orig_name.to_lowercase(), None)
        };
        Ok(match name.as_str() {
            "ap" | "map" => Box::new(AveragePrecision::new(&self, judgments.clone())),
            "rr" | "mrr" => Box::new(ReciprocalRank),
            "ndcg" => Box::new(NDCG::new(depth, &self, judgments.clone())),
            _ => Err(format!("Invalid training measure: \"{}\"", orig_name))?,
        })
    }

    pub fn evaluate_mean(&self, model: &Model, evaluator: &Evaluator) -> f64 {
        let mut sum_score = 0.0;
        let num_scores = self.data_by_query.len() as f64;
        for (qid, docs) in self.data_by_query.iter() {
            // Predict for every document:
            let mut ranked_list: Vec<_> = docs
                .iter()
                .cloned()
                .map(|index| {
                    RankedInstance::new(
                        model.score(&self.instances[index].features),
                        self.instances[index].gain,
                        index as u32,
                    )
                })
                .collect();
            // Sort largest to smallest:
            ranked_list.sort_unstable();
            sum_score += evaluator.score(&qid, &ranked_list);
        }
        sum_score / num_scores
    }

    pub fn load_libsvm(
        path: &str,
        feature_names: Option<&HashMap<u32, String>>,
    ) -> Result<Self, Box<std::error::Error>> {
        let reader = io_helper::open_reader(path)?;
        let mut instances = Vec::new();
        for inst in libsvm::instances(reader) {
            let inst = TrainingInstance::try_new(inst?)?;
            instances.push(inst);
        }
        Ok(Self::new(instances, feature_names))
    }

    pub fn new(data: Vec<TrainingInstance>, feature_names: Option<&HashMap<u32, String>>) -> Self {
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

        Self {
            instances: data,
            features,
            n_dim,
            normalization: None,
            data_by_query,
            feature_names: feature_names.cloned().unwrap_or(HashMap::new()),
        }
    }
}
