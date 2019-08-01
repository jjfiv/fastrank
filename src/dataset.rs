use crate::io_helper;
use crate::libsvm;
use crate::normalizers::Normalizer;
use crate::stats::{ComputedStats, StreamingStats};
use ordered_float::NotNan;
use rand::prelude::*;
use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::f64;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub feature_stats: HashMap<u32, ComputedStats>,
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

pub trait DatasetSampling {
    /// Sample this dataset randomly to frate percent of features and srate percent of instances.
    /// At least one feature and one instance is selected no matter how small the percentage.
    fn random_sample<R: Rng>(&self, frate: f64, srate: f64, rand: &mut R) -> SampledDatasetRef;
}

impl DatasetSampling for RankingDataset {
    fn random_sample<R: Rng>(&self, frate: f64, srate: f64, rand: &mut R) -> SampledDatasetRef {
        let features = self.features();
        let instances = self.instances();

        let n_features = cmp::max(1, ((features.len() as f64) * frate) as usize);
        let n_instances = cmp::max(1, ((instances.len() as f64) * srate) as usize);

        let features = features
            .choose_multiple(rand, n_features)
            .cloned()
            .collect();
        let instances = instances
            .choose_multiple(rand, n_instances)
            .cloned()
            .collect();

        SampledDatasetRef {
            parent: self.get_ref(),
            features,
            instances,
        }
    }
}

pub trait RankingDataset: Send + Sync {
    fn get_ref(&self) -> DatasetRef;
    fn features(&self) -> Vec<u32>;
    fn n_dim(&self) -> u32;
    fn instances(&self) -> Vec<u32>;
    fn get_instance(&self, id: u32) -> &TrainingInstance;
    fn instances_by_query(&self) -> HashMap<String, Vec<u32>>;
    fn queries(&self) -> Vec<String>;
    /// For printing, the name if available or the number.
    fn feature_name(&self, fid: u32) -> String;
    /// Lookup a feature value for a particular instance.
    fn get_feature_value(&self, instance: u32, fid: u32) -> Option<f64>;
    // Given a name or number as a string, lookup the feature id:
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<u32, Box<Error>>;
}

/// This is an Arc wrapper around a LoadedRankingDataset, for cheaper copies.
#[derive(Clone)]
pub struct DatasetRef {
    pub data: Arc<LoadedRankingDataset>,
}
/// Just proxy these requests to the inner (expensive-copy) implementation.
impl RankingDataset for DatasetRef {
    fn get_ref(&self) -> DatasetRef {
        self.clone()
    }
    fn features(&self) -> Vec<u32> {
        self.data.features()
    }
    fn n_dim(&self) -> u32 {
        self.data.n_dim()
    }
    fn instances(&self) -> Vec<u32> {
        self.data.instances()
    }
    fn get_instance(&self, id: u32) -> &TrainingInstance {
        self.data.get_instance(id)
    }
    fn instances_by_query(&self) -> HashMap<String, Vec<u32>> {
        self.data.instances_by_query()
    }
    fn queries(&self) -> Vec<String> {
        self.data.queries()
    }
    fn feature_name(&self, fid: u32) -> String {
        self.data.feature_name(fid)
    }
    fn get_feature_value(&self, instance: u32, fid: u32) -> Option<f64> {
        self.data.get_feature_value(instance, fid)
    }
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<u32, Box<Error>> {
        self.data.try_lookup_feature(name_or_num)
    }
}

#[derive(Clone)]
pub struct SampledDatasetRef {
    pub parent: DatasetRef,
    pub features: Vec<u32>,
    pub instances: Vec<u32>,
}

impl RankingDataset for SampledDatasetRef {
    fn get_ref(&self) -> DatasetRef {
        self.parent.clone()
    }
    fn features(&self) -> Vec<u32> {
        self.features.clone()
    }
    fn n_dim(&self) -> u32 {
        self.features.len() as u32
    }
    fn instances(&self) -> Vec<u32> {
        self.instances.clone()
    }
    fn get_instance(&self, id: u32) -> &TrainingInstance {
        self.parent.get_instance(id)
    }
    fn instances_by_query(&self) -> HashMap<String, Vec<u32>> {
        let mut out = HashMap::new();
        for id in self.instances.iter().cloned() {
            out.entry(self.parent.get_instance(id).qid.to_owned())
                .or_insert(Vec::new())
                .push(id);
        }
        out
    }
    fn queries(&self) -> Vec<String> {
        let mut out: HashSet<&str> = HashSet::new();
        for id in self.instances.iter().cloned() {
            out.insert(&self.parent.get_instance(id).qid);
        }
        out.iter().map(|s| s.to_string()).collect()
    }
    fn feature_name(&self, fid: u32) -> String {
        self.parent.feature_name(fid)
    }
    fn get_feature_value(&self, instance: u32, fid: u32) -> Option<f64> {
        self.parent.get_feature_value(instance, fid)
    }
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<u32, Box<Error>> {
        let fid = self.parent.try_lookup_feature(name_or_num)?;
        if self.features.contains(&fid) {
            return Ok(fid);
        } else {
            Err(format!(
                "Feature not in subsample: {}: {}",
                name_or_num, fid
            ))?
        }
    }
}

impl DatasetRef {
    pub fn load_libsvm(
        path: &str,
        feature_names: Option<&HashMap<u32, String>>,
    ) -> Result<DatasetRef, Box<std::error::Error>> {
        Ok(DatasetRef {
            data: Arc::new(LoadedRankingDataset::load_libsvm(path, feature_names)?),
        })
    }
    pub fn new(data: Vec<TrainingInstance>, feature_names: Option<&HashMap<u32, String>>) -> Self {
        DatasetRef {
            data: Arc::new(LoadedRankingDataset::new(data, feature_names)),
        }
    }
}

pub struct LoadedRankingDataset {
    pub instances: Vec<TrainingInstance>,
    pub features: Vec<u32>,
    pub n_dim: u32,
    pub normalization: Option<Normalizer>,
    pub data_by_query: HashMap<String, Vec<u32>>,
    pub feature_names: HashMap<u32, String>,
}

impl LoadedRankingDataset {
    pub fn into_ref(self) -> DatasetRef {
        DatasetRef {
            data: Arc::new(self),
        }
    }
    pub fn load_libsvm(
        path: &str,
        feature_names: Option<&HashMap<u32, String>>,
    ) -> Result<LoadedRankingDataset, Box<std::error::Error>> {
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
        let mut data_by_query: HashMap<String, Vec<u32>> = HashMap::new();
        for (i, inst) in data.iter().enumerate() {
            data_by_query
                .entry(inst.qid.clone())
                .or_insert(Vec::new())
                .push(i as u32);
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

        LoadedRankingDataset {
            instances: data,
            features,
            n_dim,
            normalization: None,
            data_by_query,
            feature_names: feature_names.cloned().unwrap_or(HashMap::new()),
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
}

impl RankingDataset for LoadedRankingDataset {
    fn get_ref(&self) -> DatasetRef {
        panic!("This is too expensive!")
    }
    fn features(&self) -> Vec<u32> {
        self.features.clone()
    }
    fn n_dim(&self) -> u32 {
        self.n_dim
    }
    fn instances(&self) -> Vec<u32> {
        (0..self.instances.len()).map(|i| i as u32).collect()
    }
    fn get_instance(&self, id: u32) -> &TrainingInstance {
        &self.instances[id as usize]
    }
    fn instances_by_query(&self) -> HashMap<String, Vec<u32>> {
        self.data_by_query.clone()
    }
    fn queries(&self) -> Vec<String> {
        self.data_by_query
            .iter()
            .map(|(k, _v)| k)
            .cloned()
            .collect()
    }
    fn feature_name(&self, fid: u32) -> String {
        self.feature_names
            .get(&fid)
            .cloned()
            .unwrap_or(format!("{}", fid))
    }
    fn get_feature_value(&self, instance: u32, fid: u32) -> Option<f64> {
        self.instances[instance as usize].features.get(fid)
    }
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<u32, Box<Error>> {
        if let Some((num, _)) = self
            .feature_names
            .iter()
            .find(|(_, v)| v.as_str() == name_or_num)
        {
            if let Some(idx) = self.features.iter().position(|n| n == num) {
                return Ok(idx as u32);
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
            return Ok(idx as u32);
        } else {
            return Err(format!(
                "Feature #{} not present in actual dataset!",
                name_or_num
            ))?;
        }
    }
}
