use crate::instance::{FeatureRead, Instance};
use crate::io_helper;
use crate::libsvm;
use crate::model::Model;
use crate::normalizers::Normalizer;
use crate::{FeatureId, InstanceId};
use ordered_float::NotNan;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::f64;
use std::sync::Arc;

pub fn load_feature_names_json(
    path: &str,
) -> Result<HashMap<FeatureId, String>, Box<dyn Error>> {
    let reader = io_helper::open_reader(path)?;
    let data: HashMap<String, String> = serde_json::from_reader(reader)?;
    let data: Result<HashMap<FeatureId, String>, _> = data
        .into_iter()
        .map(|(k, v)| {
            k.parse::<usize>()
                .map(|num| (FeatureId::from_index(num), v))
        })
        .collect();
    Ok(data?)
}

pub trait RankingDataset: Send + Sync {
    fn get_ref(&self) -> Option<DatasetRef>;
    fn features(&self) -> Vec<FeatureId>;
    fn n_dim(&self) -> u32;
    fn instances(&self) -> Vec<InstanceId>;
    fn instances_by_query(&self) -> HashMap<String, Vec<InstanceId>>;

    fn score(&self, id: InstanceId, model: &dyn Model) -> NotNan<f64>;
    fn gain(&self, id: InstanceId) -> NotNan<f32>;
    fn query_id(&self, id: InstanceId) -> &str;
    /// If the dataset has names, return Some(name)
    fn document_name(&self, id: InstanceId) -> Option<&str>;

    fn queries(&self) -> Vec<String>;
    /// For printing, the name if available or the number.
    fn feature_name(&self, fid: FeatureId) -> String;
    /// Lookup a feature value for a particular instance.
    fn get_feature_value(&self, instance: InstanceId, fid: FeatureId) -> Option<f64>;
    // Given a name or number as a string, lookup the feature id:
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<FeatureId, Box<dyn Error>>;
}

/// This is an Arc wrapper around a LoadedRankingDataset, for cheaper copies.
#[derive(Clone)]
pub struct DatasetRef {
    pub data: Arc<dyn RankingDataset>,
}
/// Just proxy these requests to the inner (expensive-copy) implementation.
impl RankingDataset for DatasetRef {
    fn get_ref(&self) -> Option<DatasetRef> {
        Some(self.data.get_ref().unwrap_or(self.clone()))
    }
    fn features(&self) -> Vec<FeatureId> {
        self.data.features()
    }
    fn n_dim(&self) -> u32 {
        self.data.n_dim()
    }
    fn instances(&self) -> Vec<InstanceId> {
        self.data.instances()
    }
    fn score(&self, id: InstanceId, model: &dyn Model) -> NotNan<f64> {
        self.data.score(id, model)
    }
    fn gain(&self, id: InstanceId) -> NotNan<f32> {
        self.data.gain(id)
    }
    fn query_id(&self, id: InstanceId) -> &str {
        self.data.query_id(id)
    }
    fn document_name(&self, id: InstanceId) -> Option<&str> {
        self.data.document_name(id)
    }
    fn instances_by_query(&self) -> HashMap<String, Vec<InstanceId>> {
        self.data.instances_by_query()
    }
    fn queries(&self) -> Vec<String> {
        self.data.queries()
    }
    fn feature_name(&self, fid: FeatureId) -> String {
        self.data.feature_name(fid)
    }
    fn get_feature_value(&self, instance: InstanceId, fid: FeatureId) -> Option<f64> {
        self.data.get_feature_value(instance, fid)
    }
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<FeatureId, Box<dyn Error>> {
        self.data.try_lookup_feature(name_or_num)
    }
}

#[derive(Clone)]
pub struct SampledDatasetRef {
    pub parent: DatasetRef,
    pub features: Vec<FeatureId>,
    pub instances: Vec<InstanceId>,
}

impl SampledDatasetRef {
    pub fn into_ref(self) -> DatasetRef {
        DatasetRef {
            data: Arc::new(self),
        }
    }
}

impl RankingDataset for SampledDatasetRef {
    fn get_ref(&self) -> Option<DatasetRef> {
        self.parent.get_ref()
    }
    fn features(&self) -> Vec<FeatureId> {
        self.features.clone()
    }
    fn n_dim(&self) -> u32 {
        self.features.len() as u32
    }
    fn instances(&self) -> Vec<InstanceId> {
        self.instances.clone()
    }
    fn instances_by_query(&self) -> HashMap<String, Vec<InstanceId>> {
        let mut out = HashMap::new();
        for id in self.instances.iter().cloned() {
            out.entry(self.parent.query_id(id).to_owned())
                .or_insert(Vec::new())
                .push(id);
        }
        out
    }
    fn score(&self, id: InstanceId, model: &dyn Model) -> NotNan<f64> {
        self.parent.score(id, model)
    }
    fn gain(&self, id: InstanceId) -> NotNan<f32> {
        self.parent.gain(id)
    }
    fn query_id(&self, id: InstanceId) -> &str {
        self.parent.query_id(id)
    }
    fn document_name(&self, id: InstanceId) -> Option<&str> {
        self.parent.document_name(id)
    }
    fn queries(&self) -> Vec<String> {
        let mut out: HashSet<&str> = HashSet::new();
        for id in self.instances.iter().cloned() {
            out.insert(self.parent.query_id(id));
        }
        out.iter().map(|s| s.to_string()).collect()
    }
    fn feature_name(&self, fid: FeatureId) -> String {
        self.parent.feature_name(fid)
    }
    fn get_feature_value(&self, instance: InstanceId, fid: FeatureId) -> Option<f64> {
        self.parent.get_feature_value(instance, fid)
    }
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<FeatureId, Box<dyn Error>> {
        let fid = self.parent.try_lookup_feature(name_or_num)?;
        if self.features.contains(&fid) {
            return Ok(fid);
        } else {
            Err(format!(
                "Feature not in subsample: {}: {}",
                name_or_num,
                fid.to_index()
            ))?
        }
    }
}

impl DatasetRef {
    pub fn load_libsvm(
        path: &str,
        feature_names: Option<&HashMap<FeatureId, String>>,
    ) -> Result<DatasetRef, Box<dyn std::error::Error>> {
        Ok(DatasetRef {
            data: Arc::new(LoadedRankingDataset::load_libsvm(path, feature_names)?),
        })
    }
    pub fn new(data: Vec<Instance>, feature_names: Option<&HashMap<FeatureId, String>>) -> Self {
        DatasetRef {
            data: Arc::new(LoadedRankingDataset::new(data, feature_names)),
        }
    }
}

pub struct LoadedRankingDataset {
    pub instances: Vec<Instance>,
    pub features: Vec<FeatureId>,
    pub n_dim: u32,
    pub normalization: Option<Normalizer>,
    pub data_by_query: HashMap<String, Vec<InstanceId>>,
    pub feature_names: HashMap<FeatureId, String>,
}

impl LoadedRankingDataset {
    pub fn into_ref(self) -> DatasetRef {
        DatasetRef {
            data: Arc::new(self),
        }
    }
    pub fn load_libsvm(
        path: &str,
        feature_names: Option<&HashMap<FeatureId, String>>,
    ) -> Result<LoadedRankingDataset, Box<dyn std::error::Error>> {
        let reader = io_helper::open_reader(path)?;
        let mut instances = Vec::new();
        for inst in libsvm::instances(reader) {
            let inst = Instance::try_new(inst?)?;
            instances.push(inst);
        }
        Ok(Self::new(instances, feature_names))
    }
    pub fn new(data: Vec<Instance>, feature_names: Option<&HashMap<FeatureId, String>>) -> Self {
        // Collect features that are actually present.
        let mut features: HashSet<FeatureId> = HashSet::new();
        // Collect training instances by the query.
        let mut data_by_query: HashMap<String, Vec<InstanceId>> = HashMap::new();
        for (i, inst) in data.iter().enumerate() {
            data_by_query
                .entry(inst.qid.clone())
                .or_insert(Vec::new())
                .push(InstanceId::from_index(i));
            features.extend(inst.features.ids());
        }
        // Get a sorted list of active features in this dataset.
        let mut features: Vec<FeatureId> = features.iter().cloned().collect();
        features.sort_unstable();

        // Calculate the number of features, including any missing ones, so we can have a dense linear model.
        let n_dim = features
            .iter()
            .cloned()
            .max()
            .expect("No features defined!")
            .to_index()
            + 1;

        LoadedRankingDataset {
            instances: data,
            features,
            n_dim: n_dim as u32,
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
    pub fn try_remove_feature(&mut self, name_or_num: &str) -> Result<(), Box<dyn Error>> {
        let fid = self.try_lookup_feature(name_or_num)?;
        self.features.swap_remove(fid.to_index());
        Ok(())
    }
}

impl RankingDataset for LoadedRankingDataset {
    fn get_ref(&self) -> Option<DatasetRef> {
        None
        //panic!("This is too expensive!")
    }
    fn features(&self) -> Vec<FeatureId> {
        self.features.clone()
    }
    fn n_dim(&self) -> u32 {
        self.n_dim
    }
    fn instances(&self) -> Vec<InstanceId> {
        (0..self.instances.len())
            .map(|i| InstanceId::from_index(i))
            .collect()
    }
    fn instances_by_query(&self) -> HashMap<String, Vec<InstanceId>> {
        self.data_by_query.clone()
    }
    fn queries(&self) -> Vec<String> {
        self.data_by_query
            .iter()
            .map(|(k, _v)| k)
            .cloned()
            .collect()
    }
    fn feature_name(&self, fid: FeatureId) -> String {
        self.feature_names
            .get(&fid)
            .cloned()
            .unwrap_or(format!("{}", fid.to_index()))
    }
    fn get_feature_value(&self, instance: InstanceId, fid: FeatureId) -> Option<f64> {
        self.instances[instance.to_index()].features.get(fid)
    }
    fn score(&self, id: InstanceId, model: &dyn Model) -> NotNan<f64> {
        model.score(&self.instances[id.to_index()].features)
    }
    fn gain(&self, id: InstanceId) -> NotNan<f32> {
        self.instances[id.to_index()].gain.clone()
    }
    fn query_id(&self, id: InstanceId) -> &str {
        self.instances[id.to_index()].qid.as_str()
    }
    fn document_name(&self, id: InstanceId) -> Option<&str> {
        self.instances[id.to_index()]
            .docid
            .as_ref()
            .map(|s| s.as_str())
    }
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<FeatureId, Box<dyn Error>> {
        try_lookup_feature(self, &self.feature_names, name_or_num)
    }
}

pub fn try_lookup_feature(
    dataset: &dyn RankingDataset,
    feature_names: &HashMap<FeatureId, String>,
    name_or_num: &str,
) -> Result<FeatureId, Box<dyn Error>> {
    let features = dataset.features();
    if let Some((num, _)) = feature_names
        .iter()
        .find(|(_, v)| v.as_str() == name_or_num)
    {
        if let Some(idx) = features.iter().position(|n| n == num) {
            return Ok(FeatureId::from_index(idx));
        } else {
            return Err(format!(
                "Named feature not present in actual dataset! {}",
                name_or_num
            ))?;
        }
    }

    let num = name_or_num
        .parse::<usize>()
        .map(|id| FeatureId::from_index(id))
        .map_err(|_| {
            format!(
                "Could not turn {} into a name or number in this dataset.",
                name_or_num
            )
        })?;
    if let Some(idx) = features.iter().position(|n| *n == num) {
        return Ok(FeatureId::from_index(idx));
    } else {
        return Err(format!(
            "Feature #{} not present in actual dataset!",
            name_or_num
        ))?;
    }
}
