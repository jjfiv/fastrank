use crate::dataset::{DatasetRef, RankingDataset};
use crate::instance::FeatureRead;
use crate::model::Model;
use crate::{FeatureId, InstanceId};
use ordered_float::NotNan;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum TypedArrayRef {
    DenseI32(&'static [i32]),
    DenseI64(&'static [i64]),
    DenseF32(&'static [f32]),
    DenseF64(&'static [f64]),
}

impl TypedArrayRef {
    pub fn len(&self) -> usize {
        match self {
            TypedArrayRef::DenseI32(arr) => arr.len(),
            TypedArrayRef::DenseI64(arr) => arr.len(),
            TypedArrayRef::DenseF32(arr) => arr.len(),
            TypedArrayRef::DenseF64(arr) => arr.len(),
        }
    }
    pub fn get_i32(&self, index: usize) -> Option<i32> {
        match self {
            TypedArrayRef::DenseI32(arr) => arr.get(index).cloned(),
            TypedArrayRef::DenseI64(_) => None,
            TypedArrayRef::DenseF32(_) => None,
            TypedArrayRef::DenseF64(_) => None,
        }
    }
    pub fn get_i64(&self, index: usize) -> Option<i64> {
        match self {
            TypedArrayRef::DenseI32(arr) => arr.get(index).cloned().map(|x| x as i64),
            TypedArrayRef::DenseI64(arr) => arr.get(index).cloned(),
            TypedArrayRef::DenseF32(_) => None,
            TypedArrayRef::DenseF64(_) => None,
        }
    }
    pub fn get_f32(&self, index: usize) -> Option<f32> {
        match self {
            TypedArrayRef::DenseI32(arr) => arr.get(index).cloned().map(|x| x as f32),
            TypedArrayRef::DenseI64(arr) => arr.get(index).cloned().map(|x| x as f32),
            TypedArrayRef::DenseF32(arr) => arr.get(index).cloned(),
            TypedArrayRef::DenseF64(arr) => arr.get(index).cloned().map(|x| x as f32),
        }
    }
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        match self {
            TypedArrayRef::DenseI32(arr) => arr.get(index).cloned().map(|x| x as f64),
            TypedArrayRef::DenseI64(arr) => arr.get(index).cloned().map(|x| x as f64),
            TypedArrayRef::DenseF32(arr) => arr.get(index).cloned().map(|x| x as f64),
            TypedArrayRef::DenseF64(arr) => arr.get(index).cloned(),
        }
    }
    pub fn dot(&self, weights: &[f64], start: usize) -> f64 {
        let mut sum = 0.0;
        match self {
            TypedArrayRef::DenseI32(_) => todo! {},
            TypedArrayRef::DenseI64(_) => todo! {},
            TypedArrayRef::DenseF32(arr) => {
                for (w, x) in arr[start..].iter().cloned().zip(weights.iter().cloned()) {
                    sum += (w as f64) * x;
                }
            }
            TypedArrayRef::DenseF64(arr) => {
                for (w, x) in arr[start..].iter().zip(weights) {
                    sum += w * x;
                }
            }
        }
        sum
    }
}

#[derive(Debug, Clone)]
pub struct DenseDataset {
    n_features: usize,
    n_instances: usize,
    xs: TypedArrayRef,
    ys: TypedArrayRef,
    qid_strings: HashMap<i64, String>,
    qids: TypedArrayRef,
    feature_names: HashMap<FeatureId, String>,
}

impl DenseDataset {
    pub fn into_ref(self) -> DatasetRef {
        DatasetRef {
            data: Arc::new(self),
        }
    }
    pub fn try_new(
        n_instances: usize,
        n_features: usize,
        xs: &'static [f32],
        ys: &'static [f64],
        qids: &'static [i64],
    ) -> Result<DenseDataset, Box<dyn Error>> {
        let mut qid_nos = Vec::new();
        let mut qid_strings = HashMap::new();

        for qid in qids.iter().cloned() {
            qid_strings.entry(qid).or_insert_with(|| format!("{}", qid));
            qid_nos.push(qid);
        }

        Ok(DenseDataset {
            n_instances,
            n_features,
            xs: TypedArrayRef::DenseF32(xs),
            ys: TypedArrayRef::DenseF64(ys),
            qids: TypedArrayRef::DenseI64(qids),
            qid_strings,
            feature_names: HashMap::new(),
        })
    }
}

struct DenseDatasetInstance<'dataset> {
    dataset: &'dataset DenseDataset,
    id: InstanceId,
}

impl FeatureRead for DenseDatasetInstance<'_> {
    fn get(&self, idx: FeatureId) -> Option<f64> {
        self.dataset.get_feature_value(self.id, idx)
    }
    fn dotp(&self, weights: &[f64]) -> f64 {
        let start = self.id.to_index() * self.dataset.n_features;
        self.dataset.xs.dot(weights, start)
    }
}

impl RankingDataset for DenseDataset {
    fn get_ref(&self) -> Option<DatasetRef> {
        None
        //panic!("Use into_ref() instead!")
    }
    fn is_sampled(&self) -> bool {
        false
    }
    fn features(&self) -> Vec<FeatureId> {
        (0..self.n_features)
            .map(|i| FeatureId::from_index(i))
            .collect()
    }
    fn n_dim(&self) -> u32 {
        self.n_features as u32
    }
    fn instances(&self) -> Vec<InstanceId> {
        (0..self.n_instances)
            .map(|i| InstanceId::from_index(i))
            .collect()
    }
    fn instances_by_query(&self) -> HashMap<String, Vec<InstanceId>> {
        let mut ref_map = HashMap::<&str, Vec<InstanceId>>::new();
        for i in 0..self.qids.len() {
            let qid_no = self.qids.get_i64(i).unwrap();
            let qid_str = &self.qid_strings[&qid_no];
            ref_map
                .entry(qid_str.as_str())
                .or_default()
                .push(InstanceId::from_index(i));
        }
        ref_map
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }
    fn score(&self, id: InstanceId, model: &dyn Model) -> NotNan<f64> {
        let instance = DenseDatasetInstance { id, dataset: self };
        model.score(&instance)
    }
    fn gain(&self, id: InstanceId) -> NotNan<f32> {
        let index = id.to_index();
        let y = self
            .ys
            .get_f32(index)
            .expect("only valid TrainingInstances should exist");
        NotNan::new(y)
            .map_err(|_| format!("NaN in ys[{}]", index))
            .unwrap()
    }
    fn query_id(&self, id: InstanceId) -> &str {
        let qid_no = self.qids.get_i64(id.to_index()).unwrap();
        self.qid_strings[&qid_no].as_str()
    }
    fn document_name(&self, _id: InstanceId) -> Option<&str> {
        // TODO: someday support names array!
        None
    }
    fn queries(&self) -> Vec<String> {
        self.qid_strings.values().cloned().collect()
    }
    /// For printing, the name if available or the number.
    fn feature_name(&self, fid: FeatureId) -> String {
        self.feature_names
            .get(&fid)
            .cloned()
            .unwrap_or_else(|| format!("{}", fid.to_index()))
    }
    /// Lookup a feature value for a particular instance.
    fn get_feature_value(&self, instance: InstanceId, fid: FeatureId) -> Option<f64> {
        let index = self.n_features * instance.to_index() + fid.to_index();
        self.xs.get_f64(index)
    }
    // Given a name or number as a string, lookup the feature id:
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<FeatureId, Box<dyn Error>> {
        crate::dataset::try_lookup_feature(self, &self.feature_names, name_or_num)
    }
}
