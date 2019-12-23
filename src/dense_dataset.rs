use crate::dataset::{DatasetRef, RankingDataset};
use crate::instance::FeatureRead;
use crate::model::Model;
use crate::{FeatureId, InstanceId};
use ordered_float::NotNan;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::error::Error;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DenseDataset {
    n_features: usize,
    n_instances: usize,
    xs: &'static [f32],
    ys: &'static [f64],
    qid_strings: HashMap<u32, String>,
    qids: Vec<u32>,
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
            let qid_no = u32::try_from(qid)?;
            qid_strings
                .entry(qid_no)
                .or_insert_with(|| format!("{}", qid_no));
            qid_nos.push(qid_no);
        }

        Ok(DenseDataset {
            n_instances,
            n_features,
            xs,
            ys,
            qids: qid_nos,
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
        let end = start + self.dataset.n_features;
        let row = &self.dataset.xs[start..end];
        let mut out = 0.0;
        for (feature, weight) in row.iter().cloned().zip(weights.iter().cloned()) {
            out += f64::from(feature) * weight;
        }
        out
    }
}

impl RankingDataset for DenseDataset {
    fn get_ref(&self) -> Option<DatasetRef> {
        None
        //panic!("Use into_ref() instead!")
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
        for (i, qid_no) in self.qids.iter().enumerate() {
            let qid_str = &self.qid_strings[qid_no];
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
            .get(index)
            .expect("only valid TrainingInstances should exist");
        NotNan::new(*y as f32)
            .map_err(|_| format!("NaN in ys[{}]", index))
            .unwrap()
    }
    fn query_id(&self, id: InstanceId) -> &str {
        let qid_no = self.qids[id.to_index()];
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
        let val = self.xs.get(index).expect("Indexes should be valid!");
        Some(f64::from(*val))
    }
    // Given a name or number as a string, lookup the feature id:
    fn try_lookup_feature(&self, name_or_num: &str) -> Result<FeatureId, Box<dyn Error>> {
        crate::dataset::try_lookup_feature(self, &self.feature_names, name_or_num)
    }
}
