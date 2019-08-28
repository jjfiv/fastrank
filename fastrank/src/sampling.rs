use crate::dataset::{DatasetRef, RankingDataset, SampledDatasetRef};
use crate::FeatureId;
use crate::InstanceId;
use rand::prelude::*;
use std::cmp;
use std::collections::HashSet;

pub trait DatasetSampling {
    /// Sample this dataset randomly to frate percent of features and srate percent of instances.
    /// At least one feature and one instance is selected no matter how small the percentage.
    fn random_sample<R: Rng>(&self, frate: f64, srate: f64, rand: &mut R) -> SampledDatasetRef;

    /// This represents a deterministic sampling of instances.
    fn with_instances(&self, instances: &[InstanceId]) -> SampledDatasetRef;

    /// This represents a deterministic sampling of queries.
    fn with_queries(&self, queries: &[String]) -> SampledDatasetRef;

    /// This represents a dataset sample with a deterministic subset of features.
    /// Errors when no features remaining or features to keep not available.
    fn with_features(&self, features: &[FeatureId]) -> Result<SampledDatasetRef, String>;

    fn train_test<R: Rng>(
        &self,
        test_fraction: f64,
        rand: &mut R,
    ) -> (SampledDatasetRef, SampledDatasetRef);
}

impl DatasetRef {
    fn get_ref_or_clone(&self) -> DatasetRef {
        self.get_ref().unwrap_or(self.clone())
    }
}

impl DatasetSampling for DatasetRef {
    fn random_sample<R: Rng>(&self, frate: f64, srate: f64, rand: &mut R) -> SampledDatasetRef {
        let mut features = self.features();
        let mut queries = self.queries();

        // By sorting here, we're being defensive against the fact that features and queries were likely to have been collected into a set or hashset at some point. This will lead to non-deterministic ordering separate from the seed! When we sample from that, we end up getting different samples despite having the exact same RNG states!
        queries.sort_unstable();
        features.sort_unstable();

        let n_features = cmp::max(1, ((features.len() as f64) * frate) as usize);
        let n_queries = cmp::max(1, ((queries.len() as f64) * srate) as usize);

        let features = features
            .choose_multiple(rand, n_features)
            .cloned()
            .collect();
        let queries: HashSet<&str> = queries
            .choose_multiple(rand, n_queries)
            .map(|s| s.as_str())
            .collect();
        let mut instances: Vec<InstanceId> = Vec::new();

        for (qid, qinst) in self.instances_by_query() {
            if queries.contains(qid.as_str()) {
                instances.extend(qinst);
            }
        }

        SampledDatasetRef {
            parent: self.clone(),
            features,
            instances,
        }
    }
    fn with_instances(&self, instances: &[InstanceId]) -> SampledDatasetRef {
        SampledDatasetRef {
            parent: self.get_ref_or_clone(),
            instances: instances.iter().cloned().collect(),
            features: self.features(),
        }
    }

    /// This represents a dataset sample with a deterministic subset of features.
    fn with_features(&self, features: &[FeatureId]) -> Result<SampledDatasetRef, String> {
        let valid_features: HashSet<FeatureId> = self.features().into_iter().collect();
        let mut keep_features = HashSet::new();
        let mut missing_features = HashSet::new();
        for fid in features {
            if valid_features.contains(fid) {
                keep_features.insert(fid);
            } else {
                missing_features.insert(fid);
            }
        }
        if missing_features.len() > 0 {
            Err(format!("Missing Features: {:?}", missing_features))
        } else if keep_features.len() == 0 {
            Err(format!("No Features!"))
        } else {
            Ok(SampledDatasetRef {
                parent: self.get_ref_or_clone(),
                instances: self.instances(),
                features: keep_features.into_iter().cloned().collect(),
            })
        }
    }

    fn with_queries(&self, queries: &[String]) -> SampledDatasetRef {
        let query_set: HashSet<&str> = queries.iter().map(|s| s.as_str()).collect();
        let mut instances: Vec<InstanceId> = Vec::new();

        for (qid, qinst) in self.instances_by_query() {
            if query_set.contains(qid.as_str()) {
                instances.extend(qinst);
            }
        }

        SampledDatasetRef {
            parent: self.get_ref_or_clone(),
            instances,
            features: self.features(),
        }
    }

    fn train_test<R: Rng>(
        &self,
        test_fraction: f64,
        rand: &mut R,
    ) -> (SampledDatasetRef, SampledDatasetRef) {
        let mut qs = self.queries();
        let n_test_qs = ((qs.len() as f64) * test_fraction) as usize;
        if n_test_qs <= 0 {
            panic!(
                "Must be some testing data selected: frac={}!",
                test_fraction
            );
        }
        if n_test_qs >= qs.len() {
            panic!("Must be some training data left!");
        }

        qs.shuffle(rand);
        let test_qs = qs.split_off(n_test_qs);
        let train_qs = qs;

        (self.with_queries(&train_qs), self.with_queries(&test_qs))
    }
}
