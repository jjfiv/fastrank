use crate::dataset::{RankingDataset, SampledDatasetRef};
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

    /// This represents a determinisitc sampling of queries.
    fn with_queries(&self, queries: &[String]) -> SampledDatasetRef;

    fn train_test<R: Rng>(
        &self,
        test_fraction: f64,
        rand: &mut R,
    ) -> (SampledDatasetRef, SampledDatasetRef);
}

impl DatasetSampling for &dyn RankingDataset {
    fn random_sample<R: Rng>(&self, frate: f64, srate: f64, rand: &mut R) -> SampledDatasetRef {
        let features = self.features();
        let queries = self.queries();

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
            parent: self.get_ref(),
            features,
            instances,
        }
    }
    fn with_instances(&self, instances: &[InstanceId]) -> SampledDatasetRef {
        SampledDatasetRef {
            parent: self.get_ref(),
            instances: instances.iter().cloned().collect(),
            features: self.features(),
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
            parent: self.get_ref(),
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
