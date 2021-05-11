use crate::InstanceId;
use crate::{
    dataset::{DatasetRef, RankingDataset, SampledDatasetRef},
    model::Model,
};
use crate::{evaluators::RankedInstance, FeatureId};
use crate::{heap::ScoringHeap, randutil};
use oorandom::Rand64;
use std::cmp;
use std::collections::HashSet;

pub trait DatasetSampling {
    /// Sample this dataset randomly to frate percent of features and srate percent of instances.
    /// At least one feature and one instance is selected no matter how small the percentage.
    fn random_sample(&self, frate: f64, srate: f64, rand: &mut Rand64) -> SampledDatasetRef;

    /// This represents a deterministic sampling of instances and features.
    fn select(&self, instances: &[InstanceId], features: &[FeatureId]) -> SampledDatasetRef;

    /// This represents a deterministic sampling of instances.
    fn with_instances(&self, instances: &[InstanceId]) -> SampledDatasetRef;

    /// This represents a deterministic sampling of queries.
    fn with_queries(&self, queries: &[String]) -> SampledDatasetRef;

    /// This represents a dataset sample with a deterministic subset of features.
    /// Errors when no features remaining or features to keep not available.
    fn with_features(&self, features: &[FeatureId]) -> Result<SampledDatasetRef, String>;

    fn from_topk(&self, model: &dyn Model, k: usize) -> SampledDatasetRef;

    fn train_test(
        &self,
        test_fraction: f64,
        rand: &mut Rand64,
    ) -> (SampledDatasetRef, SampledDatasetRef);
}

impl DatasetRef {
    fn get_ref_or_clone(&self) -> DatasetRef {
        self.get_ref().unwrap_or_else(|| self.clone())
    }
}

impl DatasetSampling for DatasetRef {
    fn random_sample(&self, frate: f64, srate: f64, rand: &mut Rand64) -> SampledDatasetRef {
        let mut features = self.features();
        let mut instances = self.instances();

        // By sorting here, we're being defensive against the fact that features and queries were likely to have been collected into a set or hashset at some point. This will lead to non-deterministic ordering separate from the seed! When we sample from that, we end up getting different samples despite having the exact same RNG states!
        features.sort_unstable();
        instances.sort_unstable();

        let n_features = cmp::max(1, ((features.len() as f64) * frate) as usize);
        let n_instances = cmp::max(1, ((instances.len() as f64) * srate) as usize);

        let features = randutil::sample_with_replacement(&features, rand, n_features);
        let instances = randutil::sample_with_replacement(&instances, rand, n_instances);

        SampledDatasetRef {
            parent: self.clone(),
            features,
            instances,
        }
    }
    /// Sample both instances and features.
    fn select(&self, instances: &[InstanceId], features: &[FeatureId]) -> SampledDatasetRef {
        SampledDatasetRef {
            parent: self.get_ref_or_clone(),
            instances: instances.to_vec(),
            features: features.to_vec(),
        }
    }

    /// Sample just instances.
    fn with_instances(&self, instances: &[InstanceId]) -> SampledDatasetRef {
        SampledDatasetRef {
            parent: self.get_ref_or_clone(),
            instances: instances.to_vec(),
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

    fn train_test(
        &self,
        test_fraction: f64,
        rand: &mut Rand64,
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

        randutil::shuffle(&mut qs, rand);
        let test_qs = qs.split_off(n_test_qs);
        let train_qs = qs;

        (self.with_queries(&train_qs), self.with_queries(&test_qs))
    }

    fn from_topk(&self, model: &dyn Model, k: usize) -> SampledDatasetRef {
        let features = self.features();
        let mut keep_instances = Vec::with_capacity(k * self.queries().len());
        for (_qid, ids) in self.instances_by_query() {
            let mut heap = ScoringHeap::new(k);
            for id in ids.iter().cloned() {
                let score = self.score(id, model);
                heap.offer(RankedInstance {
                    score,
                    gain: self.gain(id),
                    identifier: id,
                });
            }
            assert!(heap.len() <= k);

            keep_instances.extend(heap.drain_unordered().into_iter().map(|ri| ri.identifier));
        }
        keep_instances.sort_unstable();
        SampledDatasetRef {
            parent: self.get_ref_or_clone(),
            instances: keep_instances,
            features,
        }
    }
}
