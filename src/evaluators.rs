use crate::dataset::{DatasetRef, RankingDataset};
use crate::model::Model;
use crate::qrel::QuerySetJudgments;
use crate::stats::PercentileStats;
use crate::InstanceId;
use oorandom::Rand64;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

const NUM_BOOTSTRAP_SAMPLES: u32 = 200;

#[derive(Debug)]
pub struct RankedInstance {
    pub score: f64,
    pub gain: f32,
    pub identifier: InstanceId,
}

impl PartialEq for RankedInstance {
    fn eq(&self, other: &RankedInstance) -> bool {
        self.cmp(&other) == Ordering::Equal
    }
}

impl PartialOrd for RankedInstance {
    fn partial_cmp(&self, other: &RankedInstance) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Eq for RankedInstance {}

/// Natural sort: first by socre, descending, then by gain ascending (yielding pessimistic scores on ties), finally by identifier.
impl Ord for RankedInstance {
    /// Treat NaN as ties:
    fn cmp(&self, other: &RankedInstance) -> Ordering {
        // score: desc
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Less) => return Ordering::Greater,
            Some(Ordering::Greater) => return Ordering::Less,
            _ => {}
        }
        // gain: asc
        match self.gain.partial_cmp(&other.gain) {
            Some(Ordering::Less) => return Ordering::Less,
            Some(Ordering::Greater) => return Ordering::Greater,
            _ => {}
        }
        // identifier: id
        self.identifier.cmp(&other.identifier)
    }
}

impl RankedInstance {
    pub fn new(score: f64, gain: f32, identifier: InstanceId) -> Self {
        Self {
            score,
            gain,
            identifier,
        }
    }
    pub fn is_relevant(&self) -> bool {
        self.gain > 0.0
    }
}

pub struct DatasetVectors<'dataset> {
    _dataset: &'dataset dyn RankingDataset,
    instances: Vec<InstanceId>,
    gains: Vec<f32>,
    qid_to_index: HashMap<&'dataset str, Vec<usize>>,
}

impl<'d> DatasetVectors<'d> {
    pub fn new(dataset: &'d dyn RankingDataset) -> DatasetVectors<'d> {
        let query_ids: Vec<&'d str> = dataset.query_ids();
        let mut qid_to_indices: HashMap<&'d str, Vec<usize>> =
            HashMap::with_capacity(query_ids.len());
        for (i, qid) in query_ids.into_iter().enumerate() {
            qid_to_indices.entry(qid).or_default().push(i);
        }
        Self {
            _dataset: dataset,
            instances: dataset.instances(),
            gains: dataset.gains(),
            qid_to_index: qid_to_indices,
        }
    }
}

#[derive(Clone)]
pub struct SetEvaluator {
    dataset: DatasetRef,
    evaluator: Arc<dyn Evaluator>,
}

impl SetEvaluator {
    pub fn name(&self) -> String {
        self.evaluator.name()
    }

    pub fn print_standard_eval(
        split_name: &str,
        model: &dyn Model,
        dataset: &DatasetRef,
        judgments: &Option<QuerySetJudgments>,
    ) {
        println!("{} Performance:", split_name);
        for measure in &["map", "rr", "ndcg@5", "ndcg"] {
            let evaluator = SetEvaluator::create(dataset, measure, judgments.clone())
                .expect("print_standard_eval should only have valid measures!");
            let (p5, p25, p50, p75, p95) = evaluator
                .bootstrap_eval(NUM_BOOTSTRAP_SAMPLES, model)
                .summary();
            println!(
                "\t{}:\tMean={:.3}\tPercentiles=({:.3} {:.3} {:.3} {:.3} {:.3})",
                evaluator.name(),
                evaluator.evaluate_mean(model),
                p5,
                p25,
                p50,
                p75,
                p95
            );
        }
    }

    pub fn create(
        dataset: &DatasetRef,
        orig_name: &str,
        judgments: Option<QuerySetJudgments>,
    ) -> Result<SetEvaluator, Box<dyn std::error::Error>> {
        let (name, depth) = if let Some(at_point) = orig_name.find('@') {
            let (lhs, rhs) = orig_name.split_at(at_point);
            let depth = rhs[1..]
                .parse::<usize>()
                .map_err(|_| format!("Couldn't parse after the @ in \"{}\": {}", orig_name, rhs))?;
            (lhs.to_lowercase(), Some(depth))
        } else {
            (orig_name.to_lowercase(), None)
        };
        Ok(SetEvaluator {
            dataset: dataset.clone(),
            evaluator: match name.as_str() {
                "ap" | "map" => Arc::new(AveragePrecision::new(dataset, judgments.clone())),
                "rr" | "mrr" => Arc::new(ReciprocalRank),
                "ndcg" => Arc::new(NDCG::new(depth, dataset, judgments.clone())),
                _ => Err(format!("Invalid training measure: \"{}\"", orig_name))?,
            },
        })
    }

    pub fn bootstrap_eval(&self, num_trials: u32, model: &dyn Model) -> PercentileStats {
        let data = self.evaluate_to_vec(model);
        let n = data.len() as u64;
        let mut means = Vec::new();
        let mut rng = Rand64::new(0xdeadbeef);
        for _ in 0..num_trials {
            let mut sum = 0.0;
            for _ in 0..n {
                let index = rng.rand_range(0..n) as usize;
                sum += data[index];
            }
            means.push(sum / (n as f64))
        }
        PercentileStats::new(&means)
    }

    pub fn fast_eval(&self, model: &dyn Model, vectors: &DatasetVectors<'_>) -> f64 {
        debug_assert_eq!(vectors.instances.len(), self.dataset.instances().len());

        // only deal with generating scores, when vectors are pre-created:
        let predictions: Vec<f64> = self.dataset.score_all(model);
        let mut sum = 0.0;
        let mut n = 0;

        let mut ranked_list: Vec<RankedInstance> = Vec::with_capacity(1000);
        for (qid, indices) in vectors.qid_to_index.iter() {
            ranked_list.clear();

            // Predict for every document:
            for index in indices.into_iter().cloned() {
                let score = predictions[index];
                let gain = vectors.gains[index];
                ranked_list.push(RankedInstance::new(score, gain, vectors.instances[index]));
            }
            // Sort largest to smallest:
            ranked_list.sort_unstable();
            sum += self.evaluator.score(&qid, &ranked_list);
            n += 1;
        }
        if n == 0 {
            return 0.0;
        }
        return sum / (n as f64);
    }

    pub fn evaluate_mean(&self, model: &dyn Model) -> f64 {
        let scores = self.evaluate_to_vec(model);
        if scores.len() == 0 {
            return 0.0;
        }
        let n = scores.len() as f64;
        let mut sum = 0.0;
        for s in scores {
            sum += s;
        }
        return sum / n;
    }

    pub fn evaluate_to_map(&self, model: &dyn Model) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        for (qid, docs) in self.dataset.instances_by_query().iter() {
            // Predict for every document:
            let mut ranked_list: Vec<_> = docs
                .iter()
                .cloned()
                .map(|index| {
                    let score = self.dataset.score(index, model);
                    let gain = self.dataset.gain(index);
                    RankedInstance::new(score, gain, index)
                })
                .collect();
            // Sort largest to smallest:
            ranked_list.sort_unstable();
            scores.insert(qid.to_owned(), self.evaluator.score(&qid, &ranked_list));
        }
        scores
    }

    pub fn evaluate_to_vec(&self, model: &dyn Model) -> Vec<f64> {
        let mut scores = Vec::new();

        let predictions: Vec<f64> = self.dataset.score_all(model);

        let instances = self.dataset.instances();
        let gains: Vec<f32> = self.dataset.gains();
        let query_ids: Vec<&str> = self.dataset.query_ids();

        let mut qid_to_indices: HashMap<&str, Vec<usize>> = HashMap::with_capacity(query_ids.len());
        for (i, qid) in query_ids.into_iter().enumerate() {
            qid_to_indices.entry(qid).or_default().push(i);
        }

        for (qid, indices) in qid_to_indices {
            // Predict for every document:
            let mut ranked_list: Vec<_> = indices
                .iter()
                .cloned()
                .map(|index| {
                    let score = predictions[index];
                    let gain = gains[index];
                    RankedInstance::new(score, gain, instances[index])
                })
                .collect();
            // Sort largest to smallest:
            ranked_list.sort_unstable();
            scores.push(self.evaluator.score(&qid, &ranked_list));
        }
        scores
    }
}

pub trait Evaluator: Send + Sync {
    fn name(&self) -> String;
    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64;
}

#[derive(Clone)]
pub struct ReciprocalRank;

impl Evaluator for ReciprocalRank {
    fn name(&self) -> String {
        String::from("RR")
    }
    fn score(&self, _qid: &str, ranked_list: &[RankedInstance]) -> f64 {
        compute_recip_rank(ranked_list)
    }
}

pub fn compute_dcg(gains: &[f32], depth: Option<usize>, ideal: bool) -> f64 {
    // Gain of 0.0 is a positive value, so we need to expand or contact to "depth" if it's given.
    let mut gain_vector: Vec<f32> = gains.to_vec();
    if ideal {
        gain_vector.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        gain_vector.reverse();
    }
    if let Some(depth) = depth {
        gain_vector.resize(depth, 0.0);
    }
    let mut dcg = 0.0;
    for (i, gain) in gain_vector.into_iter().enumerate() {
        let i = i as f64;
        let gain = gain as f64;
        dcg += ((2.0 as f64).powf(gain) - 1.0) / (i + 2.0).log2();
    }
    dcg
}

#[derive(Clone)]
pub struct NDCG {
    depth: Option<usize>,
    ideal_gains: Arc<HashMap<String, Option<f64>>>,
}
impl NDCG {
    pub fn new(
        depth: Option<usize>,
        dataset: &DatasetRef,
        judgments: Option<QuerySetJudgments>,
    ) -> Self {
        let mut query_norms: HashMap<String, Option<f64>> = HashMap::new();

        for (qid, instance_ids) in dataset.instances_by_query().iter() {
            // Determine the total number of relevant documents:
            let all_gains: Option<Vec<f32>> = judgments
                .as_ref()
                .and_then(|j| j.get(qid))
                .map(|data| data.gain_vector());
            // Calculate if unavailable in config:
            let ideal_gains: Vec<f32> = all_gains.unwrap_or_else(|| {
                instance_ids
                    .iter()
                    .map(|index| dataset.gain(*index))
                    .collect()
            });
            // Insert ideal if available:
            query_norms.insert(
                qid.clone(),
                if ideal_gains.iter().cloned().filter(|g| *g > 0.0).count() == 0 {
                    None
                } else {
                    Some(compute_dcg(&ideal_gains, depth, true))
                },
            );
        }

        Self {
            depth,
            ideal_gains: Arc::new(query_norms),
        }
    }
}

impl Evaluator for NDCG {
    fn name(&self) -> String {
        if let Some(depth) = self.depth {
            format!("NDCG@{}", depth)
        } else {
            String::from("NDCG")
        }
    }
    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64 {
        let actual_gain_vector: Vec<f32> = ranked_list.iter().map(|ri| ri.gain).collect();

        let normalizer = self.ideal_gains.get(qid).cloned().unwrap_or_else(|| {
            if actual_gain_vector
                .iter()
                .cloned()
                .filter(|g| *g > 0.0)
                .count()
                == 0
            {
                None
            } else {
                Some(compute_dcg(&actual_gain_vector, self.depth, true))
            }
        });

        if let Some(ideal_dcg) = normalizer {
            // Compute NDCG:
            let actual_dcg = compute_dcg(&actual_gain_vector, self.depth, false);
            if actual_dcg > ideal_dcg {
                panic!(
                    "qid: {}, actual_gain_vector: {:?} ideal_dcg: {}",
                    qid, actual_gain_vector, ideal_dcg
                )
            }
            actual_dcg / ideal_dcg
        } else {
            // If not gains, there's nothing to calculate.
            0.0
        }
    }
}

#[derive(Clone)]
pub struct AveragePrecision {
    /// Norms are the number of relevant by query for mAP.
    query_norms: Arc<HashMap<String, u32>>,
}

impl AveragePrecision {
    pub fn new(dataset: &DatasetRef, judgments: Option<QuerySetJudgments>) -> Self {
        let mut query_norms = HashMap::new();

        for (qid, instance_ids) in dataset.instances_by_query().iter() {
            // Determine the total number of relevant documents:
            let param_num_relevant: Option<u32> = judgments
                .as_ref()
                .and_then(|j| j.get(qid))
                .map(|data| data.num_relevant());
            // Calculate if unavailable in config:
            let num_relevant: u32 = param_num_relevant.unwrap_or_else(|| {
                instance_ids
                    .iter()
                    .filter(|index| dataset.gain(**index) > 0.0)
                    .count() as u32
            });

            if num_relevant > 0 {
                query_norms.insert(qid.clone(), num_relevant);
            }
        }

        Self {
            query_norms: Arc::new(query_norms),
        }
    }
}

impl Evaluator for AveragePrecision {
    fn name(&self) -> String {
        String::from("AP")
    }
    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64 {
        let num_relevant = self
            .query_norms
            .get(qid)
            .cloned()
            .unwrap_or_else(|| compute_num_relevant(ranked_list));

        compute_ap(ranked_list, num_relevant)
    }
}

pub fn compute_num_relevant(ranked_list: &[RankedInstance]) -> u32 {
    ranked_list.iter().filter(|ri| ri.is_relevant()).count() as u32
}

pub fn compute_ap(ranked_list: &[RankedInstance], num_relevant: u32) -> f64 {
    if num_relevant == 0 {
        return 0.0;
    }
    // Compute AP:
    let mut recall_points = 0;
    let mut sum_precision = 0.0;
    for rank in ranked_list
        .iter()
        .map(|ri| ri.is_relevant())
        .enumerate()
        .filter(|(_, rel)| *rel)
        .map(|(i, _)| i + 1)
    {
        recall_points += 1;
        sum_precision += f64::from(recall_points) / (rank as f64);
    }
    sum_precision / (num_relevant as f64)
}

pub fn compute_recip_rank(ranked_list: &[RankedInstance]) -> f64 {
    if let Some(rel_rank) = ranked_list
        .iter()
        .map(|ri| ri.is_relevant())
        .enumerate()
        .filter(|(_, rel)| *rel)
        .nth(0)
        .map(|(i, _)| i + 1)
    {
        return 1.0 / (rel_rank as f64);
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ri(score: f64, gain: f32, id: usize) -> RankedInstance {
        RankedInstance::new(score, gain, InstanceId::from_index(id))
    }
    #[test]
    fn test_rank_ties() {
        let mut instances = vec![
            ri(2.0, 0.0, 4),
            ri(2.0, 1.0, 3),
            ri(2.0, 2.0, 1),
            ri(2.0, 2.0, 2),
            ri(1.0, 2.0, 5),
        ];
        // getting: 5,4,3,1,2
        instances.sort();
        assert_eq!(
            vec![4, 3, 1, 2, 5],
            instances
                .into_iter()
                .map(|ri| ri.identifier.to_index())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_ap() {
        let instances = vec![
            ri(0.9, 1.0, 1),
            ri(0.7, 0.0, 2),
            ri(0.5, 0.0, 3),
            ri(0.4, 1.0, 4),
        ];
        let num_rel = compute_num_relevant(&instances);
        assert_eq!(num_rel, 2);
        let ap = compute_ap(&instances, num_rel);
        // mean(1, 1/2)
        assert_eq!(ap, 0.75);
    }

    const TREC_TOLERANCE: f64 = 0.00005;

    fn assert_trec_eq(x: f64, y: f64) {
        if (x - y).abs() > TREC_TOLERANCE {
            panic!("{} != {} at tolerance={}", x, y, TREC_TOLERANCE);
        }
    }

    #[test]
    fn test_compute_ndcg() {
        let data: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let ideal = compute_dcg(&data, None, true);
        let actual = compute_dcg(&data, None, false);

        assert_trec_eq(0.7328, actual / ideal);
    }
}
