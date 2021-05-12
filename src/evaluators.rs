use crate::dataset::{DatasetRef, RankingDataset};
use crate::heap::ScoringHeap;
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

pub struct ScoredDV<'a, 'b> {
    pub score: f64,
    pub index: usize,
    pub vectors: &'a DatasetVectors<'b>,
}

impl<'a, 'b> PartialEq for ScoredDV<'a, 'b> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(&other) == Ordering::Equal
    }
}

impl<'a, 'b> PartialOrd for ScoredDV<'a, 'b> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<'a, 'b> Eq for ScoredDV<'a, 'b> {}

/// Natural sort: first by socre, descending, then by gain ascending (yielding pessimistic scores on ties), finally by identifier.
impl<'a, 'b> Ord for ScoredDV<'a, 'b> {
    /// Treat NaN as ties:
    fn cmp(&self, other: &Self) -> Ordering {
        // score: desc
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Less) => return Ordering::Greater,
            Some(Ordering::Greater) => return Ordering::Less,
            _ => {}
        }
        // gain: asc
        match (self.vectors.gains[self.index]).partial_cmp(&other.vectors.gains[other.index]) {
            Some(Ordering::Less) => return Ordering::Less,
            Some(Ordering::Greater) => return Ordering::Greater,
            _ => {}
        }
        // identifier: id
        self.vectors.instances[self.index].cmp(&other.vectors.instances[other.index])
    }
}

pub struct DatasetVectors<'dataset> {
    _dataset: &'dataset dyn RankingDataset,
    pub instances: Vec<InstanceId>,
    pub gains: Vec<f32>,
    pub qid_to_index: HashMap<&'dataset str, Vec<usize>>,
    pub max_depth: usize,
}

impl<'d> DatasetVectors<'d> {
    pub fn new(dataset: &'d dyn RankingDataset) -> DatasetVectors<'d> {
        let query_ids: Vec<&'d str> = dataset.query_ids();
        let mut qid_to_indices: HashMap<&'d str, Vec<usize>> =
            HashMap::with_capacity(query_ids.len());
        for (i, qid) in query_ids.into_iter().enumerate() {
            qid_to_indices.entry(qid).or_default().push(i);
        }
        let max_depth = qid_to_indices
            .values()
            .map(|ids| ids.len())
            .max()
            .unwrap_or_default();
        Self {
            _dataset: dataset,
            instances: dataset.instances(),
            gains: dataset.gains(),
            qid_to_index: qid_to_indices,
            max_depth,
        }
    }
}

#[derive(Clone)]
pub struct SetEvaluator {
    dataset: DatasetRef,
    evaluator: Arc<dyn Evaluator>,
}

impl SetEvaluator {
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
                "ap" | "map" => Arc::new(AveragePrecision::new(depth, dataset, judgments.clone())),
                "rr" | "mrr" => Arc::new(ReciprocalRank(depth)),
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

    pub fn fast_eval2(&self, predictions: &[f64], vectors: &DatasetVectors<'_>) -> f64 {
        debug_assert_eq!(vectors.instances.len(), self.dataset.n_instances() as usize);

        let mut sum = 0.0;
        let mut n = 0;
        let depth = self.evaluator.depth().unwrap_or(vectors.max_depth);

        let mut heap: ScoringHeap<ScoredDV> = ScoringHeap::new(depth);
        let mut ranked_list: Vec<RankedInstance> = Vec::with_capacity(depth);
        for (qid, indices) in vectors.qid_to_index.iter() {
            ranked_list.clear();
            heap.clear();

            // Predict for every document:
            for index in indices.into_iter().cloned() {
                let score = predictions[index];
                heap.offer(ScoredDV {
                    score,
                    index,
                    vectors,
                });
            }
            // Sort largest to smallest:
            heap.drain_into_mapping(
                |sdv| {
                    RankedInstance::new(
                        sdv.score,
                        vectors.gains[sdv.index],
                        vectors.instances[sdv.index],
                    )
                },
                &mut ranked_list,
            );
            debug_assert!(ranked_list[0] < ranked_list[1]);
            sum += self.evaluator.score(&qid, &ranked_list);
            n += 1;
        }
        if n == 0 {
            return 0.0;
        }
        return sum / (n as f64);
    }

    pub fn fast_eval<'d>(
        &self,
        model: &dyn Model,
        queries: &[&'d str],
        vectors: &DatasetVectors<'d>,
    ) -> f64 {
        debug_assert_eq!(vectors.instances.len(), self.dataset.n_instances() as usize);

        let mut sum = 0.0;
        let mut n = 0;
        let depth = self.evaluator.depth().unwrap_or(vectors.max_depth);

        let mut heap: ScoringHeap<ScoredDV> = ScoringHeap::new(depth);
        let mut ranked_list: Vec<RankedInstance> = Vec::with_capacity(depth);
        for qid in queries.iter() {
            let indices = &vectors.qid_to_index[qid];
            ranked_list.clear();
            heap.clear();

            // Predict for every document:
            for index in indices.into_iter().cloned() {
                let id = vectors.instances[index];
                let score = self.dataset.score(id, model);
                //let gain = vectors.gains[index];
                heap.offer(ScoredDV {
                    score,
                    index,
                    vectors,
                });
            }
            // Sort largest to smallest:
            heap.drain_into_mapping(
                |sdv| {
                    RankedInstance::new(
                        sdv.score,
                        vectors.gains[sdv.index],
                        vectors.instances[sdv.index],
                    )
                },
                &mut ranked_list,
            );
            debug_assert!(ranked_list[0] < ranked_list[1]);
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

impl Evaluator for SetEvaluator {
    fn name(&self) -> String {
        self.evaluator.name()
    }

    fn depth(&self) -> Option<usize> {
        self.evaluator.depth()
    }

    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64 {
        self.evaluator.score(qid, ranked_list)
    }
}

pub trait Evaluator: Send + Sync {
    fn name(&self) -> String;
    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64;
    fn depth(&self) -> Option<usize>;
}

#[derive(Clone)]
pub struct ReciprocalRank(Option<usize>);

impl Evaluator for ReciprocalRank {
    fn name(&self) -> String {
        if let Some(depth) = self.0 {
            format!("RR@{}", depth)
        } else {
            String::from("RR")
        }
    }
    fn score(&self, _qid: &str, ranked_list: &[RankedInstance]) -> f64 {
        compute_recip_rank(ranked_list, self.0)
    }

    fn depth(&self) -> Option<usize> {
        self.0
    }
}

/// Autogenerated via python; first 64: ``1/log2(int)`` precomputed.
const INT_INV_LOG2: [f64; 64] = [
    0.0,
    f64::NAN,
    1.0,
    0.6309297535714575,
    0.5,
    0.43067655807339306,
    0.38685280723454163,
    0.3562071871080222,
    0.3333333333333333,
    0.31546487678572877,
    0.3010299956639812,
    0.2890648263178879,
    0.27894294565112987,
    0.27023815442731974,
    0.26264953503719357,
    0.2559580248098155,
    0.25,
    0.24465054211822604,
    0.23981246656813146,
    0.23540891336663824,
    0.23137821315975915,
    0.227670248696953,
    0.22424382421757544,
    0.22106472945750374,
    0.21810429198553155,
    0.21533827903669653,
    0.21274605355336318,
    0.2103099178571525,
    0.20801459767650948,
    0.20584683246043448,
    0.2037950470905062,
    0.20184908658209985,
    0.2,
    0.19823986317056053,
    0.1965616322328226,
    0.1949590218937863,
    0.19342640361727081,
    0.19195872000656014,
    0.1905514124267734,
    0.18920035951687003,
    0.18790182470910757,
    0.18665241123894338,
    0.1854490234153689,
    0.18428883314870617,
    0.18316925091363362,
    0.18208790046993825,
    0.1810425967800402,
    0.18003132665669264,
    0.17905223175104137,
    0.1781035935540111,
    0.17718382013555792,
    0.17629143438888212,
    0.17542506358195453,
    0.17458343004804494,
    0.17376534287144002,
    0.1729696904450771,
    0.17219543379409813,
    0.17144160057391347,
    0.17070727966372012,
    0.16999161628691403,
    0.16929380759878143,
    0.1686130986895011,
    0.16794877895704194,
    0.16730017881017412,
];

#[inline(always)]
fn fast_int_invlog2(x: usize) -> f64 {
    INT_INV_LOG2
        .get(x)
        .cloned()
        .unwrap_or_else(|| 1.0 / (x as f64).log2())
}

pub fn compute_dcg_slice(gains: &[f32], depth: Option<usize>) -> f64 {
    // Gain of 0.0 is a positive value, so we need to expand or contact to "depth" if it's given.
    let depth = match depth {
        Some(x) => std::cmp::min(gains.len(), x),
        None => gains.len(),
    };
    let mut dcg = 0.0;
    for (i, gain) in gains[..depth]
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, g)| *g > 0.0)
    {
        //let gain = *gain as f64;
        //let pow_gain = (1 << (*gain as u32) - 1) as f64;
        dcg += ((2.0f64).powf(gain as f64) - 1.0) * fast_int_invlog2(i + 2);
    }
    dcg
}

pub fn compute_dcg(gains: &[f32], depth: Option<usize>, ideal: bool) -> f64 {
    if !ideal {
        return compute_dcg_slice(gains, depth);
    }
    // Gain of 0.0 is a positive value, so we need to expand or contact to "depth" if it's given.
    let mut gain_vector: Vec<f32> = gains.to_vec();
    if ideal {
        gain_vector.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        gain_vector.reverse();
    }
    compute_dcg_slice(&gain_vector, depth)
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
        let actual_gain_vector: Vec<f32> = ranked_list
            .iter()
            .take(self.depth.unwrap_or(ranked_list.len()))
            .map(|ri| ri.gain)
            .collect();

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
            let actual_dcg = compute_dcg_slice(&actual_gain_vector, self.depth);
            debug_assert!(
                actual_dcg <= ideal_dcg,
                "actual {:.3}, ideal {:.3}",
                actual_dcg,
                ideal_dcg
            );
            actual_dcg / ideal_dcg
        } else {
            // If not gains, there's nothing to calculate.
            0.0
        }
    }

    fn depth(&self) -> Option<usize> {
        self.depth
    }
}

#[derive(Clone)]
pub struct AveragePrecision {
    depth: Option<usize>,
    /// Norms are the number of relevant by query for mAP.
    query_norms: Arc<HashMap<String, u32>>,
}

impl AveragePrecision {
    pub fn new(
        depth: Option<usize>,
        dataset: &DatasetRef,
        judgments: Option<QuerySetJudgments>,
    ) -> Self {
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
            depth,
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
        if let Some(limit) = self.depth {
            compute_ap(
                &ranked_list[..std::cmp::min(limit, ranked_list.len())],
                std::cmp::min(limit as u32, num_relevant),
            )
        } else {
            compute_ap(ranked_list, num_relevant)
        }
    }
    fn depth(&self) -> Option<usize> {
        self.depth
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

pub fn compute_recip_rank(ranked_list: &[RankedInstance], depth: Option<usize>) -> f64 {
    if let Some(rel_rank) = ranked_list
        .iter()
        .take(depth.unwrap_or(ranked_list.len()))
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
    fn test_rank_ties_heap() {
        let instances = vec![
            ri(2.0, 0.0, 4),
            ri(2.0, 1.0, 3),
            ri(2.0, 2.0, 1),
            ri(2.0, 2.0, 2),
            ri(1.0, 2.0, 5),
        ];
        let mut scoring_heap = ScoringHeap::new(3);
        for inst in instances {
            scoring_heap.offer(inst);
        }
        let output = scoring_heap.into_vec();
        assert_eq!(
            vec![4, 3, 1],
            output
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
