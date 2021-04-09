/// This module defines classification and regression trees.
use crate::stats;
use crate::Scored;
use crate::{
    dataset::{DatasetRef, RankingDataset},
    sampling::DatasetSampling,
};
use crate::{model::TreeNode, normalizers::FeatureStats};
use crate::{FeatureId, InstanceId};
use oorandom::Rand64;
use ordered_float::NotNan;

/// Unsafe: const-fn: create a zero that's not a NaN.
const ZERO_F64: NotNan<f64> = unsafe { NotNan::unchecked_new(0.0) };

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Copy)]
pub enum SplitSelectionStrategy {
    SquaredError,
    BinaryGiniImpurity,
    InformationGain, // entropy
    TrueVarianceReduction,
}

impl SplitSelectionStrategy {
    pub fn importance(&self, lhs: &[NotNan<f32>], rhs: &[NotNan<f32>]) -> NotNan<f64> {
        match self {
            SplitSelectionStrategy::SquaredError => -(squared_error(lhs) + squared_error(rhs)),
            SplitSelectionStrategy::BinaryGiniImpurity => {
                let lhs_w = lhs.len() as f64;
                let rhs_w = rhs.len() as f64;
                let lhs_gini = gini_impurity(lhs) * lhs_w;
                let rhs_gini = gini_impurity(rhs) * rhs_w;
                // Negative so that we minimize the impurity across the splits.
                -(lhs_gini + rhs_gini)
            }
            SplitSelectionStrategy::InformationGain => {
                let lhs_w = lhs.len() as f64;
                let rhs_w = rhs.len() as f64;
                let lhs_e = entropy(lhs) * lhs_w;
                let rhs_e = entropy(rhs) * rhs_w;
                // Negative so that we minimize the entropy across the splits.
                -(lhs_e + rhs_e)
            }
            SplitSelectionStrategy::TrueVarianceReduction => {
                let lhs_w = lhs.len() as f64;
                let rhs_w = rhs.len() as f64;
                let lhs_variance = gain_stats(lhs).unwrap().variance * lhs_w;
                let rhs_variance = gain_stats(rhs).unwrap().variance * rhs_w;
                -NotNan::new(lhs_variance + rhs_variance).expect("variance NaN")
            }
        }
    }
}

pub fn compute_output(ids: &[InstanceId], dataset: &dyn RankingDataset) -> NotNan<f64> {
    if ids.len() == 0 {
        return ZERO_F64.clone();
    }
    let mut gain_sum = 0.0;
    for gain in ids.iter().cloned().map(|index| dataset.gain(index)) {
        gain_sum += gain.into_inner() as f64;
    }
    NotNan::new(gain_sum / (ids.len() as f64)).expect("Leaf output NaN.")
}

pub fn gain_stats(gains: &[NotNan<f32>]) -> Option<stats::ComputedStats> {
    let mut label_stats = stats::StreamingStats::new();
    for it in gains {
        label_stats.push(it.into_inner() as f64);
    }
    label_stats.finish()
}

fn average_gain(gains: &[NotNan<f32>]) -> NotNan<f64> {
    if gains.len() == 0 {
        return ZERO_F64;
    }
    let mut gain_sum = 0.0;
    for gain in gains {
        gain_sum += gain.into_inner() as f64;
    }
    NotNan::new(gain_sum / (gains.len() as f64)).expect("Leaf output NaN.")
}
fn squared_error(gains: &[NotNan<f32>]) -> NotNan<f64> {
    let output = average_gain(gains);

    let mut sum_sq_errors = 0.0;
    for gain in gains {
        let diff = output - f64::from(gain.into_inner());
        sum_sq_errors += (diff * diff).into_inner();
    }
    NotNan::new(sum_sq_errors).unwrap()
}
fn gini_impurity(gains: &[NotNan<f32>]) -> NotNan<f64> {
    if gains.len() == 0 {
        return ZERO_F64;
    }
    let count = gains.len() as f64;
    let positive = gains
        .iter()
        .cloned()
        .filter(|it| it.into_inner() > 0.0)
        .count();
    let p_yes = (positive as f64) / count;
    let p_no = (count - (positive as f64)) / count;
    let gini = p_yes * (1.0 - p_yes) + p_no * (1.0 - p_no);
    NotNan::new(gini).expect("gini was NaN")
}
fn plogp(x: f64) -> NotNan<f64> {
    if x == 0.0 {
        ZERO_F64
    } else {
        NotNan::new(x * x.log2()).expect("entropy/plogp returned NaN")
    }
}
fn entropy(gains: &[NotNan<f32>]) -> NotNan<f64> {
    if gains.len() == 0 {
        return ZERO_F64;
    }
    let count = gains.len() as f64;
    let positive = gains
        .iter()
        .cloned()
        .filter(|it| it.into_inner() > 0.0)
        .count();
    let p_yes = (positive as f64) / count;
    let p_no = (count - (positive as f64)) / count;
    let entropy = -plogp(p_yes) - plogp(p_no);
    entropy
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CARTParams {
    pub seed: u64,
    pub split_method: SplitSelectionStrategy,
    pub min_leaf_support: u32,
    pub split_candidates: Option<u32>,
    pub max_depth: u32,
    pub feature_tolerance: f64,
    pub only_use_features_once: bool,
}

impl Default for CARTParams {
    fn default() -> Self {
        let mut rand = Rand64::new(0xdeadbeef);
        Self {
            seed: rand.rand_u64(),
            split_method: SplitSelectionStrategy::SquaredError,
            min_leaf_support: 10,
            split_candidates: Some(3),
            max_depth: 8,
            feature_tolerance: 1E-6,
            only_use_features_once: true,
        }
    }
}

struct RecursionParams {
    current_depth: u32,
    only_use_features_once: bool,
}

impl RecursionParams {
    fn subset(&self) -> Self {
        Self {
            current_depth: self.current_depth + 1,
            only_use_features_once: self.only_use_features_once,
        }
    }
    fn done(&self, dataset: &DatasetRef) -> bool {
        dataset.features().is_empty() || dataset.instances().is_empty()
    }
    fn choose_split(
        &self,
        dataset: &DatasetRef,
        fsc: &FeatureSplitCandidate,
    ) -> (DatasetRef, DatasetRef) {
        let mut remaining_features = dataset.features();
        // Delete the feature we've used:
        if self.only_use_features_once {
            let pos = remaining_features
                .iter()
                .position(|fid| *fid == fsc.fid)
                .expect("Feature chosen still exists in list.");
            remaining_features.remove(pos);
        }

        (
            dataset.select(&fsc.lhs, &remaining_features).into_ref(),
            dataset.select(&fsc.rhs, &remaining_features).into_ref(),
        )
    }
    fn to_output(&self, dataset: &DatasetRef) -> NotNan<f64> {
        compute_output(&dataset.instances(), dataset)
    }
}

struct FeatureSplitCandidate {
    fid: FeatureId,
    split: NotNan<f64>,
    lhs: Vec<InstanceId>,
    rhs: Vec<InstanceId>,
    importance: NotNan<f64>,
}

#[derive(Debug)]
struct SplitCandidate {
    pos: usize,
    split: NotNan<f64>,
    importance: NotNan<f64>,
}

fn generate_split_candidate(
    params: &CARTParams,
    fid: FeatureId,
    dataset: &dyn RankingDataset,
    stats: &stats::ComputedStats,
) -> Option<FeatureSplitCandidate> {
    let instances = dataset.instances();
    // Is there any variance in this feature?
    if stats.min() + params.feature_tolerance > stats.max() {
        return None;
    }

    let mut instance_feature: Vec<Scored<InstanceId>> = instances
        .iter()
        .cloned()
        .map(|i| Scored::new(dataset.get_feature_value(i, fid).unwrap_or(0.0), i))
        .collect();
    instance_feature.sort_unstable();

    let scores: Vec<NotNan<f64>> = instance_feature.iter().map(|sf| sf.score).collect();
    let ids: Vec<InstanceId> = instance_feature.into_iter().map(|sf| sf.item).collect();
    let mut gains = Vec::with_capacity(instances.len());
    for id in instances.iter() {
        gains.push(dataset.gain(*id))
    }
    let label_stats = gain_stats(&gains)?;
    if label_stats.max == label_stats.min {
        return None;
    }

    let split_positions: Vec<Scored<usize>> = if let Some(k) = params.split_candidates {
        let range = stats.max - stats.min;
        let splits: Vec<NotNan<f64>> = (1..k)
            .map(|i| (i as f64) / (k as f64))
            .map(|f| NotNan::new(f * range + stats.min).unwrap())
            .collect();
        // collect instance index in ids/scores where the "splits" are.
        let mut split_positions: Vec<Scored<usize>> = Vec::new();
        let mut ids_i = 0;
        for position in splits.iter() {
            // linearly classify:
            while ids_i < ids.len() && scores[ids_i] < *position {
                ids_i += 1;
            }
            if let Some(prev) = split_positions.last() {
                if prev.item == ids_i {
                    continue;
                }
            }
            split_positions.push(Scored::new(position.into_inner(), ids_i));
        }
        split_positions
    } else {
        let mut split_positions: Vec<Scored<usize>> = Vec::with_capacity(scores.len() - 1);
        for i in 1..scores.len() - 1 {
            let rhs = scores[i + 1];
            if let Some(last) = split_positions.last() {
                if last.score + params.feature_tolerance > rhs {
                    continue;
                }
            }
            split_positions.push(Scored::new(rhs.into_inner(), i + 1));
        }
        split_positions
    };

    // evaluate the splits!
    let mut best: Option<SplitCandidate> = None;
    for scored_index in split_positions.into_iter() {
        let right_side = scored_index.item;
        let split = scored_index.score;
        if right_side < params.min_leaf_support as usize
            || (gains.len() - right_side) < params.min_leaf_support as usize
        {
            continue;
        }
        let (lhs, rhs) = gains.split_at(right_side);
        let importance = params.split_method.importance(lhs, rhs);
        if let Some(best) = best.as_mut() {
            let must_beat = best.importance;
            if importance > must_beat {
                best.pos = right_side;
                best.split = split;
                best.importance = importance;
            }
        } else {
            best = Some(SplitCandidate {
                pos: right_side,
                split,
                importance,
            })
        }
    }
    best.map(|sc| {
        let (lhs, rhs) = ids.split_at(sc.pos);
        FeatureSplitCandidate {
            fid,
            split: sc.split,
            importance: sc.importance,
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        }
    })
}

#[derive(Debug, Clone)]
pub enum NoTreeReason {
    StepDone,
    DepthExceeded,
    SplitTooSmall,
    NoFeatureSplitCandidates,
}

pub fn learn_cart_tree(params: &CARTParams, dataset: &DatasetRef) -> TreeNode {
    let step = RecursionParams {
        current_depth: 1,
        only_use_features_once: params.only_use_features_once,
    };

    let root = learn_recursive(params, dataset, &step);
    match root {
        Ok(tree) => tree,
        Err(_e) => {
            TreeNode::LeafNode(step.to_output(dataset))
            //panic!("{:?} fids={:?} N={}", _e, step.features, step.instances.len())
        }
    }
}

fn learn_recursive(
    params: &CARTParams,
    dataset: &DatasetRef,
    step: &RecursionParams,
) -> Result<TreeNode, NoTreeReason> {
    // Gone too deep:
    if step.done(dataset) {
        return Err(NoTreeReason::StepDone);
    }
    if step.current_depth >= params.max_depth {
        return Err(NoTreeReason::DepthExceeded);
    }
    // Cannot split further:
    if dataset.instances().len() < (params.min_leaf_support) as usize {
        return Err(NoTreeReason::SplitTooSmall);
    }

    let feature_stats = FeatureStats::compute(dataset).feature_stats;

    let mut candidates: Vec<FeatureSplitCandidate> = dataset
        .features()
        .iter()
        .flat_map(|fid| {
            feature_stats
                .get(fid)
                .and_then(|stats| generate_split_candidate(params, *fid, dataset, stats))
        })
        .collect();

    candidates.sort_unstable_by_key(|fsc| fsc.importance);
    if let Some(fsc) = candidates.last() {
        let (lhs_d, rhs_d) = step.choose_split(dataset, fsc);

        let left_child = learn_recursive(params, &lhs_d, &step.subset())
            .unwrap_or_else(|_| TreeNode::LeafNode(step.to_output(&lhs_d)));
        let right_child = learn_recursive(params, &rhs_d, &step.subset())
            .unwrap_or_else(|_| TreeNode::LeafNode(step.to_output(&rhs_d)));
        Ok(TreeNode::FeatureSplit {
            fid: fsc.fid,
            split: fsc.split,
            lhs: Box::new(left_child),
            rhs: Box::new(right_child),
        })
    } else {
        Err(NoTreeReason::NoFeatureSplitCandidates)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dataset::DatasetRef;
    use crate::instance::{Features, Instance};

    fn single_feature(x: f32) -> Features {
        Features::Dense32(vec![x])
    }
    const DELTA: f64 = 1e-5;
    fn assert_float_eq(attr: &str, x: f64, y: f64) {
        if (x - y).abs() > DELTA {
            panic!("{} failure: {} != {} at tolerance={}", attr, x, y, DELTA);
        }
    }
    #[test]
    fn test_regression_tree() {
        let xs: Vec<f32> = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            .iter()
            .map(|x| *x as f32)
            .collect();
        let ys: Vec<NotNan<f32>> = [7, 7, 7, 7, 2, 2, 2, 12, 12, 12]
            .iter()
            .map(|y| NotNan::new(*y as f32).unwrap())
            .collect();

        let training_instances: Vec<Instance> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| Instance::new(ys[i], "query".to_string(), None, single_feature(*x)))
            .collect();

        let dataset = DatasetRef::new(training_instances, None);
        let params = CARTParams {
            min_leaf_support: 1,
            max_depth: 10,
            split_candidates: Some(32),
            only_use_features_once: false,
            split_method: SplitSelectionStrategy::SquaredError,
            ..CARTParams::default()
        };
        let tree = learn_cart_tree(&params, &dataset);

        for inst in dataset.instances() {
            let py = dataset.score(inst, &tree);
            assert_float_eq(
                &format!(
                    "x={}",
                    dataset
                        .get_feature_value(inst, FeatureId::from_index(0))
                        .unwrap()
                ),
                *py,
                dataset.gain(inst).into_inner().into(),
            );
        }
    }
}
