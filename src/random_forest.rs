use crate::dataset::*;
use crate::evaluators::Evaluator;
use crate::model::{Model, WeightedEnsemble};
use crate::stats;
use crate::Scored;
use ordered_float::NotNan;
use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use std::cmp;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplitSelectionStrategy {
    SquaredError(),
    DifferenceInLabelMeans(),
    MinLabelStddev(),
    BinaryGiniImpurity(),
    InformationGain(), // entropy
    TrueVarianceReduction(),
}

fn compute_output(ids: &[u32], dataset: &RankingDataset) -> NotNan<f64> {
    if ids.len() == 0 {
        return NotNan::new(0.0).unwrap();
    }
    let mut gain_sum = 0.0;
    for gain in ids
        .iter()
        .cloned()
        .map(|index| dataset.instances[index as usize].gain)
    {
        gain_sum += gain.into_inner() as f64;
    }
    NotNan::new(gain_sum / (ids.len() as f64)).expect("Leaf output NaN.")
}
fn squared_error(ids: &[u32], dataset: &RankingDataset) -> NotNan<f64> {
    let output = compute_output(ids, dataset);

    let mut sum_sq_errors = 0.0;
    for gain in ids
        .iter()
        .cloned()
        .map(|index| dataset.instances[index as usize].gain)
    {
        let diff = output - f64::from(cmp::max(NotNan::new(1.0).unwrap(), gain).into_inner());
        sum_sq_errors += (diff * diff).into_inner();
    }
    NotNan::new(sum_sq_errors).unwrap()
}
fn gini_impurity(ids: &[u32], dataset: &RankingDataset) -> NotNan<f64> {
    if ids.len() == 0 {
        return NotNan::new(0.0).unwrap();
    }
    let count = ids.len() as f64;
    let positive = ids
        .iter()
        .cloned()
        .filter(|index| dataset.instances[*index as usize].is_relevant())
        .count();
    let p_yes = (positive as f64) / count;
    let p_no = (count - (positive as f64)) / count;
    let gini = p_yes * (1.0 - p_yes) + p_no * (1.0 - p_no);
    NotNan::new(gini).expect("gini was NaN")
}
fn plogp(x: f64) -> NotNan<f64> {
    if x == 0.0 {
        NotNan::new(0.0).unwrap()
    } else {
        NotNan::new(x * x.log2()).expect("entropy/plogp returned NaN")
    }
}
fn entropy(ids: &[u32], dataset: &RankingDataset) -> NotNan<f64> {
    if ids.len() == 0 {
        return NotNan::new(0.0).unwrap();
    }
    let count = ids.len() as f64;
    let positive = ids
        .iter()
        .cloned()
        .filter(|index| dataset.instances[*index as usize].is_relevant())
        .count();
    let p_yes = (positive as f64) / count;
    let p_no = (count - (positive as f64)) / count;
    let entropy = -plogp(p_yes) - plogp(p_no);
    entropy
}

impl SplitSelectionStrategy {
    fn importance(&self, lhs: &[u32], rhs: &[u32], dataset: &RankingDataset) -> NotNan<f64> {
        match self {
            SplitSelectionStrategy::SquaredError() => {
                -(squared_error(lhs, dataset) + squared_error(rhs, dataset))
            }
            SplitSelectionStrategy::DifferenceInLabelMeans() => {
                let lhs_stats = dataset.label_stats(lhs).unwrap();
                let rhs_stats = dataset.label_stats(rhs).unwrap();
                NotNan::new((lhs_stats.mean() - rhs_stats.mean()).abs()).unwrap()
            }
            SplitSelectionStrategy::MinLabelStddev() => {
                let lhs_stddev = dataset.label_stats(lhs).unwrap().stddev();
                let rhs_stddev = dataset.label_stats(rhs).unwrap().stddev();
                // Negative so we minimize the standard deviations.
                -cmp::min(lhs_stddev, rhs_stddev)
            }
            SplitSelectionStrategy::BinaryGiniImpurity() => {
                let total = lhs.len() + rhs.len();
                let lhs_w = (lhs.len() as f64) / (total as f64);
                let rhs_w = (rhs.len() as f64) / (total as f64);
                let lhs_gini = gini_impurity(lhs, dataset) * lhs_w;
                let rhs_gini = gini_impurity(rhs, dataset) * rhs_w;
                // Negative so that we minimize the impurity across the splits.
                -(lhs_gini + rhs_gini)
            }
            SplitSelectionStrategy::InformationGain() => {
                let total = lhs.len() + rhs.len();
                let lhs_w = (lhs.len() as f64) / (total as f64);
                let rhs_w = (rhs.len() as f64) / (total as f64);
                let lhs_e = entropy(lhs, dataset) * lhs_w;
                let rhs_e = entropy(rhs, dataset) * rhs_w;
                // Negative so that we minimize the entropy across the splits.
                -(lhs_e + rhs_e)
            }
            SplitSelectionStrategy::TrueVarianceReduction() => {
                let total = lhs.len() + rhs.len();
                let lhs_w = (lhs.len() as f64) / (total as f64);
                let rhs_w = (rhs.len() as f64) / (total as f64);
                let lhs_variance = dataset.label_stats(lhs).unwrap().variance * lhs_w;
                let rhs_variance = dataset.label_stats(rhs).unwrap().variance * rhs_w;
                -NotNan::new(lhs_variance + rhs_variance).expect("variance NaN")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct RandomForestParams {
    pub seed: u64,
    pub quiet: bool,
    pub num_trees: u32,
    pub weight_trees: bool,
    pub split_method: SplitSelectionStrategy,
    pub instance_sampling_rate: f64,
    pub feature_sampling_rate: f64,
    pub min_leaf_support: u32,
    pub split_candidates: u32,
    pub max_depth: u32,
}

impl Default for RandomForestParams {
    fn default() -> Self {
        Self {
            weight_trees: false,
            seed: thread_rng().next_u64(),
            split_method: SplitSelectionStrategy::DifferenceInLabelMeans(),
            quiet: false,
            num_trees: 100,
            instance_sampling_rate: 0.5,
            feature_sampling_rate: 0.25,
            min_leaf_support: 10,
            split_candidates: 3,
            max_depth: 8,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum TreeNode {
    FeatureSplit {
        fid: u32,
        split: NotNan<f64>,
        lhs: Box<TreeNode>,
        rhs: Box<TreeNode>,
    },
    LeafNode(NotNan<f64>),
}

impl TreeNode {
    fn depth(&self) -> u32 {
        match self {
            TreeNode::LeafNode(_) => 1,
            TreeNode::FeatureSplit { lhs, rhs, .. } => 1 + cmp::max(lhs.depth(), rhs.depth()),
        }
    }
}

impl Model for TreeNode {
    fn score(&self, features: &Features) -> NotNan<f64> {
        match self {
            TreeNode::LeafNode(score) => score.clone(),
            TreeNode::FeatureSplit {
                fid,
                split,
                lhs,
                rhs,
            } => {
                let fval =
                    NotNan::new(features.get(*fid).unwrap_or(0.0)).expect("NaN in feature eval...");
                if fval <= *split {
                    lhs.score(features)
                } else {
                    rhs.score(features)
                }
            }
        }
    }
}

struct RecursionParams {
    current_depth: u32,
    features: Vec<u32>,
    instances: Vec<u32>,
}

impl RecursionParams {
    fn subset(&self, instances: Vec<u32>) -> Self {
        Self {
            current_depth: self.current_depth + 1,
            features: self.features.clone(),
            instances,
        }
    }
    fn done(&self) -> bool {
        self.features.is_empty() || self.instances.is_empty()
    }
    fn choose_split(&self, fsc: &FeatureSplitCandidate) -> (RecursionParams, RecursionParams) {
        (self.subset(fsc.lhs.clone()), self.subset(fsc.rhs.clone()))
    }
    fn to_output(&self, dataset: &RankingDataset) -> NotNan<f64> {
        compute_output(&self.instances, dataset)
    }
}

struct FeatureSplitCandidate {
    fid: u32,
    split: NotNan<f64>,
    lhs: Vec<u32>,
    rhs: Vec<u32>,
    importance: NotNan<f64>,
}

#[derive(Debug)]
struct SplitCandidate {
    pos: usize,
    split: NotNan<f64>,
    importance: NotNan<f64>,
}

fn generate_split_candidate(
    params: &RandomForestParams,
    fid: u32,
    instances: &[u32],
    dataset: &RankingDataset,
    stats: &stats::ComputedStats,
) -> Option<FeatureSplitCandidate> {
    let label_stats = dataset.label_stats(instances)?;
    if label_stats.max == label_stats.min {
        return None;
    }
    let k = params.split_candidates;
    let range = stats.max - stats.min;

    let mut instance_feature: Vec<Scored<u32>> = instances
        .iter()
        .cloned()
        .map(|i| {
            Scored::new(
                dataset.instances[i as usize]
                    .features
                    .get(fid)
                    .unwrap_or(0.0),
                i,
            )
        })
        .collect();
    instance_feature.sort_unstable();

    let scores: Vec<NotNan<f64>> = instance_feature.iter().map(|sf| sf.score).collect();
    let ids: Vec<u32> = instance_feature.into_iter().map(|sf| sf.item).collect();

    // TODO all splits instead...
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
    // evaluate the splits!
    let mut best = Vec::new();
    for scored_index in split_positions.into_iter() {
        let right_side = scored_index.item;
        let split = scored_index.score;
        let (lhs, rhs) = ids.split_at(right_side);
        if lhs.len() < params.min_leaf_support as usize
            || rhs.len() < params.min_leaf_support as usize
        {
            continue;
        }
        let importance = params.split_method.importance(lhs, rhs, dataset);
        best.push(SplitCandidate {
            pos: right_side,
            split,
            importance,
        });
    }

    best.sort_unstable_by_key(|sc| sc.importance);
    best.last().map(|sc| {
        let (lhs, rhs) = ids.split_at(sc.pos);
        FeatureSplitCandidate {
            fid,
            split: sc.split,
            importance: sc.importance,
            lhs: lhs.iter().cloned().collect(),
            rhs: rhs.iter().cloned().collect(),
        }
    })
}

pub fn learn_ensemble(
    params: &RandomForestParams,
    dataset: &RankingDataset,
    evaluator: &Evaluator,
) -> WeightedEnsemble {
    let mut rand = Xoshiro256StarStar::seed_from_u64(params.seed);
    let seeds: Vec<(u32, u64)> = (0..params.num_trees)
        .map(|i| (i, rand.next_u64()))
        .collect();

    let mut trees: Vec<Scored<TreeNode>> = Vec::new();
    if !params.quiet {
        println!("-----------------------");
        println!("|{:>7}|{:>7}|{:>7}|", "Tree", "Depth", evaluator.name());
        println!("-----------------------");
    }

    trees.par_extend(seeds.into_par_iter().map(|(idx, rand_seed)| {
        let tree = learn_decision_tree(rand_seed, params, dataset);
        let eval = dataset.evaluate_mean(&tree, evaluator);
        if !params.quiet {
            println!("|{:>7}|{:>7}|{:>7.3}|", idx + 1, tree.depth(), eval);
        }
        Scored::new(eval, tree)
    }));

    if !params.quiet {
        println!("-----------------------");
    }

    WeightedEnsemble::new(
        trees
            .into_iter()
            .map(|tree| {
                let m: Arc<dyn Model> = Arc::new(tree.item);
                Scored::new(
                    if params.weight_trees {
                        tree.score.into_inner()
                    } else {
                        1.0
                    },
                    m,
                )
            })
            .collect(),
    )
}

pub fn learn_decision_tree(
    rand_seed: u64,
    params: &RandomForestParams,
    dataset: &RankingDataset,
) -> TreeNode {
    let mut rand = Xoshiro256StarStar::seed_from_u64(rand_seed);
    let n_features = cmp::max(
        1,
        ((dataset.features.len() as f64) * params.feature_sampling_rate) as usize,
    );
    let n_instances = cmp::max(
        params.min_leaf_support as usize,
        ((dataset.instances.len() as f64) * params.instance_sampling_rate) as usize,
    );
    let step = RecursionParams {
        features: dataset
            .features
            .choose_multiple(&mut rand, n_features)
            .cloned()
            .collect(),
        instances: (0..dataset.instances.len())
            .map(|i| i as u32)
            .choose_multiple(&mut rand, n_instances),
        current_depth: 1,
    };

    let root = learn_recursive(params, dataset, &step);
    match root {
        Ok(tree) => tree,
        Err(_e) => {
            TreeNode::LeafNode(compute_output(&step.instances, dataset))
            //panic!("{:?} fids={:?} N={}", _e, step.features, step.instances.len())
        }
    }
}

#[derive(Debug, Clone)]
enum NoTreeReason {
    StepDone,
    DepthExceeded,
    SplitTooSmall,
    NoFeatureSplitCandidates,
}

fn learn_recursive(
    params: &RandomForestParams,
    dataset: &RankingDataset,
    step: &RecursionParams,
) -> Result<TreeNode, NoTreeReason> {
    // Gone too deep:
    if step.done() {
        return Err(NoTreeReason::StepDone);
    }
    if step.current_depth >= params.max_depth {
        return Err(NoTreeReason::DepthExceeded);
    }
    // Cannot split further:
    if step.instances.len() < (params.min_leaf_support * 2) as usize {
        return Err(NoTreeReason::SplitTooSmall);
    }

    let feature_stats = dataset
        .compute_feature_subsets(&step.features, &step.instances)
        .feature_stats;

    let mut candidates: Vec<FeatureSplitCandidate> = step
        .features
        .iter()
        .flat_map(|fid| {
            feature_stats.get(fid).and_then(|stats| {
                generate_split_candidate(params, *fid, &step.instances, dataset, stats)
            })
        })
        .collect();

    candidates.sort_unstable_by_key(|fsc| fsc.importance);
    if let Some(fsc) = candidates.last() {
        let (lhs_p, rhs_p) = step.choose_split(fsc);

        let left_child = learn_recursive(params, dataset, &lhs_p)
            .unwrap_or(TreeNode::LeafNode(lhs_p.to_output(dataset)));
        let right_child = learn_recursive(params, dataset, &rhs_p)
            .unwrap_or(TreeNode::LeafNode(rhs_p.to_output(dataset)));
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
            .into_iter()
            .map(|x| *x as f32)
            .collect();
        let ys: Vec<NotNan<f32>> = [7, 7, 7, 7, 2, 2, 2, 12, 12, 12]
            .into_iter()
            .map(|y| NotNan::new(*y as f32).unwrap())
            .collect();

        let training_instances: Vec<TrainingInstance> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| TrainingInstance::new(ys[i], "query".to_string(), single_feature(*x)))
            .collect();

        let dataset = RankingDataset::new(training_instances, None);
        let params = RandomForestParams {
            instance_sampling_rate: 1.0,
            feature_sampling_rate: 1.0,
            num_trees: 1,
            min_leaf_support: 1,
            max_depth: 10,
            split_candidates: 32,
            split_method: SplitSelectionStrategy::SquaredError(),
            ..RandomForestParams::default()
        };
        let tree = learn_decision_tree(13, &params, &dataset);

        for inst in dataset.instances.iter() {
            let py = tree.score(&inst.features);
            assert_float_eq(
                &format!("x={}", inst.features.get(0).unwrap()),
                *py,
                inst.gain.into_inner().into(),
            );
        }
    }
}
