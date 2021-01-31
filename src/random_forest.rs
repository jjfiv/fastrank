use crate::dataset::{DatasetRef, RankingDataset};
use crate::evaluators::SetEvaluator;
use crate::model::{ModelEnum, TreeNode, WeightedEnsemble};
use crate::normalizers::FeatureStats;
use crate::sampling::DatasetSampling;
use crate::stats;
use crate::Scored;
use crate::{FeatureId, InstanceId};
use ordered_float::NotNan;
use oorandom::Rand64;
use rayon::prelude::*;
use std::cmp;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SplitSelectionStrategy {
    SquaredError(),
    BinaryGiniImpurity(),
    InformationGain(), // entropy
    TrueVarianceReduction(),
}

pub fn label_stats(
    instances: &[InstanceId],
    dataset: &dyn RankingDataset,
) -> Option<stats::ComputedStats> {
    let mut label_stats = stats::StreamingStats::new();
    for index in instances.iter().cloned() {
        label_stats.push(dataset.gain(index).into_inner() as f64);
    }
    label_stats.finish()
}
fn compute_output(ids: &[InstanceId], dataset: &dyn RankingDataset) -> NotNan<f64> {
    if ids.len() == 0 {
        return NotNan::new(0.0).unwrap();
    }
    let mut gain_sum = 0.0;
    for gain in ids.iter().cloned().map(|index| dataset.gain(index)) {
        gain_sum += gain.into_inner() as f64;
    }
    NotNan::new(gain_sum / (ids.len() as f64)).expect("Leaf output NaN.")
}
fn squared_error(ids: &[InstanceId], dataset: &dyn RankingDataset) -> NotNan<f64> {
    let output = compute_output(ids, dataset);

    let mut sum_sq_errors = 0.0;
    for gain in ids.iter().cloned().map(|index| dataset.gain(index)) {
        let diff = output - f64::from(gain.into_inner());
        sum_sq_errors += (diff * diff).into_inner();
    }
    NotNan::new(sum_sq_errors).unwrap()
}
fn gini_impurity(ids: &[InstanceId], dataset: &dyn RankingDataset) -> NotNan<f64> {
    if ids.len() == 0 {
        return NotNan::new(0.0).unwrap();
    }
    let count = ids.len() as f64;
    let positive = ids
        .iter()
        .cloned()
        .filter(|index| dataset.gain(*index).into_inner() > 0.0)
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
fn entropy(ids: &[InstanceId], dataset: &dyn RankingDataset) -> NotNan<f64> {
    if ids.len() == 0 {
        return NotNan::new(0.0).unwrap();
    }
    let count = ids.len() as f64;
    let positive = ids
        .iter()
        .cloned()
        .filter(|index| dataset.gain(*index).into_inner() > 0.0)
        .count();
    let p_yes = (positive as f64) / count;
    let p_no = (count - (positive as f64)) / count;
    let entropy = -plogp(p_yes) - plogp(p_no);
    entropy
}

impl SplitSelectionStrategy {
    fn importance(
        &self,
        lhs: &[InstanceId],
        rhs: &[InstanceId],
        dataset: &dyn RankingDataset,
    ) -> NotNan<f64> {
        match self {
            SplitSelectionStrategy::SquaredError() => {
                -(squared_error(lhs, dataset) + squared_error(rhs, dataset))
            }
            SplitSelectionStrategy::BinaryGiniImpurity() => {
                let lhs_w = lhs.len() as f64;
                let rhs_w = rhs.len() as f64;
                let lhs_gini = gini_impurity(lhs, dataset) * lhs_w;
                let rhs_gini = gini_impurity(rhs, dataset) * rhs_w;
                // Negative so that we minimize the impurity across the splits.
                -(lhs_gini + rhs_gini)
            }
            SplitSelectionStrategy::InformationGain() => {
                let lhs_w = lhs.len() as f64;
                let rhs_w = rhs.len() as f64;
                let lhs_e = entropy(lhs, dataset) * lhs_w;
                let rhs_e = entropy(rhs, dataset) * rhs_w;
                // Negative so that we minimize the entropy across the splits.
                -(lhs_e + rhs_e)
            }
            SplitSelectionStrategy::TrueVarianceReduction() => {
                let lhs_w = lhs.len() as f64;
                let rhs_w = rhs.len() as f64;
                let lhs_variance = label_stats(lhs, dataset).unwrap().variance * lhs_w;
                let rhs_variance = label_stats(rhs, dataset).unwrap().variance * rhs_w;
                -NotNan::new(lhs_variance + rhs_variance).expect("variance NaN")
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
        let mut rand = Rand64::new(0xdeadbeef);
        Self {
            weight_trees: false,
            seed: rand.rand_u64(),
            split_method: SplitSelectionStrategy::SquaredError(),
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

impl TreeNode {
    fn depth(&self) -> u32 {
        match self {
            TreeNode::LeafNode(_) => 1,
            TreeNode::FeatureSplit { lhs, rhs, .. } => 1 + cmp::max(lhs.depth(), rhs.depth()),
        }
    }
}

struct RecursionParams {
    current_depth: u32,
}

impl RecursionParams {
    fn subset(&self) -> Self {
        Self {
            current_depth: self.current_depth + 1,
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
        (
            dataset.with_instances(&fsc.lhs).into_ref(),
            dataset.with_instances(&fsc.rhs).into_ref(),
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
    params: &RandomForestParams,
    fid: FeatureId,
    dataset: &dyn RankingDataset,
    stats: &stats::ComputedStats,
) -> Option<FeatureSplitCandidate> {
    let instances = dataset.instances();
    let label_stats = label_stats(&instances, dataset)?;
    if label_stats.max == label_stats.min {
        return None;
    }
    let k = params.split_candidates;
    let range = stats.max - stats.min;

    let mut instance_feature: Vec<Scored<InstanceId>> = instances
        .iter()
        .cloned()
        .map(|i| Scored::new(dataset.get_feature_value(i, fid).unwrap_or(0.0), i))
        .collect();
    instance_feature.sort_unstable();

    let scores: Vec<NotNan<f64>> = instance_feature.iter().map(|sf| sf.score).collect();
    let ids: Vec<InstanceId> = instance_feature.into_iter().map(|sf| sf.item).collect();

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
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
        }
    })
}

pub fn learn_ensemble(
    params: &RandomForestParams,
    dataset: &DatasetRef,
    evaluator: &SetEvaluator,
) -> WeightedEnsemble {
    let mut rand = Rand64::new(params.seed.into());
    let seeds: Vec<(u32, u64)> = (0..params.num_trees)
        .map(|i| (i, rand.rand_u64()))
        .collect();

    let mut trees: Vec<Scored<TreeNode>> = Vec::new();
    if !params.quiet {
        println!("-----------------------");
        println!("|{:>7}|{:>7}|{:>7}|", "Tree", "Depth", evaluator.name());
        println!("-----------------------");
    }

    trees.par_extend(seeds.par_iter().map(|(idx, rand_seed)| {
        let mut local_rand = Rand64::new((*rand_seed).into());
        let subsample = dataset
            .random_sample(
                params.feature_sampling_rate,
                params.instance_sampling_rate,
                &mut local_rand,
            )
            .into_ref();
        let tree = learn_decision_tree(params, &subsample);
        let eval = evaluator.evaluate_mean(&tree);
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
                let m = ModelEnum::DecisionTree(tree.item);
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

pub fn learn_decision_tree(params: &RandomForestParams, dataset: &DatasetRef) -> TreeNode {
    let step = RecursionParams { current_depth: 1 };

    let root = learn_recursive(params, dataset, &step);
    match root {
        Ok(tree) => tree,
        Err(_e) => {
            TreeNode::LeafNode(step.to_output(dataset))
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
    use crate::dataset;
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
    fn test_random_forest_determinism() {
        let feature_names =
            dataset::load_feature_names_json("examples/trec_news_2018.features.json").unwrap();
        let train_dataset = dataset::LoadedRankingDataset::load_libsvm(
            "examples/trec_news_2018.train",
            Some(&feature_names),
        )
        .unwrap()
        .into_ref();
        let params = RandomForestParams {
            num_trees: 10,
            seed: 42,
            min_leaf_support: 1,
            quiet: true,
            max_depth: 10,
            split_candidates: 32,
            split_method: SplitSelectionStrategy::SquaredError(),
            ..RandomForestParams::default()
        };

        let eval = SetEvaluator::create(&train_dataset, "ndcg@5", None).unwrap();
        let mut means = Vec::new();
        for i in 0..10 {
            let model = learn_ensemble(&params, &train_dataset, &eval);
            means.push(eval.evaluate_mean(&model));
            if i > 0 {
                assert_float_eq(
                    &format!("means[{}] == means[{}]", i - 1, i),
                    means[i - 1],
                    means[i],
                );
            }
        }
        // If this assertion fails and you're OK with it, you just broke SemVer; upgrade major version.
        assert_float_eq("means[0] = predefined", means[0], 0.4367914517387043);
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
        let params = RandomForestParams {
            num_trees: 1,
            min_leaf_support: 1,
            max_depth: 10,
            split_candidates: 32,
            split_method: SplitSelectionStrategy::SquaredError(),
            ..RandomForestParams::default()
        };
        let tree = learn_decision_tree(&params, &dataset);

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
