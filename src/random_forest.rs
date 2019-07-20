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
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SplitSelectionStrategy {
    DifferenceInLabelMeans(),
    MinLabelStddev(),
    BinaryGiniImpurity(),
    InformationGain(), // entropy
    TrueVarianceReduction(),
}

impl SplitSelectionStrategy {
    fn importance(&self, lhs: &[u32], rhs: &[u32], dataset: &RankingDataset) -> NotNan<f64> {
        let worst_case = NotNan::new(std::f64::MIN).unwrap();
        if lhs.len() < 2 || rhs.len() < 2 {
            return worst_case;
        }
        match self {
            SplitSelectionStrategy::DifferenceInLabelMeans() => {
                let lhs_stats = dataset.label_stats(lhs).unwrap();
                let rhs_stats = dataset.label_stats(rhs).unwrap();
                NotNan::new((lhs_stats.mean() - rhs_stats.mean()).abs()).unwrap()
            }
            SplitSelectionStrategy::MinLabelStddev() => {
                let lhs_stddev = dataset.label_stats(lhs).unwrap().stddev();
                let rhs_stddev = dataset.label_stats(rhs).unwrap().stddev();
                -cmp::min(lhs_stddev, rhs_stddev)
            }
            SplitSelectionStrategy::BinaryGiniImpurity() => panic!("TODO"),
            SplitSelectionStrategy::InformationGain() => panic!("TODO"),
            SplitSelectionStrategy::TrueVarianceReduction() => panic!("TODO"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RandomForestParams {
    seed: u64,
    quiet: bool,
    split_method: SplitSelectionStrategy,
    instance_sampling_rate: f64,
    feature_sampling_rate: f64,
    min_leaf_support: usize,
    num_splits_per_feature: usize,
    max_depth: usize,
}

impl Default for RandomForestParams {
    fn default() -> Self {
        Self {
            seed: thread_rng().next_u64(),
            split_method: SplitSelectionStrategy::DifferenceInLabelMeans(),
            quiet: false,
            instance_sampling_rate: 0.5,
            feature_sampling_rate: 0.25,
            min_leaf_support: 10,
            num_splits_per_feature: 3,
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
                if fval < *split {
                    lhs.score(features)
                } else {
                    rhs.score(features)
                }
            }
        }
    }
}

struct RecursionParams {
    rand: Rc<Xoshiro256StarStar>,
    current_depth: usize,
    features: Vec<u32>,
    instances: Vec<u32>,
}

impl RecursionParams {
    fn subset(&self, instances: Vec<u32>) -> Self {
        Self {
            rand: self.rand.clone(),
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
        if self.instances.len() == 0 {
            return NotNan::new(0.0).unwrap();
        }
        let positive = self
            .instances
            .iter()
            .cloned()
            .filter(|index| dataset.instances[*index as usize].is_relevant())
            .count();
        return NotNan::new((positive as f64) / (self.instances.len() as f64))
            .expect("Leaf output NaN.");
    }
}

struct FeatureSplitCandidate {
    fid: u32,
    split: NotNan<f64>,
    lhs: Vec<u32>,
    rhs: Vec<u32>,
    importance: NotNan<f64>,
}

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
    let k = params.num_splits_per_feature;
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

    let splits: Vec<NotNan<f64>> = (1..k)
        .map(|i| (i as f64) / (k as f64))
        .map(|f| NotNan::new(f * range + stats.min).unwrap())
        .collect();

    let mut best = Vec::new();
    let mut splits_i = 0;
    for (pos, score) in scores.iter().enumerate() {
        if score > &splits[splits_i] {
            let (lhs, rhs) = ids.split_at(pos);
            if lhs.len() < params.min_leaf_support || rhs.len() < params.min_leaf_support {
                continue;
            }
            let importance = params.split_method.importance(lhs, rhs, dataset);
            best.push(SplitCandidate {
                pos,
                split: *score,
                importance,
            });
            splits_i += 1;
        }
        // continue
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

pub fn learn(params: &RandomForestParams, dataset: &RankingDataset) -> Option<TreeNode> {
    let step = RecursionParams {
        rand: Rc::new(Xoshiro256StarStar::seed_from_u64(params.seed)),
        features: dataset.features.clone(),
        instances: (0..dataset.instances.len()).map(|i| i as u32).collect(),
        current_depth: 1,
    };

    learn_recursive(params, dataset, &step)
}

fn learn_recursive(
    params: &RandomForestParams,
    dataset: &RankingDataset,
    step: &RecursionParams,
) -> Option<TreeNode> {
    // Gone too deep:
    if step.done() || step.current_depth >= params.max_depth {
        return None;
    }
    // Cannot split further:
    if step.instances.len() < params.min_leaf_support * 2 {
        return None;
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
        Some(TreeNode::FeatureSplit {
            fid: fsc.fid,
            split: fsc.split,
            lhs: Box::new(left_child),
            rhs: Box::new(right_child),
        })
    } else {
        None
    }
}
