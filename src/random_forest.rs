use crate::cart::*;
use crate::evaluators::SetEvaluator;
use crate::model::{ModelEnum, TreeNode, WeightedEnsemble};
use crate::sampling::DatasetSampling;
use crate::{core::Scored, dataset::DatasetRef};
use oorandom::Rand64;
use rayon::prelude::*;

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
    pub split_candidates: Option<u32>,
    pub max_depth: u32,
    pub feature_tolerance: f64,
    pub only_use_features_once: bool,
}

impl Default for RandomForestParams {
    fn default() -> Self {
        let mut rand = Rand64::new(0xdeadbeef);
        Self {
            weight_trees: false,
            seed: rand.rand_u64(),
            split_method: SplitSelectionStrategy::SquaredError,
            quiet: false,
            num_trees: 100,
            instance_sampling_rate: 0.5,
            feature_sampling_rate: 0.25,
            min_leaf_support: 10,
            split_candidates: Some(3),
            max_depth: 8,
            feature_tolerance: 1E-6,
            only_use_features_once: true,
        }
    }
}

impl RandomForestParams {
    fn to_cart_params(&self) -> CARTParams {
        CARTParams {
            seed: self.seed,
            split_method: self.split_method,
            split_candidates: self.split_candidates,
            min_leaf_support: self.min_leaf_support,
            max_depth: self.max_depth,
            feature_tolerance: self.feature_tolerance,
            only_use_features_once: self.only_use_features_once,
        }
    }
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
        println!("-------------------------");
        println!("|{:>7}|{:>7}|{:>7}|", "Tree", "Depth", evaluator.name());
        println!("-------------------------");
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
        let mut cart_params = params.to_cart_params();
        cart_params.seed = local_rand.rand_u64();
        let tree: TreeNode = learn_cart_tree(&cart_params, &subsample);
        let eval = evaluator.evaluate_mean(&tree);
        if !params.quiet {
            println!("|{:>7}|{:>7}|{:>7.3}|", idx + 1, tree.depth(), eval);
        }
        Scored::new(eval, tree)
    }));

    if !params.quiet {
        println!("-------------------------");
    }

    WeightedEnsemble::new(
        trees
            .into_iter()
            .map(|tree| {
                let m = ModelEnum::DecisionTree(tree.item);
                Scored::new(if params.weight_trees { tree.score } else { 1.0 }, m)
            })
            .collect(),
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dataset;

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
            split_candidates: Some(32),
            split_method: SplitSelectionStrategy::SquaredError,
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
        assert_float_eq("means[0] = predefined", means[0], 0.4582452225554235);
    }
}
