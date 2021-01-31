use crate::dataset::RankingDataset;
use crate::evaluators::SetEvaluator;
use crate::model::{DenseLinearRankingModel, ModelEnum, WeightedEnsemble};
use crate::FeatureId;
use crate::Scored;
use ordered_float::NotNan;
use oorandom::Rand64;
use crate::randutil::shuffle;
use rayon::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoordinateAscentParams {
    pub num_restarts: u32,
    pub num_max_iterations: u32,
    pub step_base: f64,
    pub step_scale: f64,
    pub tolerance: f64,
    pub seed: u64,
    pub normalize: bool,
    pub quiet: bool,
    pub init_random: bool,
    pub output_ensemble: bool,
}

impl Default for CoordinateAscentParams {
    fn default() -> Self {
        let mut rand = Rand64::new(0xdeadbeef);
        Self {
            num_restarts: 5,
            num_max_iterations: 25,
            step_base: 0.05,
            step_scale: 2.0,
            tolerance: 0.001,
            seed: rand.rand_u64(),
            normalize: true,
            quiet: false,
            init_random: true,
            output_ensemble: false,
        }
    }
}

impl DenseLinearRankingModel {
    fn new(n_dim: u32) -> Self {
        Self {
            weights: vec![0.0; n_dim as usize],
        }
    }

    fn reset(&mut self, init_random: bool, rand: &mut Rand64, valid_features: &[FeatureId]) {
        if init_random {
            for i in valid_features.iter() {
                self.weights[i.to_index()] = (rand.rand_float() * 2.0) - 1.0;
            }
        } else {
            self.reset_uniform(valid_features);
        }
    }

    fn reset_uniform(&mut self, valid_features: &[FeatureId]) {
        let n_dim = self.weights.len();
        // Initialize to even weights:
        self.weights.clear();
        assert_eq!(0, self.weights.len());
        self.weights.resize(n_dim, 0.0);
        for i in valid_features.iter() {
            self.weights[i.to_index()] = 1.0 / (valid_features.len() as f64);
        }
        assert_eq!(n_dim, self.weights.len());
    }

    fn l1_normalize(&mut self) {
        let mut sum = 0.0;
        for w in self.weights.iter() {
            sum += f64::abs(*w);
        }
        if sum > 0.0 {
            for w in self.weights.iter_mut() {
                *w /= sum;
            }
        }
    }
}

const SIGN: &[i32] = &[0, -1, 1];

fn optimize_inner(
    restart_id: u32,
    data: &dyn RankingDataset,
    evaluator: &SetEvaluator,
    mut rand: Rand64,
    params: &CoordinateAscentParams,
) -> Scored<DenseLinearRankingModel> {
    let quiet = params.quiet;
    let tolerance = NotNan::new(params.tolerance).unwrap();

    let fids: Vec<FeatureId> = data.features().clone();
    let model_dim = (fids
        .iter()
        .max()
        .expect("Should be at least one feature!")
        .to_index() as u32)
        + 1;

    // Initialize to even weights:
    let mut model = DenseLinearRankingModel::new(model_dim);
    model.reset(params.init_random, &mut rand, &data.features());

    // Initialize this local best (within current restart cycle):
    let start_score = evaluator.evaluate_mean(&model);
    let mut current_best = Scored::new(start_score, model.clone());

    loop {
        let mut fids = fids.clone();
        // Get new order of features for this optimization pass.
        shuffle(&mut fids, &mut rand);

        if !quiet {
            println!("Shuffle features and optimize!");
            println!("----------------------------------------");
            println!(
                "{:4}|{:<16}|{:>9}|{:>9}",
                restart_id,
                "Feature",
                "Weight",
                evaluator.name()
            );
            println!("----------------------------------------");
        }

        let successes = fids
            .iter()
            .map(|current_feature| {
                let start_score = current_best.score;
                model = current_best.item.clone();
                if params.normalize {
                    model.l1_normalize();
                }

                let current_feature_name = data.feature_name(*current_feature);

                let orig_weight = model.weights[current_feature.to_index()];
                let mut total_step;

                for dir in SIGN {
                    let mut step = params.step_base * f64::from(*dir);
                    if orig_weight != 0.0 && step.abs() > 0.5 * f64::abs(orig_weight) {
                        step = params.step_base * orig_weight.abs() * f64::from(*dir)
                    }
                    total_step = step;
                    let mut num_iter = params.num_max_iterations;
                    if *dir == 0 {
                        num_iter = 1;
                        total_step = -orig_weight;
                    }

                    for _ in 0..num_iter {
                        let w = orig_weight + total_step;
                        model.weights[current_feature.to_index()] = w;
                        let score = evaluator.evaluate_mean(&model);

                        if current_best.replace_if_better(score, model.clone()) {
                            if !quiet {
                                println!(
                                    "{:4}|{:<16}|{:>9.3}|{:>9.3}",
                                    restart_id, current_feature_name, w, score
                                );
                            }
                        }

                        step *= params.step_scale;
                        total_step += step;
                    }

                    // If found measurably better, skip other directions:
                    if (current_best.score - start_score) > tolerance {
                        break;
                    }
                } // dir

                current_best.score - start_score
            })
            .filter(|improvement| improvement > &tolerance)
            .count(); // current_feature

        // If no feature mutation leads to measurable improvement, we're done.
        if successes == 0 {
            break;
        }

        if !quiet {
            println!("---------------------------");
        }
    } // optimize-loop

    current_best
}

impl CoordinateAscentParams {
    pub fn learn(&self, data: &dyn RankingDataset, evaluator: &SetEvaluator) -> ModelEnum {
        let mut rand = Rand64::new(self.seed.into());

        assert!(data.n_dim() > 0);
        assert!(data.instances().len() > 0);
        assert!(data.queries().len() > 0);

        if !self.quiet {
            println!("---------------------------");
            println!("Training starts...");
            println!("---------------------------");
        }

        let states: Vec<_> = (0..self.num_restarts)
            .map(|restart_id| {
                (
                    restart_id,
                    Rand64::new(rand.rand_u64().into()),
                )
            })
            .collect();

        let mut history: Vec<Scored<DenseLinearRankingModel>> = Vec::new();
        history.par_extend(states.into_par_iter().map(|(restart_id, rand)| {
            if !self.quiet {
                println!(
                    "[+] Random restart #{}/{}...",
                    restart_id + 1,
                    self.num_restarts
                );
            }
            optimize_inner(restart_id, data, evaluator, rand, self)
        }));

        if !self.quiet {
            println!("---------------------------");
            println!("Finished successfully.");
        }

        if self.output_ensemble && history.len() > 1 {
            let members: Vec<Scored<ModelEnum>> = history
                .iter()
                .map(|sm| {
                    let mut model = sm.item.clone();
                    model.l1_normalize();
                    let m = ModelEnum::Linear(model);
                    Scored::new(sm.score.into_inner(), m)
                })
                .collect();
            ModelEnum::Ensemble(WeightedEnsemble::new(members))
        } else {
            ModelEnum::Linear(
                history
                    .iter()
                    .max()
                    .expect("Should be at least 1 restart!")
                    .item
                    .clone(),
            )
        }
    } // learn
} // impl
