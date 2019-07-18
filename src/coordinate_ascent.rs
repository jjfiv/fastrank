use crate::dataset::*;
use crate::evaluators::Evaluator;
use crate::WeightedEnsemble;
use crate::{Model, Scored};
use ordered_float::NotNan;
use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug)]
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
        Self {
            num_restarts: 5,
            num_max_iterations: 25,
            step_base: 0.05,
            step_scale: 2.0,
            tolerance: 0.001,
            seed: thread_rng().next_u64(),
            normalize: false,
            quiet: false,
            init_random: false,
            output_ensemble: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DenseLinearRankingModel {
    weights: Vec<f64>,
}
impl DenseLinearRankingModel {
    fn new(n_dim: u32) -> Self {
        Self {
            weights: vec![0.0; n_dim as usize],
        }
    }

    fn reset<R: Rng>(&mut self, init_random: bool, rand: &mut R) {
        if init_random {
            for i in 0..self.weights.len() {
                self.weights[i] = rand.gen_range(-1.0, 1.0);
            }
        } else {
            self.reset_uniform();
        }
    }

    fn reset_uniform(&mut self) {
        let n_dim = self.weights.len();
        // Initialize to even weights:
        self.weights.clear();
        assert_eq!(0, self.weights.len());
        self.weights.resize(n_dim, 1.0 / (n_dim as f64));
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

    fn predict(&self, features: &Features) -> f64 {
        let mut output = 0.0;
        let weights = &self.weights;
        match features {
            Features::Dense32(arr) => {
                for (feature, weight) in arr.iter().cloned().zip(weights.iter().cloned()) {
                    output += f64::from(feature) * weight;
                }
            }
            Features::Dense64(arr) => {
                for (feature, weight) in arr.iter().cloned().zip(weights.iter().cloned()) {
                    output += feature * weight;
                }
            }
            Features::Sparse32(arr) => {
                for (idx, feature) in arr.iter().cloned() {
                    output += f64::from(feature) * weights[idx as usize];
                }
            }
            Features::Sparse64(arr) => {
                for (idx, feature) in arr.iter().cloned() {
                    output += feature * weights[idx as usize];
                }
            }
        };
        output
    }
}

impl Model for DenseLinearRankingModel {
    fn score(&self, features: &Features) -> NotNan<f64> {
        NotNan::new(self.predict(features)).expect("Model.predict -> NaN")
    }
}

const SIGN: &[i32] = &[0, -1, 1];

fn optimize_inner<R: Rng>(
    restart_id: u32,
    data: &RankingDataset,
    evaluator: &Evaluator,
    mut rand: R,
    params: &CoordinateAscentParams,
) -> Scored<DenseLinearRankingModel> {
    let quiet = params.quiet;
    let tolerance = NotNan::new(params.tolerance).unwrap();

    // Initialize to even weights:
    let mut model = DenseLinearRankingModel::new(data.n_dim);
    model.reset(params.init_random, &mut rand);

    // Initialize this local best (within current restart cycle):
    let start_score = data.evaluate_mean(&model, evaluator);
    let mut current_best = Scored::new(start_score, model.clone());

    loop {
        // Get new order of features for this optimization pass.
        let mut fids: Vec<u32> = data.features.clone();
        fids.shuffle(&mut rand);

        if !quiet {
            println!("Shuffle features and optimize!");
            println!("----------------------------------------");
            println!("{:4}|{:<16}|{:>9}|{:>9}", restart_id, "Feature", "Weight", evaluator.name());
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

                let current_feature_name = data
                    .feature_names
                    .get(current_feature)
                    .cloned()
                    .unwrap_or(format!("{}", current_feature));
                let current_feature = *current_feature as usize;

                let orig_weight = model.weights[current_feature];
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
                        model.weights[current_feature] = w;
                        let score = data.evaluate_mean(&model, evaluator);

                        if current_best.replace_if_better(score, model.clone()) {
                            if !quiet {
                                println!("{:4}|{:<16}|{:>9.3}|{:>9.3}", restart_id, current_feature_name, w, score);
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

                (current_best.score - start_score)
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
    pub fn learn(&self, data: &RankingDataset, evaluator: &Evaluator) -> Box<Model> {
        let mut rand = Xoshiro256StarStar::seed_from_u64(self.seed);
        let tolerance = NotNan::new(self.tolerance).expect("Tolerance param should not be NaN.");
        let mut history: Vec<Scored<DenseLinearRankingModel>> = Vec::new();

        if !self.quiet {
            println!("---------------------------");
            println!("Training starts...");
            println!("---------------------------");
        }

        let states: Vec<_> = (0..self.num_restarts)
            .map(|restart_id| {
                (
                    restart_id,
                    Xoshiro256StarStar::seed_from_u64(rand.next_u64()),
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
            let members: Vec<Scored<Arc<dyn Model>>> = history
                .iter()
                .map(|sm| {
                    let mut model = sm.item.clone();
                    model.l1_normalize();
                    let m: Arc<dyn Model> = Arc::new(model);
                    Scored::new(sm.score.into_inner(), m)
                })
                .collect();
            Box::new(WeightedEnsemble(members))
        } else {
            Box::new(
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
