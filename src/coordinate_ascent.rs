use ordered_float::NotNan;
use crate::{Model, Scored};
use rand::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::HashMap;
use crate::dataset::*;
use crate::evaluators::Evaluator;

#[derive(Clone, Debug)]
pub struct CoordinateAscentParams {
    pub num_restarts: u32,
    pub num_max_iterations: u32,
    pub step_base: f64,
    pub step_scale: f64,
    pub tolerance: f64,
    pub seed: u64,
    pub normalize: bool,
    pub total_relevant_by_qid: Option<HashMap<String, u32>>,
    pub quiet: bool,
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
            total_relevant_by_qid: None,
        }
    }
}

#[derive(Debug,Clone)]
pub struct DenseLinearRankingModel {
    weights: Vec<f64>,
}
impl DenseLinearRankingModel {
    fn new(n_dim: u32) -> Self {
        Self { weights: vec![0.0; n_dim as usize] }
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
    fn score(&self, features: &Features) -> f64 {
        self.predict(features)
    }
}


impl CoordinateAscentParams {
    pub fn learn(&self, data: &RankingDataset, evaluator: &Evaluator) -> f64 {
        let mut rand = Xoshiro256StarStar::seed_from_u64(self.seed);
        let tolerance = NotNan::new(self.tolerance).expect("Tolerance param should not be NaN.");

        let sign = &[1, -1, 0];
        
        let mut model = DenseLinearRankingModel::new(data.n_dim);
        let mut best_model = Scored::new(0.0, model.clone());

        // The stochasticity in this algorithm comes from the order in which features are visited.
        let optimization_orders: Vec<Vec<u32>> = (0..self.num_restarts)
            .map(|_| {
                let mut fids: Vec<u32> = data.features.clone();
                fids.shuffle(&mut rand);
                fids
            })
            .collect();

        if !self.quiet {
            println!("---------------------------");
            println!("Training starts...");
            println!("---------------------------");
        }

        for (restart, fids) in optimization_orders.iter().enumerate() {
            if !self.quiet {
                println!(
                    "[+] Random restart #{}/{}...",
                    restart + 1,
                    self.num_restarts
                );
            }
            let mut consecutive_failures = 0;

            // Initialize to even weights:
            model.reset_uniform();

            // Initialize this local best (within current restart cycle):
            let start_score = evaluator.score(&model, &data);
            let mut current_best = Scored::new(start_score, model.clone());

            loop {
                //There must be at least one feature increasing whose weight helps
                if fids.len() == 1 {
                    if consecutive_failures > 0 {
                        break;
                    }
                } else {
                    // Go until there is no more to try.
                    if consecutive_failures >= fids.len() - 1 {
                        break;
                    }
                }

                if !self.quiet {
                    println!("Shuffle features and optimize!");
                    println!("---------------------------");
                    println!("{:>9}|{:>9}|{:>9}", "Feature", "Weight", "mAP");
                    println!("---------------------------");
                }

                for current_feature in fids {
                    let current_feature = *current_feature as usize;
                    let orig_weight = model.weights[current_feature];
                    let mut total_step;
                    let mut best_weight = orig_weight;
                    let mut success = false;

                    for dir in sign {
                        let mut step = self.step_base * f64::from(*dir);
                        if orig_weight != 0.0 && f64::abs(step) > 0.5 * f64::abs(orig_weight) {
                            step = self.step_base * f64::abs(orig_weight)
                        }
                        total_step = step;
                        let mut num_iter = self.num_max_iterations;
                        if *dir == 0 {
                            num_iter = 1;
                            total_step = -orig_weight;
                        }

                        for feature_trial in 0..num_iter {
                            let w = orig_weight + total_step;
                            model.weights[current_feature] = w;
                            let score = evaluator.score(&model, &data);

                            if current_best.replace_if_better(score, model.clone()) {
                                success = true;
                                best_weight = w;
                                if !self.quiet {
                                    println!("{:>9}|{:>9.3}|{:>9.3}", current_feature, w, score);
                                }
                            }
                            if feature_trial < num_iter - 1 {
                                step *= self.step_scale;
                                total_step += step;
                            }
                        }

                        // If found better, don't reset weights.
                        // Also: skip other direction search.
                        if success {
                            break;
                        } else {
                            model.weights[current_feature] = orig_weight;
                        }
                    } // dir

                    // Since we've found a better weight value.
                    model.weights[current_feature] = best_weight;
                    if self.normalize {
                        model.l1_normalize();
                    }

                    if success {
                        consecutive_failures = 0;
                    } else {
                        consecutive_failures += 1;
                    }
                } // current_feature

                if !self.quiet {
                    println!("---------------------------");
                }

                if current_best.score - start_score < tolerance {
                    break;
                }

                best_model.use_best(&current_best);
            } // optimize-loop
        }
        
        let model = best_model.item.clone();

        if !self.quiet {
            println!("---------------------------");
            println!("Finished successfully.");
        }

        evaluator.score(&model, &data)
    } // learn
} // impl
