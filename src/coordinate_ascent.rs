use crate::libsvm;
use ordered_float::NotNan;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

pub enum Features {
    Dense32(Vec<f32>),
    Dense64(Vec<f64>),
    /// Sparse 32-bit representation; must be sorted!
    Sparse32(Vec<(u32, f32)>),
    /// Sparse 64-bit representation; must be sorted!
    Sparse64(Vec<(u32, f64)>),
}

impl Features {
    pub fn get(&self, idx: u32) -> Option<f64> {
        match self {
            Features::Dense32(arr) => Some(f64::from(arr[idx as usize])),
            Features::Dense64(arr) => Some(arr[idx as usize]),
            Features::Sparse32(features) => {
                for (fidx, val) in features.iter() {
                    if *fidx == idx {
                        return Some(f64::from(*val));
                    } else if *fidx > idx {
                        break;
                    }
                }
                None
            }
            Features::Sparse64(features) => {
                for (fidx, val) in features.iter() {
                    if *fidx == idx {
                        return Some(*val);
                    } else if *fidx > idx {
                        break;
                    }
                }
                None
            }
        }
    }
    pub fn ids(&self) -> Vec<u32> {
        let mut features: Vec<u32> = Vec::new();
        match self {
            Features::Dense32(arr) => features.extend(0..(arr.len() as u32)),
            Features::Dense64(arr) => features.extend(0..(arr.len() as u32)),
            Features::Sparse32(arr) => features.extend(arr.iter().map(|(idx, _)| *idx)),
            Features::Sparse64(arr) => features.extend(arr.iter().map(|(idx, _)| *idx)),
        }
        features
    }
    fn eval_linear_model(&self, weights: &[f64]) -> f64 {
        let mut output = 0.0;
        match self {
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
            _ => {}
        };
        output
    }
}

pub struct TrainingInstance {
    pub gain: f32,
    pub qid: String,
    pub features: Features,
}

impl TrainingInstance {
    pub fn try_new(libsvm: libsvm::Instance) -> Result<TrainingInstance, &'static str> {
        Ok(TrainingInstance {
            gain: libsvm.label,
            qid: libsvm.query.ok_or("Missing qid")?,
            features: Features::Sparse32(
                libsvm
                    .features
                    .into_iter()
                    .map(|f| (f.idx, f.value))
                    .collect(),
            ),
        })
    }
    fn is_relevant(&self) -> bool {
        self.gain > 0.0
    }
}

#[derive(Clone, Debug)]
pub struct CoordinateAscentParams {
    pub num_restarts: i32,
    pub num_max_iterations: i32,
    pub step_base: f64,
    pub step_scale: f64,
    pub tolerance: f64,
    pub quiet: bool,
    pub total_relevant_by_qid: Option<HashMap<String, u32>>,
}

impl Default for CoordinateAscentParams {
    fn default() -> Self {
        Self {
            num_restarts: 5,
            num_max_iterations: 25,
            step_base: 0.05,
            step_scale: 2.0,
            tolerance: 0.001,
            quiet: false,
            total_relevant_by_qid: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoordinateAscentModel {
    pub params: CoordinateAscentParams,
    pub weights: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Scored<T: Clone> {
    pub score: NotNan<f64>,
    pub item: T,
}
impl<T: Clone> Scored<T> {
    fn new(score: f64, item: T) -> Self {
        Self {
            score: NotNan::new(score).expect("NaN found!"),
            item,
        }
    }
    fn replace_if_better(&mut self, score: f64, item: T) -> bool {
        if let Ok(score) = NotNan::new(score) {
            if score > self.score {
                self.item = item;
                self.score = score;
                return true;
            }
        }
        false
    }
    fn use_best(&mut self, other: &Scored<T>) {
        if self.score > other.score {
            return;
        } else {
            self.score = other.score;
            self.item = other.item.clone();
        }
    }
}

fn normalize(weights: &mut [f64]) {
    let mut sum = 0.0;
    for w in weights.iter() {
        sum += f64::abs(*w);
    }
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
}

impl CoordinateAscentModel {
    pub fn new() -> Self {
        Self::new_with_params(CoordinateAscentParams::default())
    }
    pub fn new_with_params(params: CoordinateAscentParams) -> Self {
        CoordinateAscentModel {
            params,
            weights: Vec::new(),
        }
    }
    fn evaluate_map(
        params: &CoordinateAscentParams,
        weights: &[f64],
        data_by_query: &HashMap<String, Vec<usize>>,
        data: &[TrainingInstance],
    ) -> f64 {
        let mut num_queries = data_by_query.len() as f64;
        let mut ap_sum = 0.0;
        for (qid, instance_ids) in data_by_query.iter() {
            // Rank data.
            let mut ranked_list: Vec<Scored<usize>> = instance_ids
                .iter()
                .cloned()
                .map(|index| Scored::new(data[index].features.eval_linear_model(weights), index))
                .collect();
            ranked_list.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));

            // Determine the total number of relevant documents:
            let param_num_relevant: Option<usize> = params
                .total_relevant_by_qid
                .as_ref()
                .and_then(|data| data.get(qid))
                .map(|num| *num as usize);
            // Calculate if unavailable in config:
            let num_relevant: usize = param_num_relevant.unwrap_or_else(|| {
                ranked_list
                    .iter()
                    .filter(|scored| data[scored.item].is_relevant())
                    .count()
            });

            // In theory, we should skip these queries!
            if num_relevant == 0 {
                continue;
            }

            // Compute AP:
            let mut recall_points = 0;
            let mut sum_precision = 0.0;
            for rank in ranked_list
                .iter()
                .map(|scored| data[scored.item].is_relevant())
                .enumerate()
                .filter(|(_, rel)| *rel)
                .map(|(i, _)| i+1)
            {
                recall_points += 1;
                sum_precision += f64::from(recall_points) / (rank as f64);
            }
            ap_sum += sum_precision / (num_relevant as f64);
        }

        // Compute Mean AP:
        ap_sum / num_queries
    }

    pub fn learn(&mut self, data: Vec<TrainingInstance>) {
        let params = &self.params;
        let mut weights: &mut Vec<f64> = &mut self.weights;
        let mut best_model: Scored<Vec<f64>> = Scored::new(0.0, Vec::new());

        let tolerance = NotNan::new(params.tolerance).expect("Tolerance param should not be NaN.");

        let sign = &[1, -1, 0];

        // Collect features that are actually present.
        let mut features: HashSet<u32> = HashSet::new();
        // Collect training instances by the query.
        let mut data_by_query: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, inst) in data.iter().enumerate() {
            data_by_query
                .entry(inst.qid.clone())
                .or_insert(Vec::new())
                .push(i);
            features.extend(inst.features.ids());
        }

        let n_dim = features
            .iter()
            .cloned()
            .max()
            .expect("No features defined!")
            + 1;
        let mut rand = thread_rng();

        if !params.quiet {
            println!("---------------------------");
            println!("Training starts...");
            println!("---------------------------");
        }
        for restart in 0..params.num_restarts {
            if !params.quiet {
                println!(
                    "[+] Random restart #{}/{}...",
                    restart + 1,
                    params.num_restarts
                );
            }
            let mut consecutive_failures = 0;

            // Initialize to even weights:
            weights.clear();
            weights.resize(n_dim as usize, 1.0 / f64::from(n_dim));

            // Initialize this local best (within current restart cycle):
            let start_score = Self::evaluate_map(&params, &weights, &data_by_query, &data);
            let mut current_best = Scored::new(start_score, weights.clone());

            loop {
                //There must be at least one feature increasing whose weight helps
                if weights.len() == 1 {
                    if consecutive_failures > 0 {
                        break;
                    }
                } else {
                    // Go until there is no more to try.
                    if consecutive_failures >= weights.len() - 1 {
                        break;
                    }
                }

                if !params.quiet {
                    println!("Shuffle features and optimize!");
                    println!("---------------------------");
                    println!("{:>9}|{:>9}|{:>9}", "Feature", "Weight", "mAP");
                    println!("---------------------------");
                }

                let mut fids: Vec<u32> = features.iter().cloned().collect();
                fids.shuffle(&mut rand);

                for current_feature in fids {
                    let current_feature = current_feature as usize;
                    let orig_weight = weights[current_feature];
                    let mut total_step = 0.0;
                    let mut best_total_step = 0.0;
                    let mut success = false;

                    for dir in sign {
                        let mut step = params.step_base * f64::from(*dir);
                        if orig_weight != 0.0 && f64::abs(step) > 0.5 * f64::abs(orig_weight) {
                            step = params.step_base * f64::abs(orig_weight)
                        }
                        total_step = step;
                        let mut num_iter = params.num_max_iterations;
                        if *dir == 0 {
                            num_iter = 1;
                            total_step = -orig_weight;
                        }

                        for feature_trial in 0..num_iter {
                            let w = orig_weight + total_step;
                            weights[current_feature] = w;
                            let score =
                                Self::evaluate_map(&params, &weights, &data_by_query, &data);

                            if current_best.replace_if_better(score, weights.clone()) {
                                success = true;
                                best_total_step = total_step;
                                if !params.quiet {
                                    println!("{:>9}|{:>9.3}|{:>9.3}", current_feature, w, score);
                                }
                            }
                            if feature_trial < num_iter - 1 {
                                step *= params.step_scale;
                                total_step += step;
                            }
                        }

                        // If found better, don't reset weights.
                        // Also: skip other direction search.
                        if success {
                            break;
                        } else {
                            weights[current_feature] = orig_weight;
                        }
                    } // dir

                    if success {
                        // Since we've found a better weight value.
                        consecutive_failures = 0;
                        weights[current_feature] = orig_weight + best_total_step;
                        normalize(&mut weights);
                        current_best.item = weights.clone();
                    } else {
                        consecutive_failures += 1;
                        weights[current_feature] = orig_weight;
                    }
                } // current_feature
                if !params.quiet {
                    println!("---------------------------");
                }

                if current_best.score - start_score < tolerance {
                    break;
                }

                best_model.use_best(&current_best);
            } // optimize-loop

            *weights = best_model.item.clone();

            if !params.quiet {
                println!("---------------------------");
                println!("Finished successfully.");
                println!(
                    "mAP: {}",
                    Self::evaluate_map(&params, &weights, &data_by_query, &data)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
