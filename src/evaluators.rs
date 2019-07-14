use std::collections::HashMap;
use crate::dataset::*;
use crate::Model;
use crate::Scored;

pub trait Evaluator {
    fn score(&self, model: &Model, dataset: &RankingDataset) -> f64;
}

pub struct MeanAveragePrecision {
    /// Norms are the number of relevant by query for mAP.
    query_norms: HashMap<String, f64>,
    num_queries: f64,
}

impl MeanAveragePrecision {
    pub fn new(dataset: &RankingDataset, total_relevant_by_qid: Option<&HashMap<String, u32>>) -> Self {
        let num_queries = dataset.data_by_query.len() as f64;
        let mut query_norms = HashMap::new();

        for (qid, instance_ids) in dataset.data_by_query.iter() {
            // Determine the total number of relevant documents:
            let param_num_relevant: Option<usize> = total_relevant_by_qid
                .and_then(|data| data.get(qid))
                .map(|num| *num as usize);
            // Calculate if unavailable in config:
            let num_relevant: usize = param_num_relevant.unwrap_or_else(|| {
                instance_ids
                    .iter()
                    .filter(|index| dataset.instances[**index].is_relevant())
                    .count()
            });

            if num_relevant > 0 {
                query_norms.insert(qid.clone(), num_relevant as f64);
            }
        }

        Self { query_norms, num_queries }
    }
}

impl Evaluator for MeanAveragePrecision {
    fn score(&self, model: &Model, dataset: &RankingDataset) -> f64 {
        let mut ap_sum = 0.0;

        for (qid, instance_ids) in dataset.data_by_query.iter() {
            let num_relevant = match self.query_norms.get(qid) {
                None => continue,
                Some(norm) => norm
            };

            // Rank data.
            let mut ranked_list: Vec<Scored<usize>> = instance_ids
                .iter()
                .cloned()
                .map(|index| Scored::new(model.score(&dataset.instances[index].features), index))
                .collect();
            ranked_list.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
            
            // Compute AP:
            let mut recall_points = 0;
            let mut sum_precision = 0.0;
            for rank in ranked_list
                .iter()
                .map(|scored| dataset.instances[scored.item].is_relevant())
                .enumerate()
                .filter(|(_, rel)| *rel)
                .map(|(i, _)| i + 1)
            {
                recall_points += 1;
                sum_precision += f64::from(recall_points) / (rank as f64);
            }
            ap_sum += sum_precision / num_relevant;
        }

        // Compute Mean AP:
        ap_sum / self.num_queries
    }
}