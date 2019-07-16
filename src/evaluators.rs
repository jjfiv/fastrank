use std::collections::HashMap;
use ordered_float::NotNan;
use crate::dataset::*;

#[derive(PartialEq, PartialOrd, Eq, Ord)]
pub struct RankedInstance {
    pub score: NotNan<f64>,
    pub gain: NotNan<f32>,
    pub identifier: u32,
}

impl RankedInstance {
    pub fn new(score: NotNan<f64>, gain: NotNan<f32>, identifier: u32) -> Self {
        Self { score, gain, identifier }
    }
    pub fn is_relevant(&self) -> bool {
        self.gain.into_inner() > 0.0
    }
}

pub trait Evaluator {
    fn name(&self) -> String;
    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64;
}

pub struct ReciprocalRank;

impl Evaluator for ReciprocalRank {
    fn name(&self) -> String {
        String::from("RR")
    }
    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64 {
        // Compute RR:
        let mut recip_rank = 0.0;
        if let Some(rel_rank) = ranked_list
            .iter()
            .map(|ri| ri.is_relevant())
            .enumerate()
            .filter(|(_, rel)| *rel)
            .nth(0)
            .map(|(i, _)| i + 1) {
            recip_rank = 1.0 / (rel_rank as f64)
        }
        return recip_rank;
    }
}


pub struct AveragePrecision {
    /// Norms are the number of relevant by query for mAP.
    query_norms: HashMap<String, usize>,
}

impl AveragePrecision {
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
                query_norms.insert(qid.clone(), num_relevant);
            }
        }

        Self { query_norms }
    }
}

impl Evaluator for AveragePrecision {
    fn name(&self) -> String {
        String::from("AP")
    }
    fn score(&self, qid: &str, ranked_list: &[RankedInstance]) -> f64 {
        let num_relevant = self.query_norms.get(qid).cloned().unwrap_or_else(|| 
            ranked_list.iter().filter(|ri| ri.is_relevant()).count()
        ); 

        if num_relevant == 0 {
            return 0.0;
        }

        // Compute AP:
        let mut recall_points = 0;
        let mut sum_precision = 0.0;
        for rank in ranked_list
            .iter()
            .map(|ri| ri.is_relevant())
            .enumerate()
            .filter(|(_, rel)| *rel)
            .map(|(i, _)| i + 1)
        {
            recall_points += 1;
            sum_precision += f64::from(recall_points) / (rank as f64);
        }
        sum_precision / (num_relevant as f64)
    }
}