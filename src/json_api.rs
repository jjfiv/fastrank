use crate::{
    cart::{learn_cart_tree, CARTParams},
    io_helper, InstanceId,
};
use std::error::Error;

use crate::coordinate_ascent::CoordinateAscentParams;
use crate::dataset::{DatasetRef, RankingDataset};
use crate::evaluators::{RankedInstance, SetEvaluator};
use crate::model::ModelEnum;
use crate::qrel::QuerySetJudgments;
use crate::random_forest;
use crate::random_forest::RandomForestParams;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct TrainRequest {
    pub measure: String,
    pub params: FastRankModelParams,
    pub judgments: Option<QuerySetJudgments>,
}

impl Default for TrainRequest {
    fn default() -> Self {
        Self {
            measure: "ndcg".to_owned(),
            params: FastRankModelParams::CoordinateAscent(CoordinateAscentParams::default()),
            judgments: None,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum FastRankModelParams {
    CoordinateAscent(CoordinateAscentParams),
    RandomForest(RandomForestParams),
    RegressionTree(CARTParams),
}

pub fn do_training(
    train_request: TrainRequest,
    dataset: &DatasetRef,
) -> Result<ModelEnum, Box<dyn Error>> {
    let evaluator = SetEvaluator::create(
        dataset,
        train_request.measure.as_str(),
        train_request.judgments,
    )?;
    Ok(match train_request.params {
        FastRankModelParams::CoordinateAscent(params) => params.learn(dataset, &evaluator),
        FastRankModelParams::RandomForest(params) => {
            ModelEnum::Ensemble(random_forest::learn_ensemble(&params, dataset, &evaluator))
        }
        FastRankModelParams::RegressionTree(params) => {
            ModelEnum::DecisionTree(learn_cart_tree(&params, dataset))
        }
    })
}

pub fn predict_scores(
    model: &ModelEnum,
    dataset: &dyn RankingDataset,
) -> Result<HashMap<usize, f64>, Box<dyn Error>> {
    let mut scores = HashMap::default();

    for (_qid, docs) in dataset.instances_by_query().iter() {
        // Predict for every document:
        for index in docs.iter().cloned() {
            let score = dataset.score(index, model);
            // TODO make this an error?
            scores.insert(index.to_index(), score);
        }
    }

    Ok(scores)
}

/// Given a model and a dataset, save a trecrun file of predictions with a given system_name to output_path.
pub fn predict_to_trecrun(
    model: &ModelEnum,
    dataset: &dyn RankingDataset,
    output_path: &str,
    system_name: &str,
    depth: usize,
) -> Result<usize, Box<dyn Error>> {
    let mut output = io_helper::open_writer(output_path)?;
    let mut records_written = 0;
    for (qid, docs) in dataset.instances_by_query().iter() {
        // Predict for every document:
        let mut ranked_list: Vec<_> = docs
            .iter()
            .cloned()
            .map(|index| {
                let score = dataset.score(index, model);
                let gain = dataset.gain(index);
                RankedInstance::new(score, gain, index)
            })
            .collect();
        // Sort largest to smallest:
        ranked_list.sort_unstable();
        for (i, sdoc) in ranked_list.iter().enumerate() {
            let rank = i + 1;
            // Logically, depth=0 is optional!
            if depth > 0 && rank > depth {
                break;
            }
            let score = sdoc.score;
            let docid = match dataset.document_name(sdoc.identifier) {
                Some(x) => x,
                None => Err(
                    "Dataset does not contain document ids and therefore cannot save to trecrun!",
                )?,
            };
            writeln!(
                output,
                "{} Q0 {} {} {} {}",
                qid, docid, rank, score, system_name
            )?;
            records_written += 1;
        }
        output.flush()?;
    }
    Ok(records_written)
}

pub fn evaluate_query(
    measure: &str,
    gains: &[f32],
    scores: &[f64],
    depth: Option<usize>,
    opts: &serde_json::Value,
) -> Result<f64, Box<dyn Error>> {
    assert_eq!(gains.len(), scores.len());
    let mut ranked = Vec::with_capacity(gains.len());
    for (i, (gain, score)) in gains.iter().zip(scores.iter()).enumerate() {
        let id = InstanceId::from_index(i);
        ranked.push(RankedInstance {
            score: *score,
            gain: *gain,
            identifier: id,
        });
    }
    // Sort ranked-list:
    ranked.sort_unstable();

    use crate::evaluators::*;
    Ok(match measure.to_lowercase().as_ref() {
        "ap" | "map" => {
            let num_rel = opts
                .get("num_rel")
                .map(|v| v.as_u64())
                .flatten()
                .map(|v| v as u32)
                .unwrap_or_else(|| compute_num_relevant(&ranked));
            compute_ap(&ranked, num_rel as u32)
        }
        "dcg" => {
            let gains: Vec<f32> = ranked.iter().map(|r| r.gain.clone()).collect();
            let compute_ideal = opts
                .get("ideal")
                .map(|v| v.as_bool())
                .flatten()
                .unwrap_or(false);
            compute_dcg(&gains, depth, compute_ideal)
        }
        "mrr" | "rr" | "recip_rank" => compute_recip_rank(&ranked),
        "ndcg" => {
            let gains: Vec<f32> = ranked.iter().map(|r| r.gain.clone()).collect();
            let ideal = compute_dcg(&gains, depth, true);
            let dcg = compute_dcg(&gains, depth, false);
            dcg / ideal
        }
        other => Err(format!("No Such Evaluator: {}", other))?,
    })
}
