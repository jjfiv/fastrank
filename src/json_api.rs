use crate::io_helper;
use std::error::Error;

use crate::coordinate_ascent::CoordinateAscentParams;
use crate::dataset::{DatasetRef, RankingDataset};
use crate::evaluators::{RankedInstance, SetEvaluator};
use crate::model::ModelEnum;
use crate::qrel::QuerySetJudgments;
use crate::random_forest;
use crate::random_forest::RandomForestParams;

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
    })
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
