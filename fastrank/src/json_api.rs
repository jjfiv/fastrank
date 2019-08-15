use std::error::Error;

use crate::coordinate_ascent::CoordinateAscentParams;
use crate::dataset::RankingDataset;
use crate::evaluators::SetEvaluator;
use crate::model::ModelEnum;
use crate::qrel::QuerySetJudgments;

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
}

pub fn do_training(
    train_request: TrainRequest,
    dataset: &dyn RankingDataset,
) -> Result<ModelEnum, Box<Error>> {
    let evaluator = SetEvaluator::create(
        dataset,
        train_request.measure.as_str(),
        train_request.judgments,
    )?;
    Ok(match train_request.params {
        FastRankModelParams::CoordinateAscent(params) => params.learn(dataset, &evaluator),
    })
}
