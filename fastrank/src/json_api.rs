use std::error::Error;

use crate::qrel::QuerySetJudgments;
use crate::evaluators::SetEvaluator;
use crate::dataset::DatasetRef;
use crate::model::ModelEnum;
use crate::coordinate_ascent::CoordinateAscentParams;

#[derive(Serialize, Deserialize)]
pub struct TrainRequest {
    pub measure: String,
    pub params: FastRankModelParams,
    pub judgments: Option<QuerySetJudgments>,
}

#[derive(Serialize, Deserialize)]
pub enum FastRankModelParams {
    CoordinateAscent(CoordinateAscentParams),
}

#[derive(Serialize, Deserialize)]
pub enum TrainResponse {
    Error(String),
    LearnedModel(ModelEnum)
}

pub fn do_training(train_request: TrainRequest, dataset: DatasetRef) -> Result<TrainResponse, Box<Error>> {
    let evaluator = SetEvaluator::create(&dataset, train_request.measure.as_str(), train_request.judgments)?;
    Ok(match train_request.params {
        FastRankModelParams::CoordinateAscent(params) => {
            let model = params.learn(&dataset, &evaluator);
            TrainResponse::LearnedModel(model)
        }
    })
}