use libc::{c_char, c_void};
use once_cell::sync::Lazy;
use serde_json;
use std::{collections::HashMap, ffi::CString, ptr, sync::Arc};
use std::{error::Error, sync::atomic::AtomicIsize};
use std::{ffi::CStr, sync::Mutex};

use crate::coordinate_ascent::CoordinateAscentParams;
use crate::dataset;
use crate::dataset::DatasetRef;
use crate::dataset::RankingDataset;
use crate::evaluators::SetEvaluator;
use crate::json_api;
use crate::json_api::{FastRankModelParams, TrainRequest};
use crate::model::ModelEnum;
use crate::qrel::QuerySetJudgments;
use crate::random_forest::RandomForestParams;
use crate::sampling::DatasetSampling;
use crate::FeatureId;

use crate::{CDataset, CModel, CQRel};

pub(crate) static ERROR_ID: AtomicIsize = AtomicIsize::new(1);
pub(crate) static ERRORS: Lazy<Arc<Mutex<HashMap<isize, String>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::default())));

pub(crate) fn next_error_id() -> isize {
    ERROR_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

pub(crate) fn store_err<T, E>(r: Result<T, E>, error: *mut isize) -> Result<T, ()>
where
    E: std::fmt::Display + Sized,
{
    match r {
        Ok(x) => Ok(x),
        Err(e) => {
            let err = next_error_id();
            ERRORS.as_ref().lock().unwrap().insert(err, e.to_string());
            unsafe {
                *error = err;
            }
            Err(())
        }
    }
}

pub(crate) fn box_to_ptr<T>(item: T) -> *const T {
    Box::into_raw(Box::new(item)) as *const T
}

pub(crate) fn store_err_to_ptr<T, E>(r: Result<T, E>, error: *mut isize) -> *const T
where
    E: std::fmt::Display + Sized,
{
    store_err(r, error).map(box_to_ptr).unwrap_or(ptr::null())
}
/// This is a JSON-API, not a C-API, really.
#[derive(Serialize, Deserialize)]
struct ErrorMessage {
    error: String,
    context: String,
}

/// Accept a string parameter!
pub(crate) fn accept_str(name: &str, input: *const c_void) -> Result<&str, Box<dyn Error>> {
    if input.is_null() {
        Err(format!("NULL pointer: {}", name))?;
    }
    let input: &CStr = unsafe { CStr::from_ptr(input as *const c_char) };
    Ok(input
        .to_str()
        .map_err(|_| format!("Could not parse {} pointer as UTF-8 string!", name))?)
}

/// Internal helper: convert string reference to pointer to be passed to Python/C. Heap allocation.
pub(crate) fn return_string(output: &str) -> *const c_void {
    let c_output: CString = CString::new(output).expect("Conversion to CString should succeed!");
    CString::into_raw(c_output) as *const c_void
}

pub(crate) fn result_to_json(
    rust_result: Result<String, Box<dyn Error>>,
    error: *mut isize,
) -> *const c_void {
    store_err(rust_result, error)
        .map(|s| return_string(&s))
        .unwrap_or(ptr::null())
}

pub(crate) fn deserialize_from_cstr_json<'a, T: serde::Deserialize<'a>>(
    json_str: Result<&'a str, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_str(json_str?)?)
}

pub(crate) fn result_load_cqrel(
    data_path: Result<&str, Box<dyn Error>>,
) -> Result<QuerySetJudgments, Box<dyn Error>> {
    crate::qrel::read_file(data_path?)
}

pub(crate) fn result_cqrel_query_json(
    cqrel: Option<&CQRel>,
    query_str: Result<&str, Box<dyn Error>>,
) -> Result<String, Box<dyn Error>> {
    let cqrel = match cqrel {
        None => Err("cqrel pointer is null!")?,
        Some(x) => x,
    };
    Ok(match query_str? {
        "to_json" => serde_json::to_string(&cqrel.actual.query_to_judgments)?,
        "queries" => serde_json::to_string(&cqrel.actual.get_queries())?,
        qid => match cqrel.actual.get(qid) {
            None => Err(format!("Unknown request: {}", qid))?,
            Some(query_judgments) => serde_json::to_string(&query_judgments)?,
        },
    })
}

pub(crate) fn result_load_ranksvm_format(
    data_path: Result<&str, Box<dyn Error>>,
    feature_names_path: Option<Result<&str, Box<dyn Error>>>,
) -> Result<DatasetRef, Box<dyn Error>> {
    let feature_names = feature_names_path
        .transpose()?
        .map(|path| dataset::load_feature_names_json(path))
        .transpose()?;
    let data_path: &str = data_path?;
    Ok(DatasetRef::load_libsvm(data_path, feature_names.as_ref())
        .map_err(|e| format!("{}: {:?}", data_path, e))?)
}

pub(crate) fn result_dataset_query_sampling(
    dataset: Option<&CDataset>,
    queries_json_list: Result<&str, Box<dyn Error>>,
) -> Result<DatasetRef, Box<dyn Error>> {
    let dataset = match dataset {
        Some(d) => d,
        None => Err("Dataset pointer is null!")?,
    };

    let queries: Vec<String> = serde_json::from_str(queries_json_list?)?;
    Ok(dataset.reference.with_queries(&queries).into_ref())
}

pub(crate) fn result_dataset_feature_sampling(
    dataset: Option<&CDataset>,
    feature_json_list: Result<&str, Box<dyn Error>>,
) -> Result<DatasetRef, Box<dyn Error>> {
    let dataset = match dataset {
        Some(d) => d,
        None => Err("Dataset pointer is null!")?,
    };
    let features: Vec<FeatureId> = serde_json::from_str(feature_json_list?)?;

    Ok(dataset.reference.with_features(&features)?.into_ref())
}

pub(crate) fn result_dataset_query_json(
    dataset: Option<&CDataset>,
    query_str: Result<&str, Box<dyn Error>>,
) -> Result<String, Box<dyn Error>> {
    let dataset = match dataset {
        Some(d) => d,
        None => Err("Dataset pointer is null!")?,
    };

    let response = match query_str? {
        "is_sampled" => {
            if dataset.reference.is_sampled() {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        "num_features" => serde_json::to_string(&dataset.reference.n_dim())?,
        "feature_ids" => serde_json::to_string(&dataset.reference.features())?,
        "num_instances" => serde_json::to_string(&dataset.reference.instances().len())?,
        "queries" => serde_json::to_string(&dataset.reference.queries())?,
        "instances_by_query" => serde_json::to_string(&dataset.reference.instances_by_query())?,
        "feature_names" => {
            let names = dataset
                .reference
                .features()
                .into_iter()
                .map(|f| dataset.reference.feature_name(f))
                .collect::<Vec<_>>();
            serde_json::to_string(&names)?
        }
        other => Err(format!("unknown_dataset_query_str: {:?}", other))?,
    };

    Ok(response)
}

pub(crate) fn result_train_model(
    train_request: Result<TrainRequest, Box<dyn Error>>,
    dataset: Option<&CDataset>,
) -> Result<ModelEnum, Box<dyn Error>> {
    let dataset = match dataset {
        Some(d) => d,
        None => Err("Dataset pointer is null!")?,
    };
    Ok(json_api::do_training(train_request?, &dataset.reference)?)
}

pub(crate) fn result_model_query_json(
    model: Option<&CModel>,
    query_str: Result<&str, Box<dyn Error>>,
) -> Result<String, Box<dyn Error>> {
    let model = match model {
        Some(d) => d,
        None => Err("Model pointer is null!")?,
    };
    match query_str? {
        "to_json" => Ok(serde_json::to_string(&model.actual)?),
        other => Err(format!("unknown_dataset_query_str {}", other))?,
    }
}

pub(crate) fn result_exec_json(
    query_str: Result<&str, Box<dyn Error>>,
) -> Result<String, Box<dyn Error>> {
    let response = match query_str? {
        "coordinate_ascent_defaults" => serde_json::to_string(&TrainRequest {
            measure: "ndcg".to_string(),
            params: FastRankModelParams::CoordinateAscent(CoordinateAscentParams::default()),
            judgments: None,
        })?,
        "random_forest_defaults" => serde_json::to_string(&TrainRequest {
            measure: "ndcg".to_string(),
            params: FastRankModelParams::RandomForest(RandomForestParams::default()),
            judgments: None,
        })?,
        other => Err(serde_json::to_string(&ErrorMessage {
            error: "unknown_query_str".to_owned(),
            context: other.to_owned(),
        })?)?,
    };

    Ok(response)
}

fn require_pointer<'a, T>(name: &str, pointer: Option<&'a T>) -> Result<&'a T, Box<dyn Error>> {
    let inner = match pointer {
        Some(p) => p,
        None => Err(format!("{} pointer is null!", name))?,
    };
    Ok(inner)
}

pub(crate) fn result_evaluate_by_query(
    model: Option<&CModel>,
    dataset: Option<&CDataset>,
    qrel: Option<&CQRel>,
    evaluator: Result<&str, Box<dyn Error>>,
) -> Result<String, Box<dyn Error>> {
    let model = &require_pointer("Model", model)?.actual;
    let dataset = &require_pointer("Dataset", dataset)?.reference;
    let qrel = qrel.map(|cq| cq.actual.clone());
    let eval = SetEvaluator::create(dataset, evaluator?, qrel)?;
    let output = eval.evaluate_to_map(model);
    Ok(serde_json::to_string(&output)?)
}

pub(crate) fn result_predict_scores(
    model: Option<&CModel>,
    dataset: Option<&CDataset>,
) -> Result<String, Box<dyn Error>> {
    let model = &require_pointer("Model", model)?.actual;
    let dataset = &require_pointer("Dataset", dataset)?.reference;
    let written = json_api::predict_scores(model, dataset)?;
    Ok(serde_json::to_string(&written)?)
}

pub(crate) fn result_predict_to_trecrun(
    model: Option<&CModel>,
    dataset: Option<&CDataset>,
    output_path: Result<&str, Box<dyn Error>>,
    system_name: Result<&str, Box<dyn Error>>,
    depth: usize,
) -> Result<String, Box<dyn Error>> {
    let model = &require_pointer("Model", model)?.actual;
    let dataset = &require_pointer("Dataset", dataset)?.reference;
    let written = json_api::predict_to_trecrun(model, dataset, output_path?, system_name?, depth)?;
    Ok(serde_json::to_string(&written)?)
}
