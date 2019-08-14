#![crate_type = "dylib"]
use libc::{c_char, c_void};
use serde_json;
#[macro_use]
extern crate serde_derive;
use fastrank::coordinate_ascent::CoordinateAscentParams;
use fastrank::dataset;
use fastrank::dataset::DatasetRef;
use fastrank::dataset::RankingDataset;
use fastrank::dense_dataset::DenseDataset;
use fastrank::json_api::*;
use fastrank::model::ModelEnum;
use fastrank::sampling::DatasetSampling;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

#[derive(Serialize, Deserialize)]
struct ErrorMessage {
    error: String,
    context: String,
}

pub struct CDataset {
    /// Reference to Rust-based Dataset.
    reference: Box<dyn RankingDataset>,
}

pub struct CModel {
    actual: ModelEnum,
}

#[repr(C)]
pub struct CResult {
    pub error_message: *const c_void,
    pub success: *const c_void,
}

impl Default for CResult {
    fn default() -> Self {
        CResult {
            error_message: ptr::null(),
            success: ptr::null(),
        }
    }
}

#[no_mangle]
pub extern "C" fn free_str(originally_from_rust: *mut c_void) {
    let _will_drop: CString = unsafe { CString::from_raw(originally_from_rust as *mut c_char) };
}

/// Note: not-recursive. Free Error Message Manually!
#[no_mangle]
pub extern "C" fn free_c_result(originally_from_rust: *mut CResult) {
    let _will_drop: Box<CResult> = unsafe { Box::from_raw(originally_from_rust) };
}

#[no_mangle]
pub extern "C" fn free_dataset(originally_from_rust: *mut CDataset) {
    let _will_drop: Box<CDataset> = unsafe { Box::from_raw(originally_from_rust) };
}

#[no_mangle]
pub extern "C" fn free_model(originally_from_rust: *mut CModel) {
    let _will_drop: Box<CModel> = unsafe { Box::from_raw(originally_from_rust) };
}

/// Internal helper: convert string reference to pointer to be passed to Python/C. Heap allocation.
fn return_string(output: &str) -> *const c_void {
    let c_output: CString = CString::new(output).expect("Conversion to CString should succeed!");
    CString::into_raw(c_output) as *const c_void
}

#[no_mangle]
pub extern "C" fn load_ranksvm_format(
    data_path: *mut c_void,
    feature_names_path: *mut c_void,
) -> *const CResult {
    let data_path: &CStr = unsafe { CStr::from_ptr(data_path as *mut c_char) };
    let feature_names_path: Option<&CStr> = if feature_names_path.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(feature_names_path as *mut c_char) })
    };
    let mut c_result = Box::new(CResult::default());
    match result_load_ranksvm_format(data_path, feature_names_path) {
        Ok(response) => {
            let c_dataset = Box::new(CDataset {
                reference: Box::new(response),
            });
            c_result.success = Box::into_raw(c_dataset) as *const c_void;
        }
        Err(e) => {
            let error_message = serde_json::to_string(&ErrorMessage {
                error: "error".to_string(),
                context: format!("{:?}", e),
            })
            .unwrap();
            c_result.error_message = return_string(&error_message);
        }
    };
    Box::into_raw(c_result)
}

fn result_load_ranksvm_format(
    data_path: &CStr,
    feature_names_path: Option<&CStr>,
) -> Result<DatasetRef, Box<Error>> {
    let data_path: &str = data_path
        .to_str()
        .map_err(|_| "Could not convert your data_path string to UTF-8!")?;
    let feature_names = feature_names_path
        .map(|s| {
            s.to_str()
                .map_err(|_| "Could not convert your feature_names_path string to UTF-8!")
        })
        .transpose()?
        .map(|path| dataset::load_feature_names_json(path))
        .transpose()?;
    Ok(DatasetRef::load_libsvm(data_path, feature_names.as_ref())?)
}

#[no_mangle]
pub extern "C" fn dataset_query_sampling(
    dataset: *mut CDataset,
    queries_json_list: *const c_void,
) -> *const CResult {
    let dataset: Option<&CDataset> = unsafe { (dataset as *mut CDataset).as_ref() };
    let queries_json_list: &CStr = unsafe { CStr::from_ptr(queries_json_list as *mut c_char) };
    let mut c_result = Box::new(CResult::default());
    match result_dataset_query_sampling(dataset, queries_json_list) {
        Ok(response) => {
            let c_dataset = Box::new(CDataset {
                reference: response,
            });
            c_result.success = Box::into_raw(c_dataset) as *const c_void;
        }
        Err(e) => {
            let error_message = serde_json::to_string(&ErrorMessage {
                error: "error".to_string(),
                context: format!("{:?}", e),
            })
            .unwrap();
            c_result.error_message = return_string(&error_message);
        }
    };
    Box::into_raw(c_result)
}

fn result_dataset_query_sampling(
    dataset: Option<&CDataset>,
    queries_json_list: &CStr,
) -> Result<Box<dyn RankingDataset>, Box<Error>> {
    let dataset = match dataset {
        Some(d) => d,
        None => Err("Dataset pointer is null!")?,
    };
    let queries_json_list: &str = queries_json_list
        .to_str()
        .map_err(|_| "Could not convert your query string to UTF-8!")?;

    let queries: Vec<String> = serde_json::from_str(queries_json_list)?;

    Ok(Box::new(dataset.reference.as_ref().with_queries(&queries)))
}

#[no_mangle]
pub extern "C" fn query_dataset_json(
    dataset: *mut c_void,
    json_cmd_str: *mut c_void,
) -> *const c_void {
    let dataset: Option<&CDataset> = unsafe { (dataset as *mut CDataset).as_ref() };
    let json_cmd_str: &CStr = unsafe { CStr::from_ptr(json_cmd_str as *mut c_char) };
    let output = match result_query_dataset_json(dataset, json_cmd_str) {
        Ok(response) => response,
        Err(e) => serde_json::to_string(&ErrorMessage {
            error: "error".to_string(),
            context: format!("{:?}", e),
        })
        .unwrap(),
    };
    let c_output: CString = CString::new(output).expect("Conversion to CString should succeed!");
    CString::into_raw(c_output) as *const c_void
}

fn result_query_dataset_json(
    dataset: Option<&CDataset>,
    query_str: &CStr,
) -> Result<String, Box<Error>> {
    let dataset = match dataset {
        Some(d) => d,
        None => Err("Dataset pointer is null!")?,
    };
    let query_str: &str = query_str
        .to_str()
        .map_err(|_| "Could not convert your query string to UTF-8!")?;

    let response = match query_str {
        "num_features" => serde_json::to_string(&dataset.reference.n_dim())?,
        "feature_ids" => serde_json::to_string(&dataset.reference.features())?,
        "num_instances" => serde_json::to_string(&dataset.reference.instances().len())?,
        "queries" => serde_json::to_string(&dataset.reference.queries())?,
        "feature_names" => {
            let names = dataset
                .reference
                .features()
                .into_iter()
                .map(|f| dataset.reference.feature_name(f))
                .collect::<Vec<_>>();
            serde_json::to_string(&names)?
        }
        other => serde_json::to_string(&ErrorMessage {
            error: "unknown_dataset_query_str".to_owned(),
            context: other.to_owned(),
        })?,
    };

    Ok(response)
}

#[no_mangle]
pub extern "C" fn query_json(json_cmd_str: *mut c_void) -> *const c_void {
    let json_cmd_str: &CStr = unsafe { CStr::from_ptr(json_cmd_str as *mut c_char) };
    let output = match result_exec_json(json_cmd_str) {
        Ok(response) => response,
        Err(e) => serde_json::to_string(&ErrorMessage {
            error: "error".to_string(),
            context: format!("{:?}", e),
        })
        .unwrap(),
    };
    let c_output: CString = CString::new(output).expect("Conversion to CString should succeed!");
    CString::into_raw(c_output) as *const c_void
}

fn result_exec_json(query_str: &CStr) -> Result<String, Box<Error>> {
    let query_str: &str = query_str
        .to_str()
        .map_err(|_| "Could not convert your query string to UTF-8!")?;

    let response = match query_str {
        "coordinate_ascent_defaults" => serde_json::to_string(&TrainRequest {
            measure: "ndcg".to_string(),
            params: FastRankModelParams::CoordinateAscent(CoordinateAscentParams::default()),
            judgments: None,
        })?,
        other => serde_json::to_string(&ErrorMessage {
            error: "unknown_query_str".to_owned(),
            context: other.to_owned(),
        })?,
    };

    Ok(response)
}

#[no_mangle]
pub extern "C" fn make_dense_dataset_f32_f64_i64(
    n: usize,
    d: usize,
    x: *const f32,
    y: *const f64,
    qids: *const i64,
) -> *const CResult {
    let x_len = n * d;
    let x_slice: &'static [f32] = unsafe { slice::from_raw_parts(x, x_len) };
    let y_slice: &'static [f64] = unsafe { slice::from_raw_parts(y, n) };
    let qid_slice: &'static [i64] = unsafe { slice::from_raw_parts(qids, n) };
    let mut c_result = Box::new(CResult::default());
    match DenseDataset::try_new(n, d, x_slice, y_slice, qid_slice) {
        Ok(dd) => {
            let success: Box<CDataset> = Box::new(CDataset {
                reference: Box::new(dd.into_ref()),
            });
            c_result.success = Box::into_raw(success) as *const c_void;
        }
        Err(e) => {
            let error_message = serde_json::to_string(&ErrorMessage {
                error: "error".to_string(),
                context: format!("{:?}", e),
            })
            .unwrap();
            c_result.error_message = return_string(&error_message);
        }
    };
    Box::into_raw(c_result)
}

#[no_mangle]
pub extern "C" fn train_model(json_cmd_str: *mut c_void, dataset: *mut c_void) -> *const CResult {
    let dataset: Option<&CDataset> = unsafe { (dataset as *mut CDataset).as_ref() };
    let json_cmd_str: &CStr = unsafe { CStr::from_ptr(json_cmd_str as *mut c_char) };

    let mut c_result = Box::new(CResult::default());
    match result_train_model(json_cmd_str, dataset) {
        Ok(m) => {
            let c_model = Box::new(CModel { actual: m });
            c_result.success = Box::into_raw(c_model) as *const c_void;
        }
        Err(e) => {
            let error_message = serde_json::to_string(&ErrorMessage {
                error: "error".to_string(),
                context: format!("{:?}", e),
            })
            .unwrap();
            c_result.error_message = return_string(&error_message);
        }
    };
    Box::into_raw(c_result)
}

fn result_train_model(
    json_cmd_str: &CStr,
    dataset: Option<&CDataset>,
) -> Result<ModelEnum, Box<Error>> {
    let train_request = json_cmd_str
        .to_str()
        .map_err(|_| "Could not convert your train_request string to UTF-8!")?;
    let dataset = match dataset {
        Some(d) => d,
        None => Err("Dataset pointer is null!")?,
    };

    let train_request: TrainRequest = serde_json::from_str(train_request)?;
    Ok(do_training(train_request, dataset.reference.as_ref())?)
}

#[no_mangle]
pub extern "C" fn query_model_json(model: *mut c_void, json_cmd_str: *mut c_void) -> *const c_void {
    let model: Option<&CModel> = unsafe { (model as *mut CModel).as_ref() };
    let json_cmd_str: &CStr = unsafe { CStr::from_ptr(json_cmd_str as *mut c_char) };
    let output = match result_query_model_json(model, json_cmd_str) {
        Ok(response) => response,
        Err(e) => serde_json::to_string(&ErrorMessage {
            error: "error".to_string(),
            context: format!("{:?}", e),
        })
        .unwrap(),
    };
    let c_output: CString = CString::new(output).expect("Conversion to CString should succeed!");
    CString::into_raw(c_output) as *const c_void
}

fn result_query_model_json(model: Option<&CModel>, query_str: &CStr) -> Result<String, Box<Error>> {
    let model = match model {
        Some(d) => d,
        None => Err("Model pointer is null!")?,
    };
    let query_str: &str = query_str
        .to_str()
        .map_err(|_| "Could not convert your query string to UTF-8!")?;

    let response = match query_str {
        "to_json" => serde_json::to_string(&model.actual)?,
        other => serde_json::to_string(&ErrorMessage {
            error: "unknown_dataset_query_str".to_owned(),
            context: other.to_owned(),
        })?,
    };

    Ok(response)
}
