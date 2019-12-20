#![crate_type = "dylib"]

#[macro_use]
extern crate serde_derive;

mod core;
pub(crate) use crate::core::FeatureId;
pub(crate) use crate::core::InstanceId;
pub(crate) use crate::core::Scored;

pub mod randutil;
/// Contains code for feature-at-a-time non-differentiable optimization.
pub mod coordinate_ascent;
pub mod dataset;
pub mod dense_dataset;
pub mod evaluators;
pub mod instance;
/// Contains code for reading compressed files based on their extension.
pub mod io_helper;
/// Contains code for reading ranklib and libsvm input files.
pub mod libsvm;
pub mod model;
pub mod normalizers;
pub mod qrel;
pub mod sampling;

pub mod json_api;

pub mod random_forest;
/// Streaming computation of statistics.
pub mod stats;

use dataset::DatasetRef;
use dense_dataset::DenseDataset;
use json_api::TrainRequest;
use model::ModelEnum;
use qrel::QuerySetJudgments;

use libc::{c_char, c_void};
use std::error::Error;
use std::ffi::CString;
use std::ptr;
use std::slice;

mod ffi;
use ffi::*;

pub struct CDataset {
    /// Reference to Rust-based Dataset.
    reference: DatasetRef,
}

pub struct CModel {
    actual: ModelEnum,
}

pub struct CQRel {
    actual: QuerySetJudgments,
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

#[no_mangle]
pub extern "C" fn free_cqrel(originally_from_rust: *mut CQRel) {
    let _will_drop: Box<CQRel> = unsafe { Box::from_raw(originally_from_rust) };
}

#[no_mangle]
pub extern "C" fn load_cqrel(data_path: *const c_void) -> *const CResult {
    result_to_c(
        result_load_cqrel(accept_str("data_path", data_path)).map(|actual| CQRel { actual }),
    )
}

#[no_mangle]
pub extern "C" fn cqrel_from_json(json_str: *const c_void) -> *const CResult {
    result_to_c(
        deserialize_from_cstr_json::<QuerySetJudgments>(accept_str("json_str", json_str))
            .map(|actual| CQRel { actual }),
    )
}

#[no_mangle]
pub extern "C" fn cqrel_query_json(cqrel: *const CQRel, query_str: *const c_void) -> *const c_void {
    let cqrel: Option<&CQRel> = unsafe { (cqrel as *mut CQRel).as_ref() };
    result_to_json(result_cqrel_query_json(
        cqrel,
        accept_str("query_str", query_str),
    ))
}

#[no_mangle]
pub extern "C" fn load_ranksvm_format(
    data_path: *mut c_void,
    feature_names_path: *mut c_void,
) -> *const CResult {
    let data_path = accept_str("data_path", data_path);
    let feature_names_path: Option<Result<&str, Box<dyn Error>>> = if feature_names_path.is_null() {
        None
    } else {
        Some(accept_str("feature_names_path", feature_names_path))
    };
    result_to_c(
        result_load_ranksvm_format(data_path, feature_names_path).map(|response| CDataset {
            reference: response,
        }),
    )
}

#[no_mangle]
pub extern "C" fn dataset_query_sampling(
    dataset: *mut CDataset,
    queries_json_list: *const c_void,
) -> *const CResult {
    let dataset: Option<&CDataset> = unsafe { (dataset as *mut CDataset).as_ref() };
    result_to_c(
        result_dataset_query_sampling(dataset, accept_str("queries_json_list", queries_json_list))
            .map(|response| CDataset {
                reference: response,
            }),
    )
}

#[no_mangle]
pub extern "C" fn dataset_feature_sampling(
    dataset: *mut CDataset,
    feature_json_list: *const c_void,
) -> *const CResult {
    let dataset: Option<&CDataset> = unsafe { (dataset as *mut CDataset).as_ref() };
    result_to_c(
        result_dataset_feature_sampling(
            dataset,
            accept_str("feature_json_list", feature_json_list),
        )
        .map(|response| CDataset {
            reference: response,
        }),
    )
}

#[no_mangle]
pub extern "C" fn dataset_query_json(
    dataset: *mut c_void,
    json_cmd_str: *mut c_void,
) -> *const c_void {
    let dataset: Option<&CDataset> = unsafe { (dataset as *mut CDataset).as_ref() };
    result_to_json(result_dataset_query_json(
        dataset,
        accept_str("dataset_query_json", json_cmd_str),
    ))
}

#[no_mangle]
pub extern "C" fn query_json(json_cmd_str: *const c_void) -> *const c_void {
    result_to_json(result_exec_json(accept_str("query_json_str", json_cmd_str)))
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
    result_to_c(
        DenseDataset::try_new(n, d, x_slice, y_slice, qid_slice).map(|dd| CDataset {
            reference: dd.into_ref(),
        }),
    )
}

#[no_mangle]
pub extern "C" fn train_model(
    train_request_json: *mut c_void,
    dataset: *mut c_void,
) -> *const CResult {
    let dataset: Option<&CDataset> = unsafe { (dataset as *mut CDataset).as_ref() };
    let request: Result<TrainRequest, _> =
        deserialize_from_cstr_json(accept_str("train_request_json", train_request_json));
    result_to_c(result_train_model(request, dataset).map(|actual| CModel { actual }))
}

#[no_mangle]
pub extern "C" fn model_from_json(json_str: *const c_void) -> *const CResult {
    result_to_c(
        deserialize_from_cstr_json::<ModelEnum>(accept_str("json_str", json_str))
            .map(|actual| CModel { actual }),
    )
}

#[no_mangle]
pub extern "C" fn model_query_json(
    model: *const c_void,
    json_cmd_str: *const c_void,
) -> *const c_void {
    let model: Option<&CModel> = unsafe { (model as *const CModel).as_ref() };
    result_to_json(result_model_query_json(
        model,
        accept_str("query_json", json_cmd_str),
    ))
}

/// returns json of qid->score for evaluator; or error-json.
#[no_mangle]
pub extern "C" fn evaluate_by_query(
    model: *const CModel,
    dataset: *const CDataset,
    qrel: *const CQRel,
    evaluator: *const c_void,
) -> *const c_void {
    let model: Option<&CModel> = unsafe { (model as *const CModel).as_ref() };
    let dataset: Option<&CDataset> = unsafe { (dataset as *const CDataset).as_ref() };
    let qrel: Option<&CQRel> = unsafe { (qrel as *const CQRel).as_ref() };
    let evaluator: Result<&str, Box<dyn Error>> = accept_str("evaluator_name", evaluator);
    result_to_json(result_evaluate_by_query(model, dataset, qrel, evaluator))
}

#[no_mangle]
pub extern "C" fn predict_to_trecrun(
    model: *const CModel,
    dataset: *const CDataset,
    output_path: *const c_void,
    system_name: *const c_void,
    depth: usize,
) -> *const c_void {
    let model: Option<&CModel> = unsafe { (model as *const CModel).as_ref() };
    let dataset: Option<&CDataset> = unsafe { (dataset as *const CDataset).as_ref() };
    let output_path: Result<&str, Box<dyn Error>> = accept_str("output_path", output_path);
    let system_name: Result<&str, Box<dyn Error>> = accept_str("system_name", system_name);
    result_to_json(result_predict_to_trecrun(
        model,
        dataset,
        output_path,
        system_name,
        depth,
    ))
}
