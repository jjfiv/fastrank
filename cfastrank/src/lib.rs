#![crate_type = "dylib"]
use libc::{c_char, c_void};
use serde_json;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::mem;
use std::slice;
use fastrank::json_api::*;
use fastrank::dense_dataset::DenseDataset;

#[no_mangle]
pub extern "C" fn free_str(originally_from_rust: *mut c_void) {
    let _will_drop: CString = unsafe { CString::from_raw(originally_from_rust as *mut c_char) };
}

#[no_mangle]
pub extern "C" fn exec_json(json_cmd_str: *mut c_void) -> *const c_void {
    let json_cmd_str: &CStr = unsafe { CStr::from_ptr(json_cmd_str as *mut c_char) };
    let output = match result_exec_json(json_cmd_str) {
        Ok(response) => response,
        Err(e) => format!("Error: {:?}", e),
    };
    let c_output: CString = CString::new(output).expect("Conversion to CString should succeed!");
    CString::into_raw(c_output) as *const c_void
}

fn result_exec_json(query_str: &CStr) -> Result<String, Box<Error>> {
    let query_str: &str = query_str
        .to_str()
        .map_err(|_| "Could not convert your query string to UTF-8!")?;

    let args: serde_json::Value = serde_json::from_str(query_str)?;
    Ok(format!("result_exec_json: {:?}", args))
}

#[no_mangle]
pub extern "C" fn train_dense_dataset_f32_f64_i64(json_cmd_str: *mut c_void, n: usize, d: usize, x: *const f32, y: *const f64, qids: *const i64) -> *const c_void {
    let x_len = n * d;
    let x_slice: &'static [f32] = unsafe { slice::from_raw_parts(x, x_len) };
    let y_slice: &'static [f64] = unsafe { slice::from_raw_parts(y, n) };
    let qid_slice: &'static [i64] = unsafe { slice::from_raw_parts(qids, n) };
    let json_cmd_str: &CStr = unsafe { CStr::from_ptr(json_cmd_str as *mut c_char) };

    let output = match train_dense_dataset_inner(json_cmd_str, n, d, x_slice, y_slice, qid_slice) {
        Ok(message) => message,
        Err(e) => format!("Error: {:?}", e),
    };

    let c_output: CString = CString::new(output).expect("Conversion to CString should succeed!");
    CString::into_raw(c_output) as *const c_void
}


fn train_dense_dataset_inner(json_cmd_str: &CStr, n: usize, d: usize, x_arr: &'static [f32], y_arr: &'static [f64], qids: &'static [i64]) -> Result<String, Box<Error>> {
    let train_request = json_cmd_str
        .to_str()
        .map_err(|_| "Could not convert your query string to UTF-8!")?;

    let train_request: TrainRequest = serde_json::from_str(train_request)?;
    let dataset = DenseDataset::try_new(n, d, x_arr, y_arr, qids)?.into_ref();
    let response = do_training(train_request, dataset)?;
    Ok(serde_json::to_string(&response)?)
}