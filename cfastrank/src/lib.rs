#![crate_type = "dylib"]
use libc::{c_char, c_void};
use serde_json;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::mem;
use std::slice;

use fastrank::dataset::DatasetRef;
use fastrank::model::Model;
use fastrank::{FeatureId, InstanceId};

#[repr(C)]
pub struct CModel {
    inner: Box<Model>,
}
#[repr(C)]
pub struct CDataset {
    inner: Box<DatasetRef>,
}

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
pub extern "C" fn create_dense_dataset_f32_f64_i64(n: usize, d: usize, x: *const f32, y: *const f64, qids: *const i64) {
    let x_len = n * d;
    let x_slice = unsafe { slice::from_raw_parts(x, x_len) };
    let y_slice = unsafe { slice::from_raw_parts(y, n) };
    let qid_slice = unsafe { slice::from_raw_parts(qids, n) };
    for fid in 0..d {
        println!("f[17][{}] = {}", fid, x_slice[17 * d + fid]);
    }
    println!("y[17]={}", y_slice[17]);
    println!("qid[17]={}", qid_slice[17]);
}