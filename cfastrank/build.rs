// from https://github.com/getsentry/milksnake
extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut config: cbindgen::Config = Default::default();
    config.language = cbindgen::Language::C;
    cbindgen::generate_with_config(&crate_dir, config)
        .expect("Cbindgen::error")
        .write_to_file("../target/cfastrank.h");
}
