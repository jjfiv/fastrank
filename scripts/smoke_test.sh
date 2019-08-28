#!/bin/bash

source venv/bin/activate

set -eu

cargo test

pip install -r cfastrank/requirements.txt
pip install -r pyfastrank/requirements.txt

export RUST_BACKTRACE=1

cargo build --release 

cd cfastrank && pyo3-pack develop -b cffi --release && cd -
cd pyfastrank && PYTHONPATH=. python3 tests/smoke_test.py && cd -
