#!/bin/bash

source venv/bin/activate

set -eu

cargo test

pip install -r cfastrank/requirements.txt

export RUST_BACKTRACE=1

cargo build --release 

rm -rf target/wheels
cd cfastrank && maturin build -b cffi --release && cd - && pip install target/wheels/*.whl
cd pyfastrank && python3 setup.py install && PYTHONPATH=. python3 tests/smoke_test.py && cd -
