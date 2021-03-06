#!/bin/bash

source venv/bin/activate

set -eu

cargo test

pip uninstall fastrank -y
pip install -r requirements.txt

export RUST_BACKTRACE=1

cargo build --release 

rm -rf target/wheels

maturin build -b cffi --release && pip install target/wheels/*.whl

python -I -m unittest discover -s tests -v
