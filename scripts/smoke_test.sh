#!/bin/bash

source venv/bin/activate

set -eu

pip install -r cfastrank/requirements.txt
pip install -r pyfastrank/requirements.txt

export RUST_BACKTRACE=1

cargo build --release 
time ./target/release/best_single_feature examples/trec_news_2018.train --test=examples/trec_news_2018.test --feature_names examples/trec_news_2018.features.json -i 0
time ./target/release/coordinate_ascent examples/trec_news_2018.train --seed=42 --test=examples/trec_news_2018.test --feature_names examples/trec_news_2018.features.json -i 0 --quiet --init_random
time ./target/release/train_random_forest examples/trec_news_2018.train --seed=42 --test=examples/trec_news_2018.test --feature_names examples/trec_news_2018.features.json -i 0 --quiet

cd cfastrank && pyo3-pack develop -b cffi --release && cd -
cd pyfastrank && PYTHONPATH=. python3 tests/smoke_test.py && cd -
