#!/bin/bash

set -eu

cargo build --release 
time ./target/release/coordinate_ascent examples/trec_news_2018.train --seed=42 --test=examples/trec_news_2018.test --feature_names examples/trec_news_2018.features.json -i 0 --quiet --init_random
time ./target/release/train_random_forest examples/trec_news_2018.train --seed=42 --test=examples/trec_news_2018.test --feature_names examples/trec_news_2018.features.json -i 0 --quiet
