#!/bin/bash

set -eu

cargo build --release 
time ./target/release/coordinate_ascent examples/trec_news_2018.train --seed=42 --test=examples/trec_news_2018.test
time ./target/release/coordinate_ascent examples/trec_news_2018.train --seed=42 --test=examples/trec_news_2018.test --quiet --init_random
