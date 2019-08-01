#!/bin/bash

set -eu

DIR="$1"

cargo build --release && \
./target/release/best_single_feature "${DIR}/train0.ranklib" --feature_names "${DIR}/feature_names.json" -i 0 --metric2t ndcg@5
