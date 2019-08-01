#!/bin/bash

#DIR="${HOME}/code/irene/scripts"
DIR="news-bg-v4"

cargo build --release && 
  ./target/release/train_random_forest ${DIR}/train0.ranklib \
  --test ${DIR}/test0.ranklib \
  --feature_names ${DIR}/feature_names.json \
  --metric2t ndcg \
  -i 0 \
  --seed 42 "$@"

