#!/bin/bash

#DIR="${HOME}/code/irene/scripts"
DIR="news-bg"

cargo build --release && 
  ./target/release/coordinate_ascent ${DIR}/train0.ranklib \
  --test ${DIR}/test0.ranklib \
  --feature_names ${DIR}/feature_names.json \
  --normalize_weights \
  --metric2t ndcg \
  --seed 42 "$@"

