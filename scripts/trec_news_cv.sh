#!/bin/bash

#DIR="${HOME}/code/irene/scripts"
DIR="bg-ranklib.v4"

cargo build --release
for x in 0 1 2 3 4; do
  ./target/release/coordinate_ascent ${DIR}/train${x}.ranklib \
    --test ${DIR}/test${x}.ranklib \
    --feature_names ${DIR}/feature_names.json \
    --normalize_weights \
    --metric2t ndcg \
    -i 0 -i clickbait_proba \
    --seed 13 | tee logs/no_click13.bg_ca_v4.${x}
  
  #./target/release/train_random_forest ${DIR}/train${x}.ranklib \
  #--test ${DIR}/test${x}.ranklib \
  #--feature_names ${DIR}/feature_names.json \
  #--metric2t ndcg \
  #--srate 0.5 \
  #--frate 0.5 \
  #--split_candidates 16 \
  #--max_depth 7 \
  #-i 0 -i clickbait_proba \
  #--seed 42 | tee logs/no_click.bg_rf_v4.${x}
done

