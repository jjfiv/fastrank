#!/bin/bash

set -eu

cargo build --release && \
./target/release/best_single_feature news-bg-v4/train0.ranklib --feature_names news-bg-v4/feature_names.json "$@"
