# FastRank [![Build Status](https://travis-ci.com/jjfiv/fastrank.svg?token=wqGZxUYsDSPaq1jz2zn6&branch=master)](https://travis-ci.com/jjfiv/fastrank) [![PyPI version](https://badge.fury.io/py/fastrank.svg)](https://badge.fury.io/py/fastrank)


My most frequently used learning-to-rank algorithms ported to rust for efficiency.

Read my [blog-post](https://jjfoley.me/2019/10/11/fastrank-alpha.html) announcing the first public version: 0.4. It's alpha because I think the API needs work, not because there's any sort of known correctness or compatiblity issues.

## Python Usage 

```shell
pip install fastrank
```

### Configuring Models

```python
from fastrank import CModel, CDataset, CQRel, TrainRequest

RANDOM_FOREST = False

if RANDOM_FOREST:
    train_request = TrainRequest.random_forest()
    params = train_request.params
    params.num_trees = 200
    params.feature_sampling_rate = 0.5
    params.instance_sampling_rate = 0.5
else:
    train_request = TrainRequest.coordinate_ascent()
    params = train_request.params
    params.init_random = True
    params.normalize = True
    
# No matter what, deterministic seed and limit print statements.
params.quiet = True
params.seed = 16710601535089033473
```

### Loading SVMrank/Ranklib files:

```python
import os

query_dir = os.path.join(os.environ['HOME'], 'code', 'queries', 'trec_news')
qrels = CQRel.load_file(os.path.join(query_dir, 'newsir18-entity.qrel'))

dataset = CDataset.open_ranksvm(
    os.path.join(data_dir, "ent.ranklib.gz"),
    os.path.join(data_dir, "feature_names.json"),
)
```

### Train & Evaluate Models

```python
from sklearn.model_selection import KFold

EVAL_MEASURE = "NDCG@5"

models = []
evals = []
folds = KFold(n_splits=5, random_state=0, shuffle=False)
features = dataset.feature_names()
features.remove("0") # ranksvm starts at 1 for many tools
queries = sorted(d2018.queries())

fdataset = d2018.subsample_feature_names(features)

for train_idx, test_idx in folds.split(queries):
    train_queries = [queries[i] for i in train_idx]
    test_queries = [queries[i] for i in test_idx]
    train = fdataset.subsample_queries(train_queries)
    test = fdataset.subsample_queries(test_queries)
    model = train.train_model(train_request)
    eval_dict = test.evaluate(model, EVAL_MEASURE, qrels)
    evals.append(eval_dict)
    models.append(model)
    print("  NDCG@5 = %1.3f" % np.mean(list(eval_dict.values())))
```

## Code Structure

### fastrank 

The core algorithms and data structures are implemented in Rust.

### cfastrank [![PyPI version](https://badge.fury.io/py/cfastrank.svg)](https://badge.fury.io/py/cfastrank)

A very thin layer of rust code provides a C-compatible API. A manylinux version is published to pypi. Don't install this manually -- install the ``fastrank`` package and let it be pulled in as a dependency.

### pyfastrank

A pure-python libary accesses the core algorithms using cffi via cfastrank. A version is published to pypi.
