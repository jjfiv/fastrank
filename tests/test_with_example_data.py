import fastrank
from fastrank.training import CoordinateAscentParams, RandomForestParams
import unittest
import tempfile
import numpy as np
from typing import List
from sklearn.datasets import load_svmlight_file
from collections import Counter
from fastrank import CQRel, CDataset, query_json, TrainRequest
import pytest


def mean(xs: List[float]) -> float:
    """Strongly type a mean; since pyright hates np.mean"""
    return sum(xs) / len(xs)


_FEATURE_EXPECTED_NDCG5 = {
    "0": 0.10882970494872854,
    "para-fraction": 0.43942925167146063,
    "caption_position": 0.3838323029697044,
    "caption_count": 0.363671198812673,
    "pagerank": 0.28879573536768505,
    "caption_partial": 0.2119744912371782,
}
_FULL_QUERIES = set(
    """
        321 336 341 347 350 362 363 367 375 378
        393 397 400 408 414 422 426 427 433 439 
        442 445 626 646 690 801 802 803 804 805 
        806 807 808 809 810 811 812 813 814 815 
        816 817 818 819 820 821 822 823 824 825
""".split()
)
_EXPECTED_QUERIES = set(
    """378 363 811 321 807 347 646 397 802 804 
        808 445 819 820 426 626 393 824 442 433 
        825 350 823 422 336 400 814 817 439 822 
        690 816 801 805 367 810 813 818 414 812 
        809 362 341 803 375""".split()
)
_EXPECTED_N = 782
_EXPECTED_D = 6
_EXPECTED_FEATURE_IDS = set(range(_EXPECTED_D))
_EXPECTED_FEATURE_NAMES = set(
    [
        "0",
        "pagerank",
        "para-fraction",
        "caption_count",
        "caption_partial",
        "caption_position",
    ]
)
QREL = CQRel.load_file("examples/newsir18-entity.qrel")
RD = CDataset.open_ranksvm(
    "examples/trec_news_2018.train",
    "examples/trec_news_2018.features.json",
)


def test_version():
    assert fastrank.__version__ == "0.8.0"


# Test out "from_numpy:"
(TRAIN_X, TRAIN_Y, TRAIN_QID) = load_svmlight_file(
    "examples/trec_news_2018.train",
    dtype=np.float32,
    zero_based=False,
    query_id=True,
)

# Train a model for further tests:
TRAIN_REQ = TrainRequest.coordinate_ascent()
ca_params = TRAIN_REQ.params
ca_params.seed = 42
ca_params.quiet = True
MODEL = RD.train_model(TRAIN_REQ)


def test_cqrel_serialization():
    qrel = QREL.to_dict()
    qrel2 = CQRel.from_dict(qrel)
    assert qrel == qrel2.to_dict()


def test_cqrel():
    qrel = QREL
    assert qrel.queries() == _FULL_QUERIES
    assert set(qrel.to_dict().keys()) == _FULL_QUERIES


def test_load_dataset():
    rd = CDataset.open_ranksvm("examples/trec_news_2018.train")
    assert rd.queries() == _EXPECTED_QUERIES
    assert rd.feature_ids() == _EXPECTED_FEATURE_IDS
    assert rd.feature_names() == set(str(x) for x in _EXPECTED_FEATURE_IDS)
    assert rd.num_features() == _EXPECTED_D
    assert rd.num_instances() == _EXPECTED_N


def test_load_dataset_feature_names():
    assert RD.queries() == _EXPECTED_QUERIES
    assert RD.feature_ids() == _EXPECTED_FEATURE_IDS
    assert RD.feature_names() == _EXPECTED_FEATURE_NAMES
    assert RD.num_features() == _EXPECTED_D
    assert RD.num_instances() == _EXPECTED_N


def test_subsample_queries():
    rd = RD
    # Test subsample:
    _SUBSET = """378 363 811 321 807 347 646 397 802 804""".split()
    sample_rd = rd.subsample_queries(_SUBSET)
    assert sample_rd.queries() == set(_SUBSET)
    assert sample_rd.num_features() == _EXPECTED_D
    assert sample_rd.feature_ids() == _EXPECTED_FEATURE_IDS
    assert sample_rd.feature_names() == _EXPECTED_FEATURE_NAMES

    # calculate how many instances rust should've selected:
    count_by_qid = Counter(TRAIN_QID)
    expected_count = sum(count_by_qid[int(q)] for q in _SUBSET)
    assert sample_rd.num_instances() == expected_count

    # Train a model:
    train_req = TRAIN_REQ.clone()
    lp: CoordinateAscentParams = train_req.params  # type:ignore
    lp.num_restarts = 1
    lp.num_max_iterations = 1
    lp.step_base = 1.0
    lp.normalize = False
    lp.init_random = False
    model = sample_rd.train_model(train_req)
    sparse = model.predict_scores(sample_rd)
    assert len(sparse) == sample_rd.num_instances()
    dense = model.predict_dense_scores(sample_rd)
    assert len(dense) > len(sparse)
    # make sure that all the ids we asked for are present:
    for (_qid, ids) in sample_rd.instances_by_query().items():
        for num in ids:
            assert num < len(dense)


def test_subsample_features():
    rd = RD
    name_to_index = rd.feature_name_to_index()
    # single feature model:
    train_req = TRAIN_REQ.clone()
    lp: CoordinateAscentParams = train_req.params  # type:ignore
    lp.num_restarts = 1
    lp.num_max_iterations = 1
    lp.step_base = 1.0
    lp.normalize = False
    lp.init_random = False
    feature_scores = {}
    for feature in _EXPECTED_FEATURE_NAMES:
        rd_single = rd.subsample_feature_names([feature])
        model = rd_single.train_model(train_req)
        feature_scores[feature] = np.mean(list(rd.evaluate(model, "ndcg@5").values()))
        assert feature_scores[feature] == pytest.approx(
            _FEATURE_EXPECTED_NDCG5[feature]
        )
        my_index = name_to_index[feature]
        for (i, w) in enumerate(model.to_dict()["Linear"]["weights"]):
            if i == my_index:
                pass
            else:
                assert w == 0.0


def test_train_model():
    model = MODEL
    assert model != None
    model._require_init()


def test_random_forest():
    TRAIN_REQ = TrainRequest.random_forest()
    TRAIN_REQ.measure = "ndcg@5"
    rfp: RandomForestParams = TRAIN_REQ.params  # type:ignore
    rfp.num_trees = 10
    rfp.seed = 42
    rfp.min_leaf_support = 1
    rfp.max_depth = 10
    rfp.split_candidates = 32
    rfp.quiet = True

    measures = []
    for _ in range(10):
        model = RD.train_model(TRAIN_REQ)
        assert len(model.to_dict()["Ensemble"]["weights"]) == 10
        # for this particular dataset, there should be no difference between calculating with and without qrels:
        ndcg5_with: float = mean(list(RD.evaluate(model, "ndcg@5", QREL).values()))
        ndcg5_without: float = mean(list(RD.evaluate(model, "ndcg@5").values()))
        assert ndcg5_with == pytest.approx(ndcg5_without)
        measures.append(ndcg5_with)
    for m in measures:
        # SemVer change-detection: need to bump major version if this is no longer true!
        assert m == pytest.approx(0.4582452225554235)


def test_model_serialization():
    model = MODEL
    assert model != None
    model._require_init()
    # ensure a deep measure is the same:
    map_orig = RD.evaluate(model, "map")
    map_after_json = RD.evaluate(model.from_dict(model.to_dict()), "map")
    assert len(map_orig) == len(map_after_json)
    assert map_orig.keys() == map_after_json.keys()
    for key, val in map_orig.items():
        assert val == map_after_json[key]


def test_from_numpy():
    # this loader supports zero-based!
    EXPECTED_FEATURE_IDS = set(range(_EXPECTED_D - 1))

    # Test out "from_numpy:"
    train_X = TRAIN_X.todense()
    train_y = TRAIN_Y
    train_qid = TRAIN_QID
    train = CDataset.from_numpy(train_X, train_y, train_qid)

    (train_N, train_D) = train_X.shape
    assert train.is_sampled() == False
    assert train.num_features() == train_D
    assert train.num_instances() == train_N
    assert train.queries() == _EXPECTED_QUERIES
    assert train.feature_ids() == EXPECTED_FEATURE_IDS
    assert train.feature_names() == set(str(x) for x in EXPECTED_FEATURE_IDS)
    assert train.num_features() == _EXPECTED_D - 1
    assert train.num_instances() == _EXPECTED_N

    model = train.train_model(TRAIN_REQ)
    model._require_init()
    scores = model.predict_scores(train)
    assert 0 in scores
    assert len(scores) - 1 in scores
    assert len(scores) == len(train_y)


def test_evaluate():
    # for this particular dataset, there should be no difference between calculating with and without qrels:
    ndcg5_with = np.mean(list(RD.evaluate(MODEL, "ndcg@5", QREL).values()))
    ndcg5_without = np.mean(list(RD.evaluate(MODEL, "ndcg@5").values()))
    assert abs(ndcg5_with - ndcg5_without) < 0.0000001


def test_sampled_evaluation():
    measure = "ndcg@5"
    measures_by_query = RD.evaluate(MODEL, measure)

    first_ten_queries = sorted(measures_by_query.keys())[:10]

    assert RD.is_sampled() == False
    partial = RD.subsample_queries(first_ten_queries)
    assert partial.is_sampled() == True
    assert len(partial.instances_by_query()) == len(first_ten_queries)

    partial_scores = partial.evaluate(MODEL, measure)
    assert len(first_ten_queries) == len(partial_scores)
    for qid in first_ten_queries:
        assert partial_scores[qid] == pytest.approx(measures_by_query[qid])


def test_trecrun():
    with tempfile.NamedTemporaryFile(mode="r") as tmpf:
        with pytest.raises(Exception) as context:
            RD.predict_trecrun(MODEL, tmpf.name)
        assert "Dataset does not contain document ids" in str(context.value)


def TRAIN_REQ_object():
    rust = TrainRequest.from_dict(query_json("coordinate_ascent_defaults"))
    py = TrainRequest()

    for _ in range(2):
        assert rust.measure == py.measure
        assert rust.judgments == py.judgments
        assert isinstance(rust.params, CoordinateAscentParams)
        assert isinstance(py.params, CoordinateAscentParams)
        assert rust.params.num_restarts == py.params.num_restarts
        assert rust.params.num_max_iterations == py.params.num_max_iterations
        assert rust.params.step_base == py.params.step_base
        assert rust.params.step_scale == py.params.step_scale
        assert rust.params.tolerance == py.params.tolerance
        assert rust.params.init_random == py.params.init_random
        assert rust.params.output_ensemble == py.params.output_ensemble
        assert rust.params.quiet == py.params.quiet

        # no serialization issues!
        py = TrainRequest.from_dict(py.to_dict())
