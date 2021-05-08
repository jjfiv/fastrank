from fastrank import clib
from fastrank.clib import CDataset, CQRel, CModel, evaluate_query
import pytest


def test_query_json_err():
    with pytest.raises(ValueError):
        message = clib.query_json("missing_query")
        print(message)


def test_not_qrel():
    with pytest.raises(ValueError):
        message = CQRel.load_file("../README.md")
        print(message)


def test_not_qrel_json():
    with pytest.raises(ValueError):
        message = CQRel.from_dict({"cat": "woof"})  # type:ignore
        print(message)


def test_qrel_missing_query():
    qrel = CQRel.load_file("examples/newsir18-entity.qrel")
    with pytest.raises(ValueError):
        print(qrel._query_json("MISSING_QUERY"))


def test_import_model():
    with pytest.raises(ValueError):
        model = CModel.from_dict({"Linear": "JK"})
        model._require_init()


def test_import_model_ok():
    model = CModel.from_dict({"Linear": {"weights": [0.1, 0.9]}})
    model._require_init()


def test_bad_model_kind():
    with pytest.raises(AssertionError):
        model = CModel.from_dict({"LOL": "JK"})
        model._require_init()


def test_bad_training():
    class FakeTrainReq:
        def to_dict(self):
            return {"what": "Not Good"}

    dataset = CDataset.open_ranksvm("examples/trec_news_2018.train")
    with pytest.raises(ValueError):
        model = dataset.train_model(FakeTrainReq())
        model._require_init()


def test_bad_qsampling():
    dataset = CDataset.open_ranksvm("examples/trec_news_2018.train")
    with pytest.raises(ValueError):
        child = dataset.subsample_queries(["NOT_REAL"])
        child._require_init()


def test_bad_fsampling():
    dataset = CDataset.open_ranksvm("examples/trec_news_2018.train")
    with pytest.raises(KeyError):
        child = dataset.subsample_feature_names(["MISSING"])
        child._require_init()


def test_bad_dataset_query():
    dataset = CDataset.open_ranksvm("examples/trec_news_2018.train")
    with pytest.raises(ValueError):
        dataset._query_json("MISSING")


def test_evaluate_fake_measure():
    gains = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6]
    with pytest.raises(ValueError):
        num = evaluate_query("FAKE", gains, scores)
        print(num)


def test_query_model():
    model = CModel.from_dict({"Linear": {"weights": [0.1, 0.9]}})
    model._require_init()
    with pytest.raises(ValueError):
        print(model._query_json("FAKE"))