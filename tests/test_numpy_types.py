from sklearn.datasets import load_svmlight_file
from fastrank import CDataset
import numpy as np
from typing import List
from fastrank.models import CoordinateAscentRanker, CoordinateAscentParams

(train_X, y, qid) = load_svmlight_file(
    "examples/trec_news_2018.train",
    zero_based=False,
    query_id=True,
)


def cheap_linear() -> CoordinateAscentRanker:
    return CoordinateAscentRanker(
        params=CoordinateAscentParams(
            seed=1234,
            num_restarts=1,
            quiet=True,
        )  # type:ignore
    )


def test_default_dtypes():
    X = train_X.todense()
    assert X[100, 3] == train_X[100, 3]
    dataset = CDataset.from_numpy(X, y, qid)
    assert dataset.num_instances() == 782
    assert dataset.num_features() == 5
    m = cheap_linear()
    m.fit_dataset(dataset)
    print(m.weights())
    assert (m.score_dataset(dataset)) >= 0.726


def test_smaller_dtypes():
    X = (train_X.todense()).astype("float32")
    assert X[100, 3] == np.float32(train_X[100, 3])
    assert X.dtype == "float32"

    dataset = CDataset.from_numpy(X, y.astype("float32"), qid.astype("int32"))
    assert dataset.num_instances() == 782
    assert dataset.num_features() == 5

    m = cheap_linear()
    m.fit_dataset(dataset)
    assert (m.score_dataset(dataset)) >= 0.726


def test_int_ys():
    dataset = CDataset.from_numpy(
        train_X.todense().astype("float32"), y.astype("int32"), qid.astype("int32")
    )
    assert dataset.num_instances() == 782
    assert dataset.num_features() == 5

    m = cheap_linear()
    m.fit_dataset(dataset)
    assert (m.score_dataset(dataset)) >= 0.726


def test_qstrings():
    X = train_X.todense()
    qids: List[str] = ["q{}".format(qn) for qn in qid]
    dataset = CDataset.from_numpy(X, y, qids)
    assert dataset.num_instances() == 782
    assert dataset.num_features() == 5
    assert dataset.queries() == set(qids)

    m = cheap_linear()
    m.fit_dataset(dataset)
    assert (m.score_dataset(dataset)) >= 0.726


def test_to_arrays():
    X = train_X.todense()
    dataset = CDataset.from_numpy(X, y, qid)
    X2 = dataset._to_dense_X()
    assert X.shape == X2.shape
    assert np.allclose(X, X2)

    y2 = dataset.get_gains()
    assert y.shape == y2.shape
    assert np.allclose(y, y2)

    qid2 = dataset.get_query_vec()
    assert len(qid2) == len(qid)
    assert [int(q) for q in qid2] == qid.tolist()
