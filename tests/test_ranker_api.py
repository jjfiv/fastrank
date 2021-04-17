import fastrank
from fastrank.models import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.datasets import load_svmlight_file

(X, y, qid) = load_svmlight_file(
    "examples/trec_news_2018.train",
    dtype=np.float32,
    zero_based=False,
    query_id=True,
)
X = X.todense()

measure = "ndcg"
RAND = 1234


def test_rf():
    rf = RandomForestRanker(measure)
    rf.params.seed = RAND
    rf.params.quiet = True
    rf.fit(X, y, qid)
    assert rf.score(X, y, qid) > 0.70


def test_sklearn_rf():
    rf = RandomForestRegressor(random_state=RAND)
    rf.fit(X, y)
    rf = ImportedRanker.from_sklearn(rf)
    assert rf.score(X, y, qid, measure) > 0.85


def test_sklearn_gbc():
    rf = GradientBoostingClassifier(random_state=RAND)
    rf.fit(X, y > 0)
    rf = ImportedRanker.from_sklearn(rf)
    assert rf.score(X, y, qid, measure) > 0.70


def test_sklearn_gbr():
    rf = GradientBoostingRegressor(random_state=RAND)
    rf.fit(X, y)
    rf = ImportedRanker.from_sklearn(rf)
    assert rf.score(X, y, qid, measure) > 0.70


def test_ca():
    ca = CoordinateAscentRanker(measure)
    ca.params.quiet = True
    ca.params.seed = RAND
    ca.fit(X, y, qid)
    assert ca.score(X, y, qid) > 0.7


def test_sklearn_lr():
    sk = LinearRegression()
    sk.fit(X, y)
    lr = ImportedRanker.from_sklearn(sk)
    assert lr.score(X, y, qid, measure) > 0.74
