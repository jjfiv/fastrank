from fastrank import clib
from fastrank.clib import CDataset, CQRel, CModel, evaluate_query
from fastrank.models import CoordinateAscentRanker
import pytest

dataset = CDataset.open_ranksvm("examples/trec_news_2018.train")
qrel = CQRel.load_file("examples/newsir18-entity.qrel")
MEASURE = "ndcg@10"


def test_query_json_err():
    model = CoordinateAscentRanker("ndcg")
    model.qrel = qrel
    model.params.num_restarts = 1
    model.fit_dataset(dataset)

    full_at_10 = model.score_dataset(dataset, MEASURE)
    best = dataset.select_topk(model.model, 10)
    part_at_10 = model.score_dataset(best, MEASURE)

    assert full_at_10 == pytest.approx(part_at_10)
