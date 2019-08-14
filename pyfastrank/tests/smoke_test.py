import unittest
import numpy as np
import ujson as json
import sklearn
from sklearn.datasets import load_svmlight_file
from collections import Counter
from fastrank import CQRel, CDataset, CModel, query_json

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


class TestRustAPI(unittest.TestCase):
    def setUp(self):
        self.qrel = CQRel()
        self.qrel.load_file("../examples/newsir18-entity.qrel")
        self.rd = CDataset()
        self.rd.open_ranksvm(
            "../examples/trec_news_2018.train",
            "../examples/trec_news_2018.features.json",
        )
        # Test out "from_numpy:"
        (self.train_X, self.train_y, self.train_qid) = load_svmlight_file(
            "../examples/trec_news_2018.train",
            dtype=np.float32,
            zero_based=False,
            query_id=True,
        )
        self.train_req = query_json("coordinate_ascent_defaults")
        ca_params = self.train_req["params"]["CoordinateAscent"]
        ca_params["init_random"] = True
        ca_params["seed"] = 42
        ca_params["quiet"] = True

    def test_cqrel(self):
        qrel = self.qrel
        self.assertEqual(qrel.queries(), _FULL_QUERIES)
        self.assertEqual(set(qrel.to_json().keys()), _FULL_QUERIES)

    def test_load_dataset(self):
        rd = CDataset()
        rd.open_ranksvm("../examples/trec_news_2018.train")
        assert rd.queries() == _EXPECTED_QUERIES
        assert rd.feature_ids() == _EXPECTED_FEATURE_IDS
        assert rd.feature_names() == set(str(x) for x in _EXPECTED_FEATURE_IDS)
        assert rd.num_features() == _EXPECTED_D
        assert rd.num_instances() == _EXPECTED_N

    def test_load_dataset_feature_names(self):
        rd = self.rd
        assert rd.queries() == _EXPECTED_QUERIES
        assert rd.feature_ids() == _EXPECTED_FEATURE_IDS
        assert rd.feature_names() == _EXPECTED_FEATURE_NAMES
        assert rd.num_features() == _EXPECTED_D
        assert rd.num_instances() == _EXPECTED_N

    def test_subsample(self):
        rd = self.rd
        # Test subsample:
        _SUBSET = """378 363 811 321 807 347 646 397 802 804""".split()
        sample_rd = rd.subsample(_SUBSET)
        assert sample_rd.queries() == set(_SUBSET)
        assert sample_rd.num_features() == _EXPECTED_D
        assert sample_rd.feature_ids() == _EXPECTED_FEATURE_IDS
        assert sample_rd.feature_names() == _EXPECTED_FEATURE_NAMES

        # calculate how many instances rust should've selected:
        count_by_qid = Counter(self.train_qid)
        expected_count = sum(count_by_qid[int(q)] for q in _SUBSET)
        assert sample_rd.num_instances() == expected_count

    def test_train_model(self):
        rd = self.rd
        model = rd.train_model(self.train_req)
        self.assertIsNotNone(model)
        model._require_init()

    def test_from_numpy(self):
        # this loader supports zero-based!
        EXPECTED_FEATURE_IDS = set(range(_EXPECTED_D - 1))

        # Test out "from_numpy:"
        train_X = self.train_X.todense()
        train_y = self.train_y
        train_qid = self.train_qid
        train = CDataset()
        train.from_numpy(train_X, train_y, train_qid)

        (train_N, train_D) = train_X.shape
        assert train.num_features() == train_D
        assert train.num_instances() == train_N
        assert train.queries() == _EXPECTED_QUERIES
        assert train.feature_ids() == EXPECTED_FEATURE_IDS
        assert train.feature_names() == set(str(x) for x in EXPECTED_FEATURE_IDS)
        assert train.num_features() == _EXPECTED_D - 1
        assert train.num_instances() == _EXPECTED_N

        model = train.train_model(self.train_req)
        model._require_init()

    def test_evaluate(self):
        rd = self.rd
        model = rd.train_model(self.train_req)
        # for this particular dataset, there should be no difference between calculating with and without qrels:
        ndcg5_with = np.mean(list(rd.evaluate(model, "ndcg@5", self.qrel).values()))
        ndcg5_without = np.mean(list(rd.evaluate(model, "ndcg@5").values()))
        assert abs(ndcg5_with - ndcg5_without) < 0.0000001


if __name__ == "__main__":
    unittest.main()
