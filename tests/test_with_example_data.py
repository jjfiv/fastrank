import unittest
import tempfile
import numpy as np
from sklearn.datasets import load_svmlight_file
from collections import Counter
from fastrank import CQRel, CDataset, query_json, TrainRequest

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


class TestRustAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qrel = CQRel.load_file("examples/newsir18-entity.qrel")
        cls.rd = CDataset.open_ranksvm(
            "examples/trec_news_2018.train",
            "examples/trec_news_2018.features.json",
        )
        # Test out "from_numpy:"
        (cls.train_X, cls.train_y, cls.train_qid) = load_svmlight_file(
            "examples/trec_news_2018.train",
            dtype=np.float32,
            zero_based=False,
            query_id=True,
        )
        cls.train_req = TrainRequest.coordinate_ascent()
        ca_params = cls.train_req.params
        ca_params.seed = 42
        ca_params.quiet = True
        cls.model = cls.rd.train_model(TestRustAPI.train_req)

    def test_cqrel_serialization(self):
        qrel = TestRustAPI.qrel.to_dict()
        qrel2 = CQRel.from_dict(qrel)
        self.assertEqual(qrel, qrel2.to_dict())

    def test_cqrel(self):
        qrel = TestRustAPI.qrel
        self.assertEqual(qrel.queries(), _FULL_QUERIES)
        self.assertEqual(set(qrel.to_dict().keys()), _FULL_QUERIES)

    def test_load_dataset(self):
        rd = CDataset.open_ranksvm("examples/trec_news_2018.train")
        assert rd.queries() == _EXPECTED_QUERIES
        assert rd.feature_ids() == _EXPECTED_FEATURE_IDS
        assert rd.feature_names() == set(str(x) for x in _EXPECTED_FEATURE_IDS)
        assert rd.num_features() == _EXPECTED_D
        assert rd.num_instances() == _EXPECTED_N

    def test_load_dataset_feature_names(self):
        rd = TestRustAPI.rd
        assert rd.queries() == _EXPECTED_QUERIES
        assert rd.feature_ids() == _EXPECTED_FEATURE_IDS
        assert rd.feature_names() == _EXPECTED_FEATURE_NAMES
        assert rd.num_features() == _EXPECTED_D
        assert rd.num_instances() == _EXPECTED_N

    def test_subsample_queries(self):
        rd = TestRustAPI.rd
        # Test subsample:
        _SUBSET = """378 363 811 321 807 347 646 397 802 804""".split()
        sample_rd = rd.subsample_queries(_SUBSET)
        assert sample_rd.queries() == set(_SUBSET)
        assert sample_rd.num_features() == _EXPECTED_D
        assert sample_rd.feature_ids() == _EXPECTED_FEATURE_IDS
        assert sample_rd.feature_names() == _EXPECTED_FEATURE_NAMES

        # calculate how many instances rust should've selected:
        count_by_qid = Counter(TestRustAPI.train_qid)
        expected_count = sum(count_by_qid[int(q)] for q in _SUBSET)
        assert sample_rd.num_instances() == expected_count

    def test_subsample_features(self):
        rd = TestRustAPI.rd
        name_to_index = rd.feature_name_to_index()
        # single feature model:
        train_req = TestRustAPI.train_req.clone()
        lp = train_req.params
        lp.num_restarts = 1
        lp.num_max_iterations = 1
        lp.step_base = 1.0
        lp.normalize = False
        lp.init_random = False
        feature_scores = {}
        for feature in _EXPECTED_FEATURE_NAMES:
            rd_single = rd.subsample_feature_names([feature])
            model = rd_single.train_model(train_req)
            feature_scores[feature] = np.mean(
                list(rd.evaluate(model, "ndcg@5").values())
            )
            self.assertAlmostEqual(
                feature_scores[feature],
                _FEATURE_EXPECTED_NDCG5[feature],
                msg="NDCG@5 single-feature ranker expectation: {0}".format(feature),
            )
            my_index = name_to_index[feature]
            for (i, w) in enumerate(model.to_dict()["Linear"]["weights"]):
                if i == my_index:
                    pass
                else:
                    self.assertAlmostEqual(w, 0.0, "Every other weight should be zero.")

    def test_train_model(self):
        rd = TestRustAPI.rd
        model = TestRustAPI.model
        self.assertIsNotNone(model)
        model._require_init()

    def test_random_forest(self):
        rd = TestRustAPI.rd
        train_req = TrainRequest.random_forest()
        train_req.measure = "ndcg@5"
        train_req.params.num_trees = 10
        train_req.params.seed = 42
        train_req.params.min_leaf_support = 1
        train_req.params.max_depth = 10
        train_req.params.split_candidates = 32
        train_req.params.quiet = True

        measures = []
        for _ in range(10):
            model = rd.train_model(train_req)
            self.assertEqual(len(model.to_dict()["Ensemble"]["weights"]), 10)
            # for this particular dataset, there should be no difference between calculating with and without qrels:
            ndcg5_with = np.mean(
                list(rd.evaluate(model, "ndcg@5", TestRustAPI.qrel).values())
            )
            ndcg5_without = np.mean(list(rd.evaluate(model, "ndcg@5").values()))
            self.assertAlmostEqual(ndcg5_with, ndcg5_without)
            measures.append(ndcg5_with)
        for m in measures:
            # SemVer change-detection: need to bump major version if this is no longer true!
            self.assertAlmostEqual(m, 0.436791451738704)

    def test_model_serialization(self):
        rd = TestRustAPI.rd
        model = TestRustAPI.model
        self.assertIsNotNone(model)
        model._require_init()
        # ensure a deep measure is the same:
        map_orig = rd.evaluate(model, "map")
        map_after_json = rd.evaluate(model.from_dict(model.to_dict()), "map")
        self.assertAlmostEqual(map_orig, map_after_json)

    def test_from_numpy(self):
        # this loader supports zero-based!
        EXPECTED_FEATURE_IDS = set(range(_EXPECTED_D - 1))

        # Test out "from_numpy:"
        train_X = TestRustAPI.train_X.todense()
        train_y = TestRustAPI.train_y
        train_qid = TestRustAPI.train_qid
        train = CDataset.from_numpy(train_X, train_y, train_qid)

        (train_N, train_D) = train_X.shape
        assert train.num_features() == train_D
        assert train.num_instances() == train_N
        assert train.queries() == _EXPECTED_QUERIES
        assert train.feature_ids() == EXPECTED_FEATURE_IDS
        assert train.feature_names() == set(str(x) for x in EXPECTED_FEATURE_IDS)
        assert train.num_features() == _EXPECTED_D - 1
        assert train.num_instances() == _EXPECTED_N

        model = train.train_model(TestRustAPI.train_req)
        model._require_init()

    def test_evaluate(self):
        rd = TestRustAPI.rd
        model = TestRustAPI.model
        # for this particular dataset, there should be no difference between calculating with and without qrels:
        ndcg5_with = np.mean(
            list(rd.evaluate(model, "ndcg@5", TestRustAPI.qrel).values())
        )
        ndcg5_without = np.mean(list(rd.evaluate(model, "ndcg@5").values()))
        assert abs(ndcg5_with - ndcg5_without) < 0.0000001

    def test_sampled_evaluation(self):
        rd = TestRustAPI.rd
        model = TestRustAPI.model
        measure = "ndcg@5"
        measures_by_query = rd.evaluate(model, measure)

        first_ten_queries = sorted(measures_by_query.keys())[:10]

        partial = rd.subsample_queries(first_ten_queries)
        self.assertEqual(len(partial.instances_by_query()), len(first_ten_queries))

        partial_scores = partial.evaluate(model, measure)
        self.assertEqual(len(first_ten_queries), len(partial_scores))
        for qid in first_ten_queries:
            self.assertAlmostEqual(partial_scores[qid], measures_by_query[qid])

    def test_trecrun(self):
        rd = TestRustAPI.rd
        model = TestRustAPI.model
        with tempfile.NamedTemporaryFile(mode="r") as tmpf:
            with self.assertRaises(Exception) as context:
                rd.predict_trecrun(model, tmpf.name)
            self.assertRegex(
                str(context.exception), "Dataset does not contain document ids"
            )

    def train_req_object(self):
        rust = TrainRequest.from_dict(query_json("coordinate_ascent_defaults"))
        py = TrainRequest()

        for _ in range(2):
            self.assertEqual(rust.measure, py.measure)
            self.assertEqual(rust.judgments, py.judgments)
            self.assertEqual(rust.params.num_restarts, py.params.num_restarts)
            self.assertEqual(
                rust.params.num_max_iterations, py.params.num_max_iterations
            )
            self.assertAlmostEqual(rust.params.step_base, py.params.step_base)
            self.assertAlmostEqual(rust.params.step_scale, py.params.step_scale)
            self.assertAlmostEqual(rust.params.tolerance, py.params.tolerance)
            self.assertEqual(rust.params.init_random, py.params.init_random)
            self.assertEqual(rust.params.output_ensemble, py.params.output_ensemble)
            self.assertEqual(rust.params.quiet, py.params.quiet)

            # no serialization issues!
            py = TrainRequest.from_dict(py.to_dict())


if __name__ == "__main__":
    unittest.main()
