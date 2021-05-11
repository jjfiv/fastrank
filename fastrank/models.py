from abc import ABC, abstractmethod
from .training import CoordinateAscentParams, RandomForestParams, TrainRequest
from .clib import CModel, CDataset, CQRel
from typing import List, Optional, Dict, Any
import numpy as np


def make_dataset(X: np.ndarray, y: np.ndarray, qids: List[str]) -> CDataset:
    (N, _) = X.shape
    assert y.shape == (N, 1) or y.shape == (N,)
    return CDataset.from_numpy(X, y, qids)


class BaseRanker(ABC):
    """
    This is the base class that all of our Rankers implement.
    """

    def __init__(self, measure: str, qrel: Optional[CQRel] = None):
        """ All Rankers at least have a measure and the chance for a qrel file. """
        self.measure = measure
        """ The measure to optimize: ``"ndcg"``, ``"map"``, and ``"mrr"`` are supported. """
        self.model: Optional[CModel] = None
        """ After `fit` or `fit_dataset` are called, this contains the learned model."""
        self.qrel: Optional[CQRel] = qrel
        """ Measures like mAP and NDCG require the number of relevant documents to be known, but not all may be in your X matrix. Providing a QREL file here ensures you have properly-normalized scores."""

    @abstractmethod
    def _get_train_request(self) -> TrainRequest:
        pass

    def fit_dataset(self, dataset: CDataset):
        """
        Train this model given a dataset object; discards any existing model weights.
        """
        self.model = dataset.train_model(self._get_train_request())

    def fit(self, X: np.ndarray, y: np.ndarray, qids: List[str]):
        """
        Constructs a dataset from `X`, `y`, and `qids`; then runs `fit_dataset` on that dataset.
        """
        self.fit_dataset(make_dataset(X, y, qids))

    def score_dataset(self, dataset: CDataset, measure: Optional[str] = None) -> float:
        """
        Computes the evaluation `measure` score for each query in the given dataset and returns the mean.
        """
        if self.model is None:
            raise ValueError("Cannot score before fit.")
        by_query = dataset.evaluate(self.model, measure or self.measure, self.qrel)
        if len(by_query) == 0:
            return 0.0
        return sum(by_query.values()) / len(by_query)

    def score(self, X: np.ndarray, y: np.ndarray, qids: List[str]) -> float:
        """
        Constructs a dataset from `X`, `y`, and `qids`; then scores the current `measure` for the current `model` against that dataset.
        """
        return self.score_dataset(make_dataset(X, y, qids))

    def predict_dataset(self, dataset: CDataset) -> np.ndarray:
        """
        Computes the ranking score for each instance in the given dataset, returning it as a numpy array.
        """
        if self.model is None:
            raise ValueError("Cannot score before fit.")
        return np.array(self.model.predict_dense_scores(dataset))

    def predict(self, X: np.ndarray, y: np.ndarray, qids: List[str]) -> np.ndarray:
        """
        Constructs a dataset from `X`, `y`, and `qids`; then computes the ranking score for each instance, returning it as a numpy array.
        """
        if self.model is None:
            raise ValueError("Cannot score before fit.")
        return self.predict_dataset(make_dataset(X, y, qids))


class CoordinateAscentRanker(BaseRanker):
    """
    A linear ranking model optimized using Coordinate Ascent, according to the given ranking `measure`.
    It will take time proportional to the number of features given.
    """

    def __init__(
        self,
        measure: str = "ndcg",
        params: CoordinateAscentParams = CoordinateAscentParams(),
    ):
        """ See `fastrank.training.CoordinateAscentParams` for configuration."""
        BaseRanker.__init__(self, measure=measure)  # type:ignore
        self.params = params

    def _get_train_request(self) -> TrainRequest:
        return TrainRequest(
            measure=self.measure, params=self.params, judgments=self.qrel
        )

    def weights(self) -> np.ndarray:
        """Inspect the linear weights learned."""
        if self.model is None:
            raise ValueError("Cannot inspect weights before fit.")
        return np.array(self.model.to_dict()["Linear"]["weights"])


class RandomForestRanker(BaseRanker):
    def __init__(
        self, measure: str = "ndcg", params: RandomForestParams = RandomForestParams()
    ):
        """ See `fastrank.training.RandomForestParams` for configuration."""
        BaseRanker.__init__(self, measure=measure)  # type:ignore
        self.params = params

    def _get_train_request(self) -> TrainRequest:
        return TrainRequest(
            measure=self.measure, params=self.params, judgments=self.qrel
        )  # type:ignore


class ImportedRanker(BaseRanker):
    """A class for wrapping an opaque `fastrank.clib.CModel` ranker."""

    def __init__(self, model: CModel, measure="ndcg"):
        BaseRanker.__init__(self, measure=measure)  # type:ignore
        self.model = model

    def _get_train_request(self) -> TrainRequest:
        raise ValueError("Can't figure out how to train.")

    @staticmethod
    def from_sklearn(
        m: Any,
        measure: str = "ndcg",
        expected_feature_dim: Optional[int] = None,
    ) -> "ImportedRanker":
        """
        Ingest a sklearn classifier or regressor as a ranker (if possible).
        Tested with:
          - SGDClassifier / SGDRegressor (other linear models should work)
          - RandomForestClassifier / RandomForestRegressor
          - ExtraTreesClassifier / ExtraTreesRegressor
          - GradientBoostingClassifier / etc.
        """
        if hasattr(m, "coef_"):
            weights = m.coef_.flatten()
            if expected_feature_dim:
                assert len(weights) == expected_feature_dim
            as_ranker = CModel.from_dict({"Linear": {"weights": weights.tolist()}})
            return ImportedRanker(as_ranker, measure=measure)
        if hasattr(m, "estimators_"):
            est = m.estimators_
            if isinstance(est, np.ndarray):
                est = est.tolist()
            models = [_sklearn_tree_to_json(t) for t in est]
            ensemble = {
                "Ensemble": {"models": models, "weights": [1.0 for _ in models]}
            }
            return ImportedRanker(CModel.from_dict(ensemble), measure=measure)
        raise ValueError("Not sure how to create a ImportedRanker from {}".format(m))


def _sklearn_tree_to_json(tree_model):
    """Recursively turn a Sci-Kit Learn Tree model into a python dictionary (which can be saved as JSON)"""
    if isinstance(tree_model, list):
        tree_model = tree_model[0]
    tree = tree_model.tree_

    from sklearn.tree import _tree

    def recurse(node, depth=0):
        """Recursively handle a given node."""
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            fid = int(tree.feature[node])
            threshold = float(tree.threshold[node])
            return {
                "FeatureSplit": {
                    "fid": fid,
                    "split": threshold,
                    "lhs": recurse(tree.children_left[node], depth + 1),
                    "rhs": recurse(tree.children_right[node], depth + 1),
                }
            }
        else:
            return {"LeafNode": tree.value[node][0].tolist()[0]}

    return {"DecisionTree": recurse(0)}