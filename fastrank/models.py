from .training import CoordinateAscentParams, RandomForestParams, TrainRequest
from .clib import CModel, CDataset, CQRel
from typing import List, Optional, Dict, Any
import numpy as np

def make_dataset(X: np.ndarray, y: np.ndarray, qids: List[str]) -> CDataset:
    (N, D) = X.shape
    assert y.shape == (N, 1) or y.shape == (N, )
    q_numbers = np.zeros((N, ), dtype=np.int64)
    q_trans: Dict[str, int] = {}
    for i, qid in enumerate(qids):
        if qid not in q_trans:
            q_trans[qid] = len(q_trans)
        q_numbers[i] = q_trans[qid]
    return CDataset.from_numpy(X, y, q_numbers)


class CoordinateAscentRanker:
    def __init__(self, measure: str = 'ndcg', params: CoordinateAscentParams = CoordinateAscentParams()):
        self.params = params
        self.measure = measure
        self.model: Optional[CModel] = None
        self.qrel: Optional[CQRel] = None

    def fit(self, X: np.ndarray, y: np.ndarray, qids: List[str]):
        dataset = make_dataset(X, y, qids)
        req = TrainRequest(measure=self.measure, params=self.params, judgments=self.qrel)
        self.model = dataset.train_model(req)

    def score(self, X: np.ndarray, y: np.ndarray, qids: List[str]) -> float:
        if self.model is None:
            raise ValueError("Cannot score before fit.")
        dataset = make_dataset(X, y, qids)
        by_query = dataset.evaluate(self.model, self.measure, self.qrel)
        if len(by_query) == 0:
            return 0.0
        return sum(by_query.values()) / len(by_query)

    def predict(self, X: np.ndarray, y: np.ndarray, qids: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Cannot score before fit.")
        dataset = make_dataset(X, y, qids)
        return np.array(self.model.predict_dense_scores(dataset))
    
    def weights(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Cannot inspect weights before fit.")
        return np.array(self.model.to_dict()['Linear']['weights'])



class RandomForestRanker:
    def __init__(self, measure: str = 'ndcg', params: RandomForestParams = RandomForestParams()):
        self.params = params
        self.measure = measure
        self.model: Optional[CModel] = None
        self.qrel: Optional[CQRel] = None


    def fit(self, X: np.ndarray, y: np.ndarray, qids: List[str]):
        dataset = make_dataset(X, y, qids)
        req = TrainRequest(measure=self.measure, params=self.params, judgments=self.qrel)
        self.model = dataset.train_model(req)

    def score(self, X: np.ndarray, y: np.ndarray, qids: List[str]) -> float:
        if self.model is None:
            raise ValueError("Cannot score before fit.")
        dataset = make_dataset(X, y, qids)
        by_query = dataset.evaluate(self.model, self.measure, self.qrel)
        if len(by_query) == 0:
            return 0.0
        return sum(by_query.values()) / len(by_query)

    def predict(self, X: np.ndarray, y: np.ndarray, qids: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Cannot score before fit.")
        dataset = make_dataset(X, y, qids)
        return np.array(self.model.predict_dense_scores(dataset))

    @staticmethod
    def from_sklearn(m: Any, measure: str = 'ndcg', params: RandomForestParams = RandomForestParams()) -> 'RandomForestRanker':

        forest = RandomForestRanker(measure, params)
        return forest

class ImportedRanker:
    def __init__(self, model: CModel):
        self.model = model

    def score(self, X: np.ndarray, y: np.ndarray, qids: List[str], measure: str, qrel: Optional[CQRel] = None) -> float:
        dataset = make_dataset(X, y, qids)
        by_query = dataset.evaluate(self.model, measure, qrel)
        if len(by_query) == 0:
            return 0.0
        return sum(by_query.values()) / len(by_query)

    def predict(self, X: np.ndarray, y: np.ndarray, qids: List[str]) -> np.ndarray:
        dataset = make_dataset(X, y, qids)
        return np.array(self.model.predict_dense_scores(dataset))
    
    @staticmethod
    def from_sklearn(m: Any, expected_feature_dim: Optional[int] = None) -> 'ImportedRanker':
        """
        Ingest a sklearn classifier or regressor as a ranker (if possible).
        Tested with:
          - SGDClassifier / SGDRegressor (other linear models should work)
          - RandomForestClassifier / RandomForestRegressor
          - ExtraTreesClassifier / ExtraTreesRegressor
          - GradientBoostingClassifier / etc.
        """
        if hasattr(m, 'coef_'):
            weights = m.coef_.flatten()
            if expected_feature_dim:
                assert(len(weights) == expected_feature_dim)
            as_ranker = CModel.from_dict({"Linear": {"weights": weights.tolist()}})
            return ImportedRanker(as_ranker)
        if hasattr(m, 'estimators_'):
            est = m.estimators_
            if isinstance(est, np.ndarray):
                est = est.tolist()
            models = [sklearn_tree_to_json(t) for t in est]
            ensemble = {"Ensemble": {
                "models": models,
                "weights": [1.0 for _ in models]
            }}
            return ImportedRanker(CModel.from_dict(ensemble))
        raise ValueError("Not sure how to create a ImportedRanker from {}".format(m))
        

def sklearn_tree_to_json(tree_model):
    """Recursively turn a SKLearn Tree model into a python dictionary (which can be saved as JSON)"""
    if isinstance(tree_model, list):
        tree_model = tree_model[0]
    tree = tree_model.tree_

    from sklearn.tree import _tree

    def recurse(node, depth=0):
        """Recursively handle a given node."""
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            fid = int(tree.feature[node])
            threshold = float(tree.threshold[node])
            return {"FeatureSplit": {
                "fid": fid,
                "split": threshold,
                "lhs": recurse(tree.children_left[node], depth+1),
                "rhs": recurse(tree.children_right[node], depth+1),
            }}
        else:
            return {"LeafNode": tree.value[node][0].tolist()[0]}

    return {"DecisionTree": recurse(0)}