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