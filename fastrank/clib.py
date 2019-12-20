import cffi
import json
from .fastrank import lib, ffi
import numpy as np
from typing import Dict, Set, List

# Keep in sync with fastrank/src/model.rs : fastrank::model::ModelEnum
_MODEL_TYPES = ["SingleFeature", "Linear", "DecisionTree", "Ensemble"]


def _handle_rust_str(result) -> str:
    """
    This method decodes bytes to UTF-8 and makes a new python string object. 
    It then frees the bytes that Rust allocated correctly.
    """
    try:
        txt = ffi.cast("char*", result)
        txt = ffi.string(txt).decode("utf-8")
        return txt
    finally:
        lib.free_str(result)


def _handle_c_result(c_result):
    """
    This handles the logical-OR struct of the CDataset { error_message, success } 
    where both the wrapper and the error_message will be freed by the end of this function.

    The success pointer is returned or an error is raised!
    """
    if c_result == ffi.NULL:
        raise ValueError("CResult should not be NULL")
    error = None
    success = None
    if c_result.error_message != ffi.NULL:
        error = _handle_rust_str(c_result.error_message)
    if c_result.success != ffi.NULL:
        success = c_result.success
    lib.free_c_result(c_result)
    _maybe_raise_error_str(error)
    return success


def _maybe_raise_error_str(rust_error_string):
    if rust_error_string is None:
        return
    if "{" in rust_error_string:
        response = json.loads(rust_error_string)
        if "error" in response and "context" in response:
            raise Exception("{0}: {1}".format(response["error"], response["context"]))
    else:
        raise Exception(rust_error_string)


def _maybe_raise_error_json(response):
    if response is None:
        return
    if isinstance(response, dict):
        if "error" in response and "context" in response:
            raise Exception("{0}: {1}".format(response["error"], response["context"]))
    return


class CQRel(object):
    """
    This class represents a loaded set of TREC relevance judgments. 
    
    This is important because some measures supported by fastrank require the total number of relevance judgments (e.g., MAP) or the maximum ideal gain (e.g., NDCG).

    Use :func:`~load_file` or :func:`~from_dict` to create one of these.
    """

    def __init__(self, pointer=None):
        """
        This constructor is essentially private; it expects a pointer from a CFFI call.
        """
        self.pointer = pointer
        self._queries = None

    def __del__(self):
        if self.pointer is not None:
            lib.free_cqrel(self.pointer)
            self.pointer = None

    @staticmethod
    def load_file(path: str) -> "CQRel":
        """Given a path to a TREC judgments file, load it into memory."""
        return CQRel(_handle_c_result(lib.load_cqrel(path.encode("utf-8"))))

    @staticmethod
    def from_dict(dictionaries: Dict[str, Dict[str, float]]) -> "CQRel":
        """Given a mapping of (qid -> (doc -> judgment)) pass it over to Rust."""
        input_str = json.dumps(dictionaries).encode("utf-8")
        return CQRel(_handle_c_result(lib.cqrel_from_json(input_str)))

    def _require_init(self):
        if self.pointer is None:
            raise ValueError("CQRel is null!")

    def _query_json(self, message="queries"):
        self._require_init()
        response = json.loads(
            _handle_rust_str(
                lib.cqrel_query_json(self.pointer, message.encode("utf-8"))
            )
        )
        _maybe_raise_error_json(response)
        return response

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert this object to a mapping of (qid -> (doc -> judgment)) for use in Python."""
        return self._query_json("to_json")

    def queries(self) -> Set[str]:
        """Get a list of judged queries from this object."""
        if self._queries is None:
            self._queries = set(self._query_json("queries"))
        return self._queries

    def query_judgments(self, qid: str) -> Dict[str, float]:
        """Given a query identifier, return the mapping of judgments available for it in this CQRel."""
        if qid in self.queries():
            return self._query_json(qid)
        raise ValueError("No qid={0} in cqrel: {1}".format(qid, self.queries()))


class CModel(object):
    """
    Usually you're going to get this from:

    - training a new model on a dataset: :func:`~fastrank.clib.CDataset.train_model`.
    - loading a saved model from a file, using :func:`~from_dict`.
    """

    def __init__(self, pointer, params=None):
        self.pointer = pointer
        self.params = params

    @staticmethod
    def _check_model_json(model_json: Dict):
        [single_key] = list(model_json.keys())
        assert single_key in _MODEL_TYPES
        # TODO: deeper checks

    @staticmethod
    def from_dict(model_json: Dict) -> "CModel":
        """Create a model from a python representation.
        
        >>> model = CModel.from_dict(json.load("saved_model.json"))
        """
        CModel._check_model_json(model_json)
        json_str = json.dumps(model_json).encode("utf-8")
        return CModel(_handle_c_result(lib.model_from_json(json_str)))

    def __del__(self):
        if self.pointer is not None:
            lib.free_model(self.pointer)
            self.pointer = None

    def _require_init(self):
        if self.pointer is None:
            raise ValueError("CModel is null!")

    def _query_json(self, message="to_json"):
        self._require_init()
        response = json.loads(
            _handle_rust_str(
                lib.model_query_json(self.pointer, message.encode("utf-8"))
            )
        )
        _maybe_raise_error_json(response)
        return response

    def to_dict(self):
        """Turn the opaque Rust model pointer into inspectable JSON structure. This ties nicely to :func:`~from_dict`.
        
        >>> model_copy = CModel.from_dict(model.to_dict())

        After which both ``model_copy`` and ``model`` will have equivalent models.
        """
        return self._query_json("to_json")

    def __str__(self):
        return str(self.to_dict())


class CDataset(object):
    """
    This class abstracts access to a rust-owned dataset.

    Construct one of these with either:

     - :func:`~open_ranksvm` a file in ranksvm/ranklib/libsvm/svmlight format.
     - :func:`~from_numpy` with pre-loaded/pre-created numpy arrays.
    """

    def __init__(self, pointer=None):
        self.pointer = pointer
        # need to hold onto any numpy arrays...
        self.numpy_arrays_to_keep = []

    def __del__(self):
        if self.pointer is not None:
            lib.free_dataset(self.pointer)
            self.pointer = None
        self.numpy_arrays_to_keep = []

    @staticmethod
    def open_ranksvm(data_path, feature_names_path=None) -> "CDataset":
        """
        Construct a dataset with optional feature names. Supports gzip, bzip2 and zstd compression.

        :param data_path: The path to your input file.
        :type data_path: str
        :param feature_names_path: The path to a JSON file of feature names (optional).
        :type feature_names_path: str

        >>> dataset = CDataset.open_ranksvm("examples/trec_news_2018.train", "examples/trec_news_2018.features.json")
        """
        data_path = data_path.encode("utf-8")
        if feature_names_path is not None:
            feature_names_path = feature_names_path.encode("utf-8")
        else:
            feature_names_path = ffi.NULL
        return CDataset(
            _handle_c_result(lib.load_ranksvm_format(data_path, feature_names_path))
        )

    @staticmethod
    def from_numpy(X, y, qid) -> "CDataset":
        """
        Construct a dataset from in-memory numpy arrays. This class will hold on to them so that Rust points at them. If you delete them, bad things will happen.

        :param X: The feature matrix, a (NxD) float32 matrix; N instances, D features. Must be dense for now.
        :type X: numpy.array
        :param y: The judgment vector, a 1xN or Nx1 float64 matrix.
        :type y: numpy.array
        :param qid: The numeric representations of query ids (not possible for all datasets, I know.) 1xN or Nx1 int64 matrix.
        :type qid: numpy.array

        We can then construct our own numpy arrays, or use the sklearn loader:

        >>> from sklearn.datasets import load_svmlight_file
        >>> (X, y, qid) = load_svmlight_file("../examples/trec_news_2018.train", dtype=np.float32, zero_based=False, query_id=True)
        >>> X = X.todense()
        >>> dataset = CDataset.from_numpy(X, y, qid)
        """
        (N, D) = X.shape
        assert N > 0
        assert D > 0
        assert len(y) == N
        assert len(qid) == N
        # TODO: be more flexible here!
        assert X.dtype == "float32"
        assert y.dtype == "float64"
        assert qid.dtype == "int64"
        # Since Rust just has a pointer to them, have python keep them!
        numpy_arrays_to_keep = [X, y, qid]
        # Pass pointers to these arrays to Rust!
        dataset = CDataset(
            _handle_c_result(
                lib.make_dense_dataset_f32_f64_i64(
                    N,
                    D,
                    ffi.cast("float *", X.ctypes.data),
                    ffi.cast("double *", y.ctypes.data),
                    ffi.cast("int64_t *", qid.ctypes.data),
                )
            )
        )
        dataset.numpy_arrays_to_keep = numpy_arrays_to_keep
        return dataset

    def _require_init(self):
        if self.pointer is None:
            raise ValueError("Forgot to call open_* or from_numpy on CDataset!")

    def subsample_queries(self, queries: List[str]) -> "CDataset":
        """
        Construct a subset of this dataset from the given query ids. 
        This can be used to implement train/test splits or cross-validation.

        >>> train = dataset.subsample_queries(["001", "002"])
        >>> test = dataset.subsample_queries(["003"])
        """
        self._require_init()
        actual_queries = self.queries()
        for q in queries:
            if q not in actual_queries:
                raise ValueError(
                    "Asked for query that does not exist in subsample: {0} not in {1}".format(
                        q, actual_queries
                    )
                )
        child = CDataset()
        # keep those alive if need-be in case they lose the parent!
        child.numpy_arrays_to_keep = self.numpy_arrays_to_keep
        request = json.dumps(queries).encode("utf-8")
        child.pointer = _handle_c_result(
            lib.dataset_query_sampling(self.pointer, request)
        )
        return child

    def subsample_feature_names(self, features: List[str]) -> "CDataset":
        """
        Construct a subset of this dataset from the given features. 
        This can be used to experiment with feature subsets and do ablation studies.

        >>> features = dataset.feature_names()
        >>> assert("pagerank" in features)
        >>> features.remove("pagerank")
        >>> no_pagerank = train.subsample_feature_names(list(features))
        """
        name_to_id = dict(
            zip(self._query_json("feature_names"), self._query_json("feature_ids"))
        )
        fnums = sorted(set(name_to_id[f] for f in features))
        fnums_str = json.dumps(fnums).encode("utf-8")
        child = CDataset(
            _handle_c_result(lib.dataset_feature_sampling(self.pointer, fnums_str))
        )
        child.numpy_arrays_to_keep = self.numpy_arrays_to_keep
        return child

    def train_model(self, train_req: "TrainRequest") -> CModel:
        """
        Train a Model on this Dataset.
        
        TODO: make this accept a :class:`~fastrank.TrainRequest` object.
        """
        self._require_init()
        train_req_str = json.dumps(train_req.to_dict()).encode("utf-8")
        train_resp = _handle_c_result(lib.train_model(train_req_str, self.pointer))
        return CModel(train_resp, train_req)

    def _query_json(self, message="num_features"):
        self._require_init()
        response = json.loads(
            _handle_rust_str(
                lib.dataset_query_json(self.pointer, message.encode("utf-8"))
            )
        )
        _maybe_raise_error_json(response)
        return response

    def num_features(self) -> int:
        """Return the number of features available in this dataset."""
        return self._query_json("num_features")

    def feature_ids(self) -> Set[int]:
        """Return a set of feature ids available in this dataset."""
        return set(self._query_json("feature_ids"))

    def feature_names(self) -> Set[str]:
        """Return a set of feature names available in this dataset."""
        return set(self._query_json("feature_names"))

    def feature_index_to_name(self) -> Dict[int, str]:
        """Returns a mapping of feature ids to feature names present in this dataset."""
        return dict(
            zip(self._query_json("feature_ids"), self._query_json("feature_names"))
        )

    def feature_name_to_index(self) -> Dict[str, int]:
        """Returns a mapping of feature names to feature ids present in this dataset."""
        return dict(
            zip(self._query_json("feature_names"), self._query_json("feature_ids"))
        )

    def num_instances(self) -> int:
        """Returns the number of instances present in this dataset."""
        return self._query_json("num_instances")

    def queries(self) -> Set[str]:
        """Collect the set of queries present in this dataset."""
        return set(self._query_json("queries"))

    def instances_by_query(self) -> Dict[str, List[int]]:
        """Collect a list of instance ids by their query."""
        return self._query_json("instances_by_query")

    def evaluate(
        self, model: CModel, evaluator: str, qrel: CQRel = None
    ) -> Dict[str, float]:
        """
        Evaluate a model across this dataset using the given evaluator, optionally with judgments passed in.

        :param model: The model to evaluate.
        :type model: CModel
        :param evaluator: The evaluator to use. Supports "ndcg", "ndcg@5", etc.
        :type evaluator: str
        :param qrel: The judgments, if any.
        :type qrel: CQRel
        :return: A mapping from query ids to evaluator scores.
        :rtype: Dict[str, float]
        """
        self._require_init()
        model._require_init()
        qrel_pointer = ffi.NULL
        if qrel is not None:
            qrel._require_init()
            qrel_pointer = qrel.pointer

        response = json.loads(
            _handle_rust_str(
                lib.evaluate_by_query(
                    model.pointer, self.pointer, qrel_pointer, evaluator.encode("utf-8")
                )
            )
        )
        _maybe_raise_error_json(response)
        return response

    def predict_trecrun(
        self,
        model: CModel,
        output_path: str,
        system_name: str = "fastrank",
        quiet=True,
        depth=0,
    ) -> int:
        """
        Save output of model on this dataset to output_path with name system_name. 
        
        :param model: Get results from this model.
        :type model: CModel
        :param output_path: Save results in TREC Run format to this file.
        :type output_path: str
        :param system_name: What you want to call your system in the final column; or else "fastrank".
        :type system_name: str
        :param quiet: Don't print success if ``quiet`` is True.
        :type quiet: bool
        :param depth: Only keep the best ``depth`` results per query unless ``depth`` is zero.
        :type depth: int
        :return: The number of records written.
        :rtype: int
        """
        self._require_init()
        model._require_init()
        output_path = output_path.encode("utf-8")
        system_name = system_name.encode("utf-8")
        response = json.loads(
            _handle_rust_str(
                lib.predict_to_trecrun(
                    model.pointer, self.pointer, output_path, system_name, depth
                )
            )
        )
        _maybe_raise_error_json(response)
        if not quiet:
            print(
                "Wrote {} records to {} as {}.".format(
                    response, output_path, system_name
                )
            )
        return response


def query_json(message: str):
    """
    This method sends any old python object as input into the Rust exec_json call.
    The request is encoded on the way in.

    Returns a JSON object decoded 'loads' to python. 
    
    Consider this private if you can.
    """
    command = message.encode("utf-8")
    response = json.loads(_handle_rust_str(lib.query_json(command)))
    _maybe_raise_error_json(response)
    return response
