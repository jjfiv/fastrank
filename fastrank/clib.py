import json
from .fastrank import lib, ffi
from typing import Dict, Set, List, Any, Optional, Union
import numpy as np

# Keep in sync with fastrank/src/model.rs : fastrank::model::ModelEnum
_MODEL_TYPES = ["SingleFeature", "Linear", "DecisionTree", "Ensemble"]


def _handle_rust_str(result) -> Optional[str]:
    """
    This method decodes bytes to UTF-8 and makes a new python string object.
    It then frees the bytes that Rust allocated correctly.
    """
    if result == ffi.NULL:
        return None
    try:
        txt = ffi.cast("char*", result)
        txt = ffi.string(txt).decode("utf-8")
        return txt
    finally:
        lib.free_str(result)


def _handle_rust_json(result) -> Any:
    str_response = _handle_rust_str(result)
    if str_response is None:
        raise ValueError("Internal Error; expected JSON, got NULL")
    return json.loads(str_response)


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


class CQRel:
    """
    This class represents a loaded set of TREC relevance judgments.

    This is important because some measures supported by fastrank require the total number of relevance judgments (e.g., MAP) or the maximum ideal gain (e.g., NDCG).

    Use `load_file` or `from_dict` to create one of these.
    """

    def __init__(self, pointer=None):
        """
        This constructor is essentially private; it expects a pointer from a CFFI call.
        """
        self.pointer = None
        assert pointer != ffi.NULL
        self.pointer = pointer
        self._queries = None

    def __del__(self):
        if self.pointer is not None:
            lib.free_cqrel(self.pointer)
            self.pointer = None

    @staticmethod
    def load_file(path: str) -> "CQRel":
        """Given a path to a TREC judgments file, load it into memory."""
        with ErrorCode() as err:
            return CQRel(lib.load_cqrel(path.encode("utf-8"), err))

    @staticmethod
    def from_dict(dictionaries: Dict[str, Dict[str, float]]) -> "CQRel":
        """Given a mapping of (qid -> (doc -> judgment)) pass it over to Rust."""
        input_str = json.dumps(dictionaries).encode("utf-8")
        with ErrorCode() as err:
            return CQRel(lib.cqrel_from_json(input_str, err))

    def _require_init(self):
        if self.pointer is None:
            raise ValueError("CQRel is null!")

    def _query_json(self, message="queries"):
        self._require_init()
        with ErrorCode() as err:
            return _handle_rust_json(
                lib.cqrel_query_json(self.pointer, message.encode("utf-8"), err)
            )

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


class CModel:
    """
    Usually you're going to get this from:

    - training a new model on a dataset: `CDataset.train_model`.
    - loading a saved model from a file, using `from_dict`.
    """

    def __init__(self, pointer, params=None):
        self.pointer = None
        assert pointer != ffi.NULL
        self.pointer = pointer
        self.params = params

    @staticmethod
    def _check_model_json(model_json: Dict):
        [single_key] = list(model_json.keys())
        assert single_key in _MODEL_TYPES
        # TODO: deeper checks

    @staticmethod
    def from_dict(model_json: Dict[str, Any]) -> "CModel":
        """Create a model from a python representation.

        >>> model = CModel.from_dict(json.load("saved_model.json"))
        """
        CModel._check_model_json(model_json)
        json_str = json.dumps(model_json).encode("utf-8")
        with ErrorCode() as err:
            return CModel(lib.model_from_json(json_str, err))

    def predict_dense_scores(
        self, dataset: "CDataset", missing: float = float("nan")
    ) -> List[float]:
        """
        Use the model to predict scores for each element of the given dataset.
        Returns a dense list of scores where the indexes should be aligned with your input.
        Substitutes ``missing`` (default=NaN) for any indices not in this dataset.
        """
        output: List[float] = []
        dict_scores = self.predict_scores(dataset)
        for (index, score) in sorted(dict_scores.items()):
            while len(output) < index:
                output.append(missing)
            if index == len(output):
                output.append(score)
            else:
                output[index] = score
        return output

    def predict_scores(self, dataset: "CDataset") -> Dict[int, float]:
        """
        Use the model to predict scores for each element of the given dataset.
        Returns a dictionary of instance-index to score.
        """
        response = _handle_rust_json(lib.predict_scores(self.pointer, dataset.pointer))
        _maybe_raise_error_json(response)
        return dict((int(k), v) for k, v in response.items())

    def __del__(self):
        if self.pointer is not None:
            lib.free_model(self.pointer)
            self.pointer = None

    def _require_init(self):
        if self.pointer is None:
            raise ValueError("CModel is null!")

    def _query_json(self, message="to_json"):
        self._require_init()
        response = _handle_rust_json(
            lib.model_query_json(self.pointer, message.encode("utf-8"))
        )
        _maybe_raise_error_json(response)
        return response

    def to_dict(self):
        """Turn the opaque Rust model pointer into inspectable JSON structure. This ties nicely to `from_dict`.

        >>> model_copy = CModel.from_dict(model.to_dict())

        After which both ``model_copy`` and ``model`` will have equivalent models.
        """
        return self._query_json("to_json")

    def __str__(self):
        return str(self.to_dict())


class CDataset:
    """
    This class abstracts access to a rust-owned dataset.

    Construct one of these with either:

     - `open_ranksvm` a file in ranksvm/ranklib/libsvm/svmlight format.
     - `from_numpy` with pre-loaded/pre-created numpy arrays.
    """

    def __init__(self, pointer=None):
        assert pointer != ffi.NULL
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

        - ``data_path``: The path to your input file.
        - ``feature_names_path``: The path to a JSON file of feature names (optional).

        >>> dataset = CDataset.open_ranksvm("examples/trec_news_2018.train", "examples/trec_news_2018.features.json")
        """
        data_path = data_path.encode("utf-8")
        if feature_names_path is not None:
            feature_names_path = feature_names_path.encode("utf-8")
        else:
            feature_names_path = ffi.NULL
        with ErrorCode() as err:
            return CDataset(lib.load_ranksvm_format(data_path, feature_names_path, err))

    @staticmethod
    def from_numpy(
        X: np.ndarray, y: np.ndarray, qids: Union[List[str], np.ndarray]
    ) -> "CDataset":
        """
        Construct a dataset from in-memory numpy arrays. This class will hold on to them so that Rust points at them. If you delete them, bad things will happen.

        - ``X``: The feature matrix, a (NxD) float3matrix; N instances, D features. Must be dense for now.
        - ``y``: The judgment vector, a 1xN or Nx1 float/int matrix.
        - ``qids``: The numeric or string representations of query ids 1xN or Nx1 int/string matrix.

        We can then construct our own numpy arrays, or use the sklearn loader:

        >>> from sklearn.datasets import load_svmlight_file
        >>> (X, y, qid) = load_svmlight_file("../examples/trec_news_2018.train", zero_based=False, query_id=True)
        >>> X = X.todense()
        >>> dataset = CDataset.from_numpy(X, y, qid)
        """
        (N, D) = X.shape
        assert N > 0
        assert D > 0
        assert len(y) == N
        assert len(qids) == N
        float_dtypes = ["float32", "float64"]
        int_dtypes = ["int32", "int64"]
        # TODO: be more flexible here!
        assert X.dtype in float_dtypes
        assert (y.dtype in float_dtypes) or (y.dtype in int_dtypes)
        # Since Rust just has a pointer to them, have python keep them!
        numpy_arrays_to_keep = [X, y]

        qid_strs: Optional[Dict[int, str]] = None
        qid_nums = None
        if isinstance(qids, list):
            qid_nums = np.zeros(N, dtype="int32")
            qid_to_index: Dict[str, int] = {}
            for i, qid in enumerate(qids):
                if qid in qid_to_index:
                    qid_nums[i] = qid_to_index[qid]
                else:
                    num = len(qid_to_index)
                    qid_to_index[qid] = num
                    qid_nums[i] = num
            # reverse:
            qid_strs = dict((v, k) for (k, v) in qid_to_index.items())
        elif isinstance(qids, np.ndarray):
            assert qids.dtype in int_dtypes
            qid_nums = qids
        else:
            raise ValueError("Unsupported type for qids: {}".format(type(qids)))

        numpy_arrays_to_keep.append(qid_nums)

        qid_str_json = json.dumps(qid_strs).encode("utf-8") if qid_strs else ffi.NULL

        # Pass pointers to these arrays to Rust!
        dataset = CDataset(
            _handle_c_result(
                lib.make_dense_dataset_v2(
                    N,
                    D,
                    ffi.cast("void *", X.ctypes.data),
                    str(X.dtype).encode("utf-8"),
                    ffi.cast("void *", y.ctypes.data),
                    str(y.dtype).encode("utf-8"),
                    ffi.cast("void *", qid_nums.ctypes.data),
                    str(qid_nums.dtype).encode("utf-8"),
                    qid_str_json,
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
        request = json.dumps(queries).encode("utf-8")

        with ErrorCode() as err:
            child = CDataset(lib.dataset_query_sampling(self.pointer, request, err))
        # keep those alive if need-be in case they lose the parent!
        child.numpy_arrays_to_keep = self.numpy_arrays_to_keep
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

    def train_model(
        self,
        train_req: "fastrank.training.TrainRequest",  # type:ignore
    ) -> CModel:
        """
        Train a Model on this Dataset.
        """
        self._require_init()
        train_req_str = json.dumps(train_req.to_dict()).encode("utf-8")
        with ErrorCode() as err:
            train_resp = lib.train_model(train_req_str, self.pointer, err)
            return CModel(train_resp, train_req)

    def _query_json(self, message="num_features"):
        self._require_init()
        response = _handle_rust_json(
            lib.dataset_query_json(self.pointer, message.encode("utf-8"))
        )
        _maybe_raise_error_json(response)
        return response

    def is_sampled(self) -> bool:
        """Returns true if this dataset has been sampled (instances or features)."""
        return self._query_json("is_sampled")

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

        - ``model``: The model to evaluate.
        - ``evaluator``: The evaluator to use. Supports "ndcg", "ndcg@5", etc.
        - ``qrel``: The judgments, if any.

        ***returns*** A mapping from query ids to evaluator scores.
        """
        self._require_init()
        model._require_init()
        qrel_pointer = ffi.NULL
        if qrel is not None:
            qrel._require_init()
            qrel_pointer = qrel.pointer

        str_response = _handle_rust_str(
            lib.evaluate_by_query(
                model.pointer, self.pointer, qrel_pointer, evaluator.encode("utf-8")
            )
        )
        assert str_response is not None
        response = json.loads(str_response)
        _maybe_raise_error_json(response)
        return response

    def predict_scores(self, model: CModel) -> Dict[int, float]:
        """
        Get a score for each instance in your dataset.
        Returns sparse results; intended if you have sampled your dataset in any way.
        """
        return model.predict_scores(self)

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

        - ``model``: Get results from this model.
        - ``output_path``: Save results in TREC Run format to this file.
        - ``system_name``: What you want to call your system in the final column; or else "fastrank".
        - ``quiet``: Don't print success if ``quiet`` is True.
        - ``depth``: Only keep the best ``depth`` results per query unless ``depth`` is zero.

        ***returns*** The number of records written.
        """
        self._require_init()
        model._require_init()
        str_response = _handle_rust_str(
            lib.predict_to_trecrun(
                model.pointer,
                self.pointer,
                output_path.encode("utf-8"),
                system_name.encode("utf-8"),
                depth,
            )
        )
        assert str_response is not None
        response = json.loads(str_response)
        _maybe_raise_error_json(response)
        if not quiet:
            print(
                "Wrote {} records to {} as {}.".format(
                    response, output_path, system_name
                )
            )
        return response


class ErrorCode:
    def __init__(self):
        self.error = ffi.new("intptr_t *")

    def ptr(self):
        return self.error

    def __enter__(self):
        self.error[0] = 0
        return self.error

    def __exit__(self, ex_type, ex_value, ex_tb):
        if self.error[0] > 0:
            raise ValueError(_handle_rust_str(lib.fetch_err(self.error[0])))


def query_json(message: str):
    """
    This method sends any old python object as input into the Rust exec_json call.
    The request is encoded on the way in.

    Returns a JSON object decoded 'loads' to python.

    Consider this private if you can.
    """
    command = message.encode("utf-8")
    with ErrorCode() as ptr:
        resp = lib.query_json(command, ptr)
    response = _handle_rust_json(resp)
    return response


def evaluate_query(
    measure: str,
    gains: Union[List[int], List[float]],
    scores: List[float],
    depth: Optional[int] = None,
    opts: Dict[str, Any] = {},
) -> float:
    n = len(gains)
    assert len(scores) == n
    encoded_depth = -1
    if depth is not None:
        assert depth > 0
        encoded_depth = depth
    gains_arr = np.array(gains, dtype="float32")
    scores_arr = np.array(scores, dtype="float64")
    measure_c = measure.encode("utf-8")
    opts_c = json.dumps(opts).encode("utf-8")
    float_ptr = ffi.cast(
        "double*",
        _handle_c_result(
            lib.evaluate_query(
                measure_c,
                n,
                ffi.cast("float*", gains_arr.ctypes.data),
                ffi.cast("double*", scores_arr.ctypes.data),
                encoded_depth,
                opts_c,
            )
        ),
    )
    number = float_ptr[0]
    lib.free_f64(float_ptr)
    return number
