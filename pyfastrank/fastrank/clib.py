import cffi
import ujson as json
from cfastrank import lib, ffi
import numpy as np
from typing import Dict, Set, List


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
    def __init__(self, pointer=None):
        self.pointer = pointer
        self._queries = None

    def __del__(self):
        if self.pointer is not None:
            lib.free_cqrel(self.pointer)
            self.pointer = None

    def load_file(self, path: str):
        if self.pointer is not None:
            raise ValueError("Cannot re-load a CQRel object.")
        self.pointer = _handle_c_result(lib.load_cqrel(path.encode("utf-8")))

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

    def to_json(self) -> Dict[str, Dict[str, float]]:
        return self._query_json("to_json")

    def queries(self) -> Set[str]:
        if self._queries is None:
            self._queries = set(self._query_json("queries"))
        return self._queries

    def query_json(self, qid: str) -> Dict[str, float]:
        if qid in self.queries():
            return self._query_json(qid)
        raise ValueError("No qid={0} in cqrel: {1}".format(qid, self.queries()))


class CModel(object):
    def __init__(self, pointer, params):
        self.pointer = pointer
        self.params = params

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

    def to_json(self):
        return self._query_json("to_json")

    def __str__(self):
        return str(self.to_json())


class CDataset(object):
    def __init__(self, pointer=None):
        self.pointer = pointer
        # need to hold onto any numpy arrays...
        self.numpy_arrays_to_keep = []

    def __del__(self):
        if self.pointer is not None:
            lib.free_dataset(self.pointer)
            self.pointer = None
        self.numpy_arrays_to_keep = []

    def open_ranksvm(self, data_path, feature_names_path=None):
        if self.pointer is not None:
            raise ValueError("Cannot call open twice!")
        data_path = data_path.encode("utf-8")
        if feature_names_path is not None:
            feature_names_path = feature_names_path.encode("utf-8")
        else:
            feature_names_path = ffi.NULL
        self.pointer = _handle_c_result(
            lib.load_ranksvm_format(data_path, feature_names_path)
        )

    def from_numpy(self, X, y, qid):
        if self.pointer is not None:
            raise ValueError("Cannot update a CDataset object after init!")
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
        self.numpy_arrays_to_keep = [X, y, qid]
        # Pass pointers to these arrays to Rust!
        self.pointer = _handle_c_result(
            lib.make_dense_dataset_f32_f64_i64(
                N,
                D,
                ffi.cast("float *", X.ctypes.data),
                ffi.cast("double *", y.ctypes.data),
                ffi.cast("int64_t *", qid.ctypes.data),
            )
        )

    def _require_init(self):
        if self.pointer is None:
            raise ValueError("Forgot to call open_* or from_numpy on CDataset!")

    def subsample(self, queries: List[str]) -> "CDataset":
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

    def train_model(self, train_req: "TrainRequest") -> CModel:
        self._require_init()
        train_req_str = json.dumps(train_req).encode("utf-8")
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
        return self._query_json("num_features")

    def feature_ids(self) -> Set[int]:
        return set(self._query_json("feature_ids"))

    def feature_names(self) -> Set[str]:
        return set(self._query_json("feature_names"))

    def num_instances(self) -> int:
        return self._query_json("num_instances")

    def queries(self) -> Set[str]:
        return set(self._query_json("queries"))

    def evaluate(
        self, model: CModel, evaluator: str, qrel: CQRel = None
    ) -> Dict[str, float]:
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


def query_json(message: str):
    """
    This method sends any old python object as input into the Rust exec_json call.
    The request is encoded on the way in.
    Returns a JSON object decoded 'loads' to python.
    """
    command = message.encode("utf-8")
    response = json.loads(_handle_rust_str(lib.query_json(command)))
    _maybe_raise_error_json(response)
    return response
