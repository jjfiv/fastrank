#%%
import cffi
import attr
import numpy as np
from cfastrank import lib, ffi
import ujson as json
import sklearn
import random
from sklearn.datasets import load_svmlight_file
from typing import Dict, List, Set
from collections import Counter


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

    def train_model(self, train_req: "TrainRequest") -> dict:
        self._require_init()
        train_req_str = json.dumps(train_req).encode("utf-8")
        train_resp = json.loads(
            _handle_rust_str(lib.train_model(train_req_str, self.pointer))
        )
        return train_resp

    def __query_json(self, message="num_features"):
        self._require_init()
        response = json.loads(
            _handle_rust_str(
                lib.query_dataset_json(self.pointer, message.encode("utf-8"))
            )
        )
        _maybe_raise_error_json(response)
        return response

    def num_features(self) -> int:
        return self.__query_json("num_features")

    def feature_ids(self) -> Set[int]:
        return set(self.__query_json("feature_ids"))

    def feature_names(self) -> Set[str]:
        return set(self.__query_json("feature_names"))

    def num_instances(self) -> int:
        return self.__query_json("num_instances")

    def queries(self) -> Set[str]:
        return set(self.__query_json("queries"))


@attr.s
class CoordinateAscentParams(object):
    num_restarts = attr.ib(type=int, default=5)
    num_max_iterations = attr.ib(type=int, default=25)
    step_base = attr.ib(type=float, default=0.05)
    step_scale = attr.ib(type=float, default=2.0)
    tolerance = attr.ib(type=float, default=0.001)
    seed = attr.ib(type=int, default=random.randint(0, 1 << 64))
    normalize = attr.ib(type=bool, default=True)
    quiet = attr.ib(type=bool, default=False)
    init_random = attr.ib(type=bool, default=True)
    output_ensemble = attr.ib(type=bool, default=False)


@attr.s
class QueryJudgments(object):
    docid_to_rel: Dict[str, float] = attr.ib(factory=dict)


@attr.s
class QuerySetJudgments(object):
    query_to_judgments: Dict[str, QueryJudgments] = attr.ib(factory=dict)


@attr.s
class TrainRequest(object):
    measure = attr.ib(type=str, default="ndcg")
    params = attr.ib(type=CoordinateAscentParams, factory=CoordinateAscentParams)
    judgments = attr.ib(type=QuerySetJudgments, default=None)


def query_json(message: str) -> str:
    """
    This method sends any old python object as input into the Rust exec_json call.
    The request is encoded on the way in.
    Returns a string.
    """
    command = message.encode("utf-8")
    response = json.loads(_handle_rust_str(lib.query_json(command)))
    _maybe_raise_error_json(response)
    return response


#%%
if __name__ == "__main__":
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
    rd = CDataset()
    rd.open_ranksvm("../examples/trec_news_2018.train")
    assert rd.queries() == _EXPECTED_QUERIES
    assert rd.feature_ids() == _EXPECTED_FEATURE_IDS
    assert rd.feature_names() == set(str(x) for x in _EXPECTED_FEATURE_IDS)
    assert rd.num_features() == _EXPECTED_D
    assert rd.num_instances() == _EXPECTED_N
    del rd
    rd = CDataset()
    rd.open_ranksvm(
        "../examples/trec_news_2018.train", "../examples/trec_news_2018.features.json"
    )
    assert rd.queries() == _EXPECTED_QUERIES
    assert rd.feature_ids() == _EXPECTED_FEATURE_IDS
    assert rd.feature_names() == _EXPECTED_FEATURE_NAMES
    assert rd.num_features() == _EXPECTED_D
    assert rd.num_instances() == _EXPECTED_N

    # Test out "from_numpy:"
    (train_X, train_y, train_qid) = load_svmlight_file(
        "../examples/trec_news_2018.train",
        dtype=np.float32,
        zero_based=False,
        query_id=True,
    )

    # Test subsample:
    _SUBSET = """378 363 811 321 807 347 646 397 802 804""".split()
    sample_rd = rd.subsample(_SUBSET)
    print(sample_rd.queries())
    assert sample_rd.queries() == set(_SUBSET)
    assert sample_rd.num_features() == _EXPECTED_D
    assert sample_rd.feature_ids() == _EXPECTED_FEATURE_IDS
    assert sample_rd.feature_names() == _EXPECTED_FEATURE_NAMES

    # calculate how many instances rust should've selected:
    count_by_qid = Counter(train_qid)
    expected_count = sum(count_by_qid[int(q)] for q in _SUBSET)
    assert sample_rd.num_instances() == expected_count

    print(TrainRequest())
    train_req = query_json("coordinate_ascent_defaults")
    ca_params = train_req["params"]["CoordinateAscent"]
    ca_params["init_random"] = True
    ca_params["seed"] = 42
    ca_params["quiet"] = True

    print(train_req)
    print(rd.train_model(train_req))

    # this loader supports zero-based!
    _EXPECTED_D -= 1
    _EXPECTED_FEATURE_IDS = set(range(_EXPECTED_D))

    train_X = train_X.todense()
    train = CDataset()
    train.from_numpy(train_X, train_y, train_qid)

    (train_N, train_D) = train_X.shape
    assert train.num_features() == train_D
    assert train.num_instances() == train_N
    assert train.queries() == _EXPECTED_QUERIES
    assert train.feature_ids() == _EXPECTED_FEATURE_IDS
    assert train.feature_names() == set(str(x) for x in _EXPECTED_FEATURE_IDS)
    assert train.num_features() == _EXPECTED_D
    assert train.num_instances() == _EXPECTED_N

    print(train.train_model(train_req))
