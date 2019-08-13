#%%
import cffi
import attr
import numpy as np
from cfastrank import lib, ffi
import ujson as json
import sklearn
import random
from sklearn.datasets import load_svmlight_file
from typing import Dict, List


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

    def __del__(self):
        if self.pointer is not None:
            lib.free_dataset(self.pointer)
            self.pointer = None

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
        print(self.pointer)

    def __query_json(self, message="num_features"):
        if self.pointer is None:
            raise ValueError("Forgot to call open_* on CDataset!")
        response = json.loads(
            _handle_rust_str(
                lib.query_dataset_json(self.pointer, message.encode("utf-8"))
            )
        )
        _maybe_raise_error_json(response)
        return response

    def num_features(self):
        return self.__query_json("num_features")

    def feature_ids(self):
        return self.__query_json("feature_ids")

    def feature_names(self):
        return self.__query_json("feature_names")

    def num_instances(self):
        return self.__query_json("num_instances")

    def queries(self):
        return self.__query_json("queries")


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
    rd = CDataset()
    rd.open_ranksvm("../examples/trec_news_2018.train")
    print(
        "num_features: {0}, num_instances: {1}, queries: {2} features: {3}".format(
            rd.num_features(), rd.num_instances(), rd.queries(), rd.feature_names()
        )
    )
    rd2 = CDataset()
    rd2.open_ranksvm(
        "../examples/trec_news_2018.train", "../examples/trec_news_2018.features.json"
    )
    print(
        "num_features: {0}, num_instances: {1}, queries: {2} features: {3}".format(
            rd.num_features(), rd.num_instances(), rd.queries(), rd.feature_names()
        )
    )

    print(TrainRequest())

    train_req = query_json("coordinate_ascent_defaults")
    ca_params = train_req["params"]["CoordinateAscent"]
    ca_params["init_random"] = True
    ca_params["seed"] = 42
    ca_params["quiet"] = True
    print(train_req)

    (train_X, train_y, train_qid) = load_svmlight_file(
        "../examples/trec_news_2018.train",
        dtype=np.float32,
        zero_based=False,
        query_id=True,
    )
    train_X = train_X.todense()
    (train_N, train_D) = train_X.shape
    print(train_N, train_D)
    train_req_str = json.dumps(train_req).encode("utf-8")
    train_resp = json.loads(
        _handle_rust_str(
            lib.train_dense_dataset_f32_f64_i64(
                train_req_str,
                train_N,
                train_D,
                ffi.cast("float *", train_X.ctypes.data),
                ffi.cast("double *", train_y.ctypes.data),
                ffi.cast("int64_t *", train_qid.ctypes.data),
            )
        )
    )
    print(train_resp)
