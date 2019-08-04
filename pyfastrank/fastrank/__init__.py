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


@attr.s
class CoordinateAscentParams(object):
    num_restarts = attr.ib(type=int, default=5)
    num_max_iterations = attr.ib(type=int, default=25)
    step_base = attr.ib(type=float, default=0.05)
    step_scale = attr.ib(type=float, default=2.0)
    tolerance = attr.ib(type=float, default=0.001)
    seed = attr.ib(type=int, default=random.randint(0,1 << 64))
    normalize = attr.ib(type=bool, default=True)
    quiet = attr.ib(type=bool, default=False)
    init_random = attr.ib(type=bool, default=True)
    output_ensemble = attr.ib(type=bool, default=False)

@attr.s
class QueryJudgments(object):
    docid_to_rel: Dict[str,float] = attr.ib(factory=dict)

@attr.s
class QuerySetJudgments(object):
    query_to_judgments: Dict[str, QueryJudgments] = attr.ib(factory=dict)

@attr.s
class TrainRequest(object):
    measure = attr.ib(type=str, default="ndcg")
    params = attr.ib(type=CoordinateAscentParams, factory=CoordinateAscentParams)
    judgments = attr.ib(type=QuerySetJudgments, default=None)




def claim_rust_str(result) -> str:
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


def query_json(message: str) -> str:
    """
    This method sends any old python object as input into the Rust exec_json call.
    The request is encoded on the way in.
    Returns a string.
    """
    command = message.encode("utf-8")
    response = json.loads(claim_rust_str(lib.query_json(command)))
    if 'error' in response and 'context' in response:
        raise Exception('{0}: {1}'.format(response['error'], response['context']))

    return response


#%%
if __name__ == "__main__":
    print(TrainRequest())

    train_req = query_json("coordinate_ascent_defaults")
    ca_params = train_req['params']['CoordinateAscent']
    ca_params['init_random'] = True
    ca_params['seed'] = 42
    ca_params['quiet'] = True
    print(train_req)
    
    train_X, train_y, train_qid = load_svmlight_file(
        "../examples/trec_news_2018.train",
        dtype=np.float32,
        zero_based=False,
        query_id=True,
    )
    train_X = train_X.todense()
    (train_N, train_D) = train_X.shape
    print(train_N, train_D)
    train_req_str = json.dumps(train_req).encode('utf-8')
    train_resp = json.loads(claim_rust_str(lib.train_dense_dataset_f32_f64_i64(
        train_req_str,
        train_N,
        train_D,
        ffi.cast("float *", train_X.ctypes.data),
        ffi.cast("double *", train_y.ctypes.data),
        ffi.cast("int64_t *", train_qid.ctypes.data),
    )))
    print(train_resp)
