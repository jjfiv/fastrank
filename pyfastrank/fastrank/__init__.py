#%%
import cffi
import numpy as np
from cfastrank import lib, ffi
import ujson as json
import sklearn
from sklearn.datasets import load_svmlight_file


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


def exec_json(obj) -> str:
    """
    This method sends any old python object as input into the Rust exec_json call.
    The request is encoded on the way in.
    Returns a string.
    """
    command = json.dumps(obj).encode("utf-8")
    response = claim_rust_str(lib.exec_json(command))
    return response


#%%
if __name__ == "__main__":
    qids = np.array(["001", "001", "002", "003", "greek_mythology"])
    print(qids.dtype)
    qs = np.array([1, 1, 1, 2, 2, 3])
    print(qs.dtype)

    print(exec_json({"hello": 7, "list": [1, 2, 3]}))

    train_X, train_y, train_qid = load_svmlight_file(
        "../examples/trec_news_2018.train",
        dtype=np.float32,
        zero_based=False,
        query_id=True,
    )
    test_X, test_y, test_qid = load_svmlight_file(
        "../examples/trec_news_2018.test",
        dtype=np.float32,
        zero_based=False,
        query_id=True,
    )

    # Don't yet support CSR matrices:
    train_X = train_X.todense()
    test_X = test_X.todense()

    print("X:: ")
    print(train_X.dtype, train_X.shape, type(train_X), train_X.__array_interface__)
    print(test_X.dtype, test_X.shape, type(test_X), test_X.__array_interface__)

    (train_N, train_D) = train_X.shape
    (test_N, test_D) = test_X.shape

    print("Y:: ")
    print(train_y.dtype, train_y.shape, type(train_y))
    print(test_y.dtype, test_y.shape, type(test_y))

    print("qid:: ")
    print(train_qid.dtype, train_qid.shape, type(train_qid))
    print(test_qid.dtype, test_qid.shape, type(test_qid))

    lib.create_dense_dataset_f32_f64_i64(
        train_N,
        train_D,
        ffi.cast("float *", train_X.ctypes.data),
        ffi.cast("double *", train_y.ctypes.data),
        ffi.cast("int64_t *", train_qid.ctypes.data),
    )
    for i in range(0, train_D):
        print("py_x[17,..]", train_X[17, i])
        print("py_y,qid", train_y[17], train_qid[17])


#%%
