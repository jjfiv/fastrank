#%%
import cffi
import numpy as np
from cfastrank import lib, ffi
import ujson as json


def claim_rust_str(result) -> str:
    """
    This method decodes bytes to UTF-8 and makes a new python string object. 
    It then frees the bytes that Rust allocated correctly.
    """
    print(type(result))
    txt = ffi.cast("char*", result)
    txt = ffi.string(txt).decode("utf-8")
    lib.free_str(result)
    return txt


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


#%%
