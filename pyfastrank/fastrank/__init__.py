#%%
import attr
import random
from typing import Dict, List, Set
from collections import Counter

from .clib import CQRel, CDataset, CModel, query_json


@attr.s
class CoordinateAscentParams(object):
    num_restarts = attr.ib(type=int, default=5)
    num_max_iterations = attr.ib(type=int, default=25)
    step_base = attr.ib(type=float, default=0.05)
    step_scale = attr.ib(type=float, default=2.0)
    tolerance = attr.ib(type=float, default=0.001)
    seed = attr.ib(type=int, default=random.randint(0, (1 << 64) - 1))
    normalize = attr.ib(type=bool, default=True)
    quiet = attr.ib(type=bool, default=False)
    init_random = attr.ib(type=bool, default=True)
    output_ensemble = attr.ib(type=bool, default=False)


@attr.s
class TrainRequest(object):
    measure = attr.ib(type=str, default="ndcg")
    params = attr.ib(type=CoordinateAscentParams, factory=CoordinateAscentParams)
    judgments = attr.ib(type=CQRel, default=None)
