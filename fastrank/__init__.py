#%%
from .clib import CQRel, CDataset, CModel, query_json
from .training import TrainRequest
from .models import CoordinateAscentRanker, RandomForestRanker, ImportedRanker, make_dataset

VERSION_TUPLE = (0,8,0)
__version__ = '{}.{}.{}'.format(*VERSION_TUPLE)
