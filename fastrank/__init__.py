#%%
from .clib import CQRel, CDataset, CModel, query_json
from .training import TrainRequest

VERSION_TUPLE = (0,7,0)
__version__ = '{}.{}.{}'.format(*VERSION_TUPLE)

__all__ = ['clib', 'training', 'CQRel', 'CDataset', 'CModel', 'query_json', 'TrainRequest']