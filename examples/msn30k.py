#%%
import fastrank
from fastrank import CModel, CDataset, TrainRequest


#%%
train = CDataset.open_ranksvm('msn30k-fold1-train.gz')
print("{} train queries...".format(len(train.queries())))
#%%
vali = CDataset.open_ranksvm('msn30k-fold1-vali.gz')
print("{} vali queries...".format(len(vali.queries())))
#%%
test = CDataset.open_ranksvm('msn30k-fold1-test.gz')
print("{} test queries...".format(len(test.queries())))

