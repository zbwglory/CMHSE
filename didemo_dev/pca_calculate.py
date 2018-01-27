import sklearn.decomposition
import json
import numpy as np
from IPython import embed
import os
from tqdm import tqdm

pca = sklearn.decomposition.TruncatedSVD(500)


train_json = json.load(open('train_data_bwzhang.json','r'))
train_total = []
path = '/data2/bwzhang/research/data/LocalizingMoments/ICEP_V3_global_pool_no_skip_direct_resize/'
sum_total = 0
for filename in tqdm(train_json):
    score = np.load(os.path.join(path, filename+'.npz'))['frame_scores'].squeeze()
    score_dim = score.shape[0]
    sum_total = sum_total + score_dim 
    score_dim_perm = np.random.permutation(score_dim)[0:int(score_dim*0.5)].tolist()
    score = score[score_dim_perm,:]
    train_total.append(score)


train_total = np.concatenate(train_total, 0)
print train_total.shape
pca.fit(train_total)
embed()
