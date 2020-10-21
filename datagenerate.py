import gc
import logging
import multiprocessing as mp
import pickle
import random
import warnings
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from defs import *

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO, format=' %(asctime)s %(message)s')

df = pd.read_csv('./data/df.csv')
submit = pd.read_csv('ResultSample.csv')
logging.info('Parmeters Seting')
core, start, end, M, DR_mode, reduce_dM = 13, 1000, 600000, 60000, 'mean', 1000
# =============================================================================
# Original data reduce dimension with mean-std pooling
# =============================================================================
dcut = int(M/reduce_dM)
logging.info(
    f'Get {DR_mode} pooling data: reduce {end-start} to {M} dimension')
GetDRdata_md5 = partial(GetDRdata, M1=start, M2=end,
                        mode='train', M=M, DR_med=DR_mode)
size = df.shape[0]
with mp.Pool(core) as p:
    train = list(tqdm(p.imap(GetDRdata_md5, df.md5[:size]), total=size))
train = sparse.csr_matrix(train)
logging.info(
    f'Get {DR_mode}-std pooling data: reduce {M} to {reduce_dM} dimension')
new_train = Reduce2std(train, reduce_dM)

GetDRdata_md5 = partial(GetDRdata, M1=start, M2=end,
                        mode='test', M=M, DR_med=DR_mode)
size = submit.shape[0]
with mp.Pool(core) as p:
    test = list(tqdm(p.imap(GetDRdata_md5, submit.md5[:size]), total=size))
test = sparse.csr_matrix(test)
logging.info(
    f'Get {DR_mode}-std pooling data: reduce {M} to {reduce_dM} dimension')
new_test = Reduce2std(test, reduce_dM)


logging.info('Save Reduce data')
file = open(
    f'train_{int(end/10000)}w_{int(M/10000)}w_{int(reduce_dM/1000)}k.pickle', 'wb')
pickle.dump(new_train, file)

logging.info('Save Reduce data')
file = open(
    f'test_{int(end/10000)}w_{int(M/10000)}w_{int(reduce_dM/1000)}k.pickle', 'wb')
pickle.dump(new_test, file)
# =============================================================================
# data size statistic
# =============================================================================
getDataSize_md5 = partial(getDataSize, mode='train', data=df)
size = df.shape[0]
with mp.Pool(core) as p:
    train_size = list(tqdm(p.imap(getDataSize_md5, df.md5[:size]), total=size))

getDataSize_md5 = partial(getDataSize, mode='test', data=df)
size = submit.shape[0]
with mp.Pool(core) as p:
    test_size = list(
        tqdm(p.imap(getDataSize_md5, submit.md5[:size]), total=size))

logging.info('Save Reduce data')
file = open(f'train_size.pickle', 'wb')
pickle.dump(train_size, file)
file = open(f'test_size.pickle', 'wb')
pickle.dump(test_size, file)
# =============================================================================
# n-gram reduction (no frequent)
# =============================================================================
logging.info(f'n-gram reduction')
ngram_md5 = partial(ngram, mode='train', M1=1024, M2=500000, M=4)
size = df.shape[0]
with mp.Pool(core) as p:
    train_4gram = list(tqdm(p.imap(ngram_md5, df.md5[:size]), total=size))
train_4gram = sparse.csr_matrix(train_4gram)

ngram_md5 = partial(ngram, mode='test', M1=1024, M2=500000, M=4)
size = submit.shape[0]
with mp.Pool(core) as p:
    test_4gram = list(tqdm(p.imap(ngram_md5, submit.md5[:size]), total=size))
test_4gram = sparse.csr_matrix(test_4gram)


logging.info(f'n-gram reduction')
file = open(f'train_4gram.pickle', 'wb')
pickle.dump(train_4gram, file)

file = open(f'test_4gram.pickle', 'wb')
pickle.dump(test_4gram, file)
# =============================================================================
# control limit feature (for different seqence length and control limit threshold)
# =============================================================================
thresholds = [0.3, 0.5, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3]
for max_seq in [x*10000 for x in range(6, 50, 5)]:
    logging.info(f'Generate train data {max_seq}')
    control_limit_md5 = partial(
        control_limit, max_seq=max_seq, mode='train', thresholds=thresholds)
    size = df.shape[0]
    with mp.Pool(core) as p:
        train_control = list(
            tqdm(p.imap(control_limit_md5, df.md5[:size]), total=size))
    train_control = sparse.csr_matrix(train_control)
    logging.info(f'Generate test data {max_seq}')
    control_limit_md5 = partial(
        control_limit, max_seq=max_seq, mode='test', thresholds=thresholds)
    size = submit.shape[0]
    with mp.Pool(core) as p:
        test_control = list(
            tqdm(p.imap(control_limit_md5, submit.md5[:size]), total=size))
    test_control = sparse.csr_matrix(test_control)
    logging.info(f'Save data {max_seq}')
    file = open(f'train_control_{int(max_seq/10000)}w.pickle', 'wb')
    pickle.dump(train_control, file)

    file = open(f'test_control_{int(max_seq/10000)}w.pickle', 'wb')
    pickle.dump(test_control, file)

# =============================================================================
# 4-gram RFFS
# =============================================================================
with open(f'train_4gram.pickle', 'rb') as file:
    train_4gram = pickle.load(file)
with open(f'test_4gram.pickle', 'rb') as file:
    test_4gram = pickle.load(file)

clf = RandomForestClassifier(max_depth=13, random_state=0, n_estimators=800,
                             max_features=1000, n_jobs=12, oob_score=True,
                             class_weight="balanced")
clf.fit(train_4gram, df.y)

fs_tar = sorted(clf.feature_importances_, reverse=True)[500]
fs = np.where(clf.feature_importances_ > fs_tar)[0]


file = open(f'train_4gram_small.pickle', 'wb')
pickle.dump(train_4gram[:, fs], file)

file = open(f'test_4gram_small.pickle', 'wb')
pickle.dump(test_4gram[:, fs], file)
