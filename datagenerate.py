import pandas as pd
import numpy as np
from scipy import sparse
import pickle,random,logging
from defs import *
from functools import partial
import warnings,gc
from tqdm import tqdm
warnings.simplefilter("ignore")
import multiprocessing as mp
from matplotlib import pyplot as plt
import seaborn as sns
logging.basicConfig(level=logging.INFO, format=' %(asctime)s %(message)s')

df = pd.read_csv('df.csv')
submit = pd.read_csv('ResultSample.csv')
logging.info('Parmeters Seting')
core,start,end,M,DR_mode,reduce_dM = 14,1000,600000,60000,'mean',1000
# =============================================================================
# Original data reduce dimension with mean-std pooling and drop outlier
# =============================================================================
dcut = int(M/reduce_dM)
logging.info(f'Get {DR_mode} pooling data: reduce {end-start} to {M} dimension')
GetDRdata_md5=partial(GetDRdata, M1=start,M2=end,mode='train',M=M,DR_med=DR_mode)
size = df.shape[0]
with mp.Pool(core) as p:
    train = list(tqdm(p.imap(GetDRdata_md5,df.md5[:size]), total=size))
train = sparse.csr_matrix(train)
gc.collect()
#plt.imshow(train.toarray(), cmap = 'gray',interpolation='nearest', aspect='auto')
logging.info(f'Get {DR_mode}-std pooling data: reduce {M} to {reduce_dM} dimension')
new_train = Reduce2std(train,reduce_dM)

#plt.imshow(new_train.toarray(), cmap = 'gray',interpolation='nearest', aspect='auto')

logging.info('Save Reduce data')
file = open(f'train_{int(end/10000)}w_{int(M/10000)}w_{int(reduce_dM/1000)}k.pickle', 'wb')
pickle.dump(new_train, file)

# =============================================================================
# Test Data
# =============================================================================
GetDRdata_md5=partial(GetDRdata, M1=start,M2=end,mode='test',M=M,DR_med=DR_mode)
size = submit.shape[0]
with mp.Pool(core) as p:
    test = list(tqdm(p.imap(GetDRdata_md5,submit.md5[:size]), total=size))
test = sparse.csr_matrix(test)
logging.info(f'Get {DR_mode}-std pooling data: reduce {M} to {reduce_dM} dimension')
new_test = Reduce2std(test,reduce_dM)

logging.info('Save Reduce data')
file = open(f'test_{int(end/10000)}w_{int(M/10000)}w_{int(reduce_dM/1000)}k.pickle', 'wb')
pickle.dump(new_test, file)


