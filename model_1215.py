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
from time import time
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime

log_filename = datetime.now().strftime("log/tk%Y-%m-%d_%H_%M_%S.log")
logging.basicConfig(level=logging.INFO, format=' %(asctime)s %(message)s')#,filename=log_filename)

logging.info('Load Data')
df = pd.read_csv('df.csv')
submit = pd.read_csv('ResultSample.csv')

logging.info('Target Variable Tansform')
ydict = {'Backdoor':1, 'PUA':2, 'PWS':3, 'Ransom':4, 'SoftwareBundler':5, 'Trojan':6,
       'TrojanDownloader':7, 'VirTool':8, 'Virus':9, 'Worm':10}
df['y'] = df['y'].map(ydict)



logging.info('Parmeter Setting')
core,M1,M2,M = 13,1024,100000,2
# 1024~57w (155w) train
# 1024~62w (172w) test

logging.info('Get data')
'''
# train data
ngram_stat_md5=partial(ngram_stat,mode='train',M=M)
size = df.shape[0]
with mp.Pool(core) as p:
    train = list(tqdm(p.imap(ngram_stat_md5,df.md5[:size]), total=size))
train = sparse.csr_matrix(train)
gc.collect()

ngram_stat_md5=partial(ngram_stat,mode='test',M=M)
size = submit.shape[0]
with mp.Pool(core) as p:
    test = list(tqdm(p.imap(ngram_stat_md5,submit.md5[:size]), total=size))
test = sparse.csr_matrix(test)
gc.collect()
with open(f'train_2gram_stat.pickle','rb') as file:
    train = pickle.load(file)
with open(f'test_2gram_stat.pickle','rb') as file:
    test = pickle.load(file)
    
with open(f'train_stat.pickle','rb') as file:
    train_stat = pickle.load(file)
with open(f'test_stat.pickle','rb') as file:
    test_stat = pickle.load(file)
'''

#################################################################################################
'''
logging.info('SVD for 1000 components')
svd = TruncatedSVD(n_components=1000, n_iter=20, random_state=666)
logging.info('SVD fit')
new_train = svd.fit_transform(train)
new_test = svd.transform(test)
logging.info('SVD fit OK')
plt.plot(range(len(svd.explained_variance_ratio_)),np.cumsum(svd.explained_variance_ratio_))


logging.info('Merge stat feature')
svdVar = np.where(np.cumsum(svd.explained_variance_ratio_)>0.99995)[0][0]
new_train = np.concatenate([new_train[:,:svdVar],train_stat.toarray()],axis=1)
new_test = np.concatenate([new_test[:,:svdVar],test_stat.toarray()],axis=1)

logging.info('Save final data')
file = open(f'new_train1215.pickle', 'wb')
pickle.dump(new_train, file)
file.close()
file = open(f'new_test_1215.pickle', 'wb')
pickle.dump(new_test, file)
file.close()
'''

with open(f'new_train1215.pickle','rb') as file:
    train = pickle.load(file)
with open(f'new_test_1215.pickle','rb') as file:
    test = pickle.load(file)
    
    

logging.info('split data')
X_train, X_test, y_train, y_test = train_test_split(new_train,df.y,test_size=0.1,random_state=2048)

model = XGBClassifier(learning_rate=0.01,
                      n_estimators=1767,
                      tree_method='gpu_hist',
                      #tree_method='hist',
                      max_depth=9,
                      min_child_weight=3, 
                      gamma=0.01,
                      subsample=0.7, 
                      colsample_bytree=0.6,
                      eval_metric='mlogloss',
                      n_jobs=11,
                      seed=666)
logging.info('Fit xgb')
model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=1,early_stopping_rounds=200)
logging.info('Fit xgb OK')

# 0.02  798  6 3 0.03 0.8 0.6 0.87112
# 0.02  768 10 3 0.01 0.8 0.6 0.86328 
# 0.01 1767  9 3 0.01 0.7 0.6 0.85965


logging.info('Single model fit')
model.fit(train,df.y)

submit.iloc[:,1:] = model.predict_proba(test)
submit.iloc[:,1:] = submit.iloc[:,1:].astype('float32')
submit.sum(axis=1)
submit.to_csv('submit1_1215.csv',index=False)



logging.info('Stacking')
score,nfolds = 0,5
kf = KFold(n_splits=nfolds, shuffle=True, random_state=666)
result = sparse.csr_matrix(np.zeros([5000,10]))
for train_idx, valid_idx in kf.split(train,df.y):
    train_data = train[train_idx,:],df.y.values[train_idx]
    valid_data = train[valid_idx,:],df.y.values[valid_idx]
    logging.info('start train')
    model.fit(train_data[0],train_data[1], eval_set=[(valid_data)],verbose=0,early_stopping_rounds=200)
    score += model.evals_result_['validation_0']['mlogloss'][-1]/nfolds
    qq = round(model.evals_result_['validation_0']['mlogloss'][-1],5) 
    logging.info(f'mlogloss: {qq}')
    result += sparse.csr_matrix((model.predict_proba(test)))/nfolds
    gc.collect()
logging.info(f'Final mlogloss: {round(score,5)}')

submit.iloc[:,1:] = result.toarray()
submit.iloc[:,1:] = submit.iloc[:,1:].astype('float32')
submit.sum(axis=1)

submit.to_csv('submit2_1215.csv',index=False)


