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
from bayes_opt import BayesianOptimization
import xgboost as xgb
import lightgbm as lgb

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
core,M1,M2,M = 6,1024,100000,2
# 1024~57w (155w) train
# 1024~62w (172w) test

logging.info('Get data')

with open(f'train_1gram_stat.pickle','rb') as file:
    train_stat = pickle.load(file)
with open(f'test_1gram_stat.pickle','rb') as file:
    test_stat = pickle.load(file)

#################################################################################################
logging.info('split data')
X_train, X_test, y_train, y_test = train_test_split(train_stat.toarray(),df.y,test_size=0.1,random_state=2048)

train_data = lgb.Dataset(data=pd.DataFrame(train_stat.toarray()), label=df.y.values)
n_folds,random_seed = 5,666

def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth,
             lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {'objective': 'multiclass',
             'num_class': 11,
             'num_iterations': 4000,
             'learning_rate': 0.03,
             'early_stopping_round': 100,
             'num_leaves': 10,
             'max_depth': 5,
             'n_jobs':-1,
             'lambda_l1': 0,
             'lambda_l2': 0,
             'min_split_gain': 0.05,
             'min_child_weight': 3}
    params["num_leaves"] = int(num_leaves)
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(max_depth)
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, 
                       stratified=True)
    return min(cv_result['multi_logloss-mean'])

lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                        'feature_fraction': (0.1, 0.9),
                                        'bagging_fraction': (0.8, 1),
                                        'max_depth': (5, 8.99),
                                        'lambda_l1': (0, 5),
                                        'lambda_l2': (0, 3),
                                        'min_split_gain': (0.001, 0.1),
                                        'min_child_weight': (5, 50)}, random_state=0)
lgbBO.maximize(init_points=3, n_iter=10)
lgbBO.res['max']['max_params']


model = LGBMClassifier(learning_rate=0.01,
                      num_iterations=1767,
                      num_leaves = 24,
                      max_depth=9,
                      min_child_weight=3, 
                      gamma=0.01,
                      subsample=0.7, 
                      colsample_bytree=0.6,
                      num_class=11,
                      objective='multiclass',
                      n_jobs=5,
                      seed=666)



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


