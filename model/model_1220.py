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
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss,make_scorer
from skopt.space import Real, Categorical, Integer
from catboost import Pool, cv


pd.options.display.max_columns = 10

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
core,M1,M2,M = 12,1024,100000,2
# 1024~57w (155w) train
# 1024~62w (172w) test

logging.info('Get data')
with open(f'train_1gram_stat.pickle','rb') as file:
    train_stat = pickle.load(file)
with open(f'test_1gram_stat.pickle','rb') as file:
    test_stat = pickle.load(file)

with open(f'train_60w_6w_1k.pickle','rb') as file:
    train = pickle.load(file)
with open(f'test_60w_6w_1k.pickle','rb') as file:
    test = pickle.load(file)

new_train = sparse.hstack([train,train_stat],format='csr')
new_test = sparse.hstack([test,test_stat],format='csr')
del train,train_stat,test,test_stat
#################################################################################################
lgb_data = lgb.Dataset(data=new_train.toarray(), label=df.y.values)
n_folds,random_seed = 5,666
def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth,subsample_for_bin,
             lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {'objective': 'multiclass',
             'num_class': 11,
             'num_iterations': 4000,
             'learning_rate': 0.03,
             'n_jobs':core}
    params["subsample_for_bin"] = int(subsample_for_bin)
    params["num_leaves"] = int(num_leaves)
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(max_depth)
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(params, lgb_data, nfold=n_folds, early_stopping_rounds=100,
                       seed=random_seed, stratified=True)
    return -min(cv_result['multi_logloss-mean'])

lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 60),
                                        'feature_fraction': (0.1, 0.9),
                                        'bagging_fraction': (0.5, 1),
                                        'max_depth': (4, 13),
                                        'subsample_for_bin':(3000,6000),
                                        'lambda_l1': (0,1),
                                        'lambda_l2': (0,1),
                                        'min_split_gain': (0.001, 0.5),
                                        'min_child_weight': (5, 50)}, random_state=0)
logging.info('Bayesian Optimzation Start')
lgbBO.maximize(init_points=20, n_iter=100)
logging.info('Bayesian Optimzation End')

logging.info(lgbBO.max)

# =============================================================================
### Lightgbm for 1-gram-stat and original-60w-1k-mean-std
{'target': -0.8761335591324919,
 'params': {'bagging_fraction': 0.7591234571011538,
  'feature_fraction': 0.21728530942570307,
  'lambda_l1': 0.2887853631779851,
  'lambda_l2': 0.5408532314624948,
  'max_depth': 12.468673410805737,
  'min_child_weight': 6.810750529318991,
  'min_split_gain': 0.007698850162408724,
  'num_leaves': 59.655888344868444,
  'subsample_for_bin': 3791.3902052686813}}
# =============================================================================

model = LGBMClassifier(learning_rate=0.01,
                      num_iterations=4000,
                      num_class=11,
                      bagging_fraction=lgbBO.max['params']['bagging_fraction'],
                      feature_fraction=lgbBO.max['params']['feature_fraction'],
                      lambda_l1=lgbBO.max['params']['lambda_l1'],
                      lambda_l2=lgbBO.max['params']['lambda_l2'],
                      max_depth=int(lgbBO.max['params']['max_depth']),
                      min_child_weight=lgbBO.max['params']['min_child_weight'],
                      min_split_gain=lgbBO.max['params']['min_split_gain'],
                      num_leaves=int(lgbBO.max['params']['num_leaves']),
                      subsample_for_bin=int(lgbBO.max['params']['subsample_for_bin']),
                      objective='multiclass',
                      n_jobs=core,
                      seed=666)
logging.info('Single model fit')
model.fit(new_train,df.y)

# =============================================================================
# Submit
# =============================================================================
logging.info('Write result 1')
submit.iloc[:,1:] = model.predict_proba(new_test)
submit.iloc[:,1:] = submit.iloc[:,1:].astype('float32')
submit.sum(axis=1)
submit.to_csv('submit1_1220.csv',index=False)


logging.info('Write result 2')
logging.info('Stacking')
score,nfolds = 0,5
kf = KFold(n_splits=nfolds, shuffle=True, random_state=666)
result = sparse.csr_matrix(np.zeros([5000,10]))
for train_idx, valid_idx in kf.split(new_train,df.y):
    train_data = new_train[train_idx,:],df.y.values[train_idx]
    valid_data = new_train[valid_idx,:],df.y.values[valid_idx]
    logging.info('start train')
    model.fit(train_data[0],train_data[1], eval_set=[(valid_data)],verbose=0,early_stopping_rounds=200)
    a = [x1 for x1,_ in model.evals_result_.items()][0]
    b = [x2 for _,x1 in model.evals_result_.items() for x2 in x1][0]
    score += model.evals_result_[a][b][-1]/nfolds
    qq = round(model.evals_result_[a][b][-1],5) 
    logging.info(f'mlogloss: {qq}')
    result += sparse.csr_matrix((model.predict_proba(new_test)))/nfolds
    gc.collect()
logging.info(f'Final mlogloss: {round(score,5)}')

submit.iloc[:,1:] = result.toarray()
submit.iloc[:,1:] = submit.iloc[:,1:].astype('float32')
submit.sum(axis=1)

submit.to_csv('submit2_1220.csv',index=False)
