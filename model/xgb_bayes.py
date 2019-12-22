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
from catboost import Pool, cv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier


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

with open(f'train_size.pickle','rb') as file:
    train_size = pickle.load(file)
with open(f'test_size.pickle','rb') as file:
    test_size = pickle.load(file)
# =============================================================================
# with open(f'train_4gram.pickle','rb') as file:
#     train_4gram = pickle.load(file)
# with open(f'test_4gram.pickle','rb') as file:
#     test_4gram = pickle.load(file)
# =============================================================================
with open(f'train_control_6w.pickle','rb') as file:
    train_control_6w = pickle.load(file)
with open(f'test_control_6w.pickle','rb') as file:
    test_control_6w = pickle.load(file)

with open(f'train_control_41w.pickle','rb') as file:
    train_control_41w = pickle.load(file)
with open(f'test_control_41w.pickle','rb') as file:
    test_control_41w = pickle.load(file)

with open(f'train_control_21w.pickle','rb') as file:
    train_control_21w = pickle.load(file)
with open(f'test_control_21w.pickle','rb') as file:
    test_control_21w = pickle.load(file)
    
#plt.imshow(train.toarray(), cmap = 'gray',interpolation='nearest', aspect='auto')
train_size = sparse.csr_matrix(np.array(train_size).reshape([5907,1]))
test_size = sparse.csr_matrix(np.array(test_size).reshape([5000,1]))


new_train = sparse.hstack([train,train_stat,train_size,
                           train_control_6w,train_control_21w,
                           train_control_41w],format='csr')
new_test = sparse.hstack([test,test_stat,test_size,
                          test_control_6w,test_control_21w,
                          test_control_41w],format='csr')
del train,train_stat,train_size,train_control_6w,train_control_21w,train_control_41w
del test,test_stat,test_size,test_control_6w,test_control_21w,test_control_41w
gc.collect()
#################################################################################################
logging.info('split data')
X_train, X_test, y_train, y_test = train_test_split(new_train,df.y,test_size=0.1,random_state=2048)

n_folds,random_seed = 5,666

def xgb_eval(gamma,num_leaves, max_depth,min_child_weight,
             colsample_bytree,subsample):
    fit_params={'early_stopping_rounds': 100, 
                'verbose': False,
                'eval_set':[[X_test,y_test]]}
    clf = XGBClassifier(learning_rate=0.03,
                        n_estimators=2000,
                        tree_method='gpu_hist',
                        max_depth=int(max_depth),
                        num_leaves = int(num_leaves),
                        gamma = gamma,
                        min_child_weight=min_child_weight, 
                        subsample=subsample, 
                        colsample_bytree=colsample_bytree,
                        eval_metric='mlogloss',
                        verbose = 0,
                        n_jobs=5,
                        seed=666)
    cv_result = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 scoring='neg_log_loss',
                                 cv=n_folds, 
                                 fit_params = fit_params,
                                 verbose=0, 
                                 n_jobs=1)    
    gc.collect()
    return cv_result.mean()
xgbBO = BayesianOptimization(xgb_eval, {'gamma':(0,1),
                                        'num_leaves': (24, 45),
                                        'max_depth': (3, 11),
                                        'min_child_weight': (2, 20),
                                        'colsample_bytree':(0.3,0.9),
                                        'subsample':(0.3,0.9)
                                        }, 
                             random_state=0)
logging.info('Bayesian Optimization Start')
xgbBO.maximize(init_points=30 , n_iter=200)
logging.info('Bayesian Optimization End')
logging.info(xgbBO.max)


# =============================================================================
# #  XGBoost for 1-gram-stat and original-60w-1k-mean-std
# =============================================================================

# =============================================================================
# model = LGBMClassifier(learning_rate=0.01,
#                       num_iterations=5000,
#                       num_class=11,
#                       bagging_fraction=lgbBO.max['params']['bagging_fraction'],
#                       feature_fraction=lgbBO.max['params']['feature_fraction'],
#                       lambda_l1=lgbBO.max['params']['lambda_l1'],
#                       lambda_l2=lgbBO.max['params']['lambda_l2'],
#                       max_depth=int(lgbBO.max['params']['max_depth']),
#                       min_child_weight=lgbBO.max['params']['min_child_weight'],
#                       min_split_gain=lgbBO.max['params']['min_split_gain'],
#                       num_leaves=int(lgbBO.max['params']['num_leaves']),
#                       subsample_for_bin=int(lgbBO.max['params']['subsample_for_bin']),
#                       objective='multiclass',
#                       n_jobs=core,
#                       seed=666)
# =============================================================================


logging.info('Calibration try')
count,nfolds = 0,5
compare_mat = np.zeros([3,5])
kf = KFold(n_splits=nfolds, shuffle=True, random_state=666)
for train_idx, valid_idx in kf.split(new_train,df.y):
    train_data = new_train[train_idx,:],df.y.values[train_idx]
    valid_data = new_train[valid_idx,:],df.y.values[valid_idx]
    logging.info('start train')
    
    model.fit(train_data[0],train_data[1])
    model_calib1 = CalibratedClassifierCV(model, method="sigmoid", cv=5)
    model_calib1.fit(train_data[0],train_data[1])
    model_calib2 = CalibratedClassifierCV(model, method="isotonic", cv=5)
    model_calib2.fit(train_data[0],train_data[1])
    
    lgb_result = log_loss(valid_data[1],model.predict_proba(valid_data[0]))
    calib_result1 = log_loss(valid_data[1],model_calib1.predict_proba(valid_data[0]))
    calib_result2 = log_loss(valid_data[1],model_calib2.predict_proba(valid_data[0])) 
    
    compare_mat[:,count] = lgb_result,calib_result1,calib_result2
    count+=1
    logging.info(f'lgb: {round(lgb_result,5)}, calib_sig:{round(calib_result1,5)}, calib_iso:{round(calib_result2,5)}')
    gc.collect()
print(compare_mat.mean(axis=0))