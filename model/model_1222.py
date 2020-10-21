from xgboost import XGBClassifier
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.metrics import log_loss, make_scorer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool, cv
from bayes_opt import BayesianOptimization
import xgboost as xgb
import matplotlib.pyplot as plt
import lightgbm as lgb
import catboost as cb
from datetime import datetime
import multiprocessing as mp
import gc
import logging
import pickle
import random
import warnings
from functools import partial

import numpy as np
import pandas as pd
from defs import *
from scipy import sparse
from tqdm import tqdm

warnings.simplefilter("ignore")


pd.options.display.max_columns = 10

log_filename = datetime.now().strftime("log/tk%Y-%m-%d_%H_%M_%S.log")
# ,filename=log_filename)
logging.basicConfig(level=logging.INFO, format=' %(asctime)s %(message)s')

logging.info('Load Data')
df = pd.read_csv('data/df.csv')
submit = pd.read_csv('ResultSample.csv')

logging.info('Target Variable Tansform')
ydict = {'Backdoor': 1, 'PUA': 2, 'PWS': 3, 'Ransom': 4, 'SoftwareBundler': 5, 'Trojan': 6,
         'TrojanDownloader': 7, 'VirTool': 8, 'Virus': 9, 'Worm': 10}
df['y'] = df['y'].map(ydict)

logging.info('Parmeter Setting')
core, M1, M2, M = 12, 1024, 100000, 2
# 1024~57w (155w) train
# 1024~62w (172w) test

logging.info('Get data')
with open(f'data/train_1gram_stat.pickle', 'rb') as file:
    train_stat = pickle.load(file)
with open(f'data/test_1gram_stat.pickle', 'rb') as file:
    test_stat = pickle.load(file)

with open(f'data/train_60w_6w_1k.pickle', 'rb') as file:
    train = pickle.load(file)
with open(f'data/test_60w_6w_1k.pickle', 'rb') as file:
    test = pickle.load(file)

with open(f'train_size.pickle', 'rb') as file:
    train_size = pickle.load(file)
with open(f'test_size.pickle', 'rb') as file:
    test_size = pickle.load(file)

with open(f'train_4gram_small.pickle', 'rb') as file:
    train_4gram_small = pickle.load(file)
with open(f'test_4gram_small.pickle', 'rb') as file:
    test_4gram_small = pickle.load(file)

with open(f'train_control_6w.pickle', 'rb') as file:
    train_control_6w = pickle.load(file)
with open(f'test_control_6w.pickle', 'rb') as file:
    test_control_6w = pickle.load(file)

with open(f'train_control_41w.pickle', 'rb') as file:
    train_control_41w = pickle.load(file)
with open(f'test_control_41w.pickle', 'rb') as file:
    test_control_41w = pickle.load(file)

with open(f'train_control_21w.pickle', 'rb') as file:
    train_control_21w = pickle.load(file)
with open(f'test_control_21w.pickle', 'rb') as file:
    test_control_21w = pickle.load(file)

#plt.imshow(train.toarray(), cmap = 'gray',interpolation='nearest', aspect='auto')
train_size = sparse.csr_matrix(np.array(train_size).reshape([5907, 1]))
test_size = sparse.csr_matrix(np.array(test_size).reshape([5000, 1]))


new_train = sparse.hstack([train, train_stat, train_size, train_4gram_small,
                           train_control_6w, train_control_21w,
                           train_control_41w], format='csr')
new_test = sparse.hstack([test, test_stat, test_size, test_4gram_small,
                          test_control_6w, test_control_21w,
                          test_control_41w], format='csr')
del train, train_stat, train_size, train_control_6w, train_control_21w, train_control_41w
del test, test_stat, test_size, test_control_6w, test_control_21w, test_control_41w
gc.collect()
#################################################################################################
lgb_data = lgb.Dataset(data=new_train.toarray(), label=df.y.values)
n_folds, random_seed = 5, 666


def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, subsample_for_bin,
             lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {'objective': 'multiclass',
              'num_class': 11,
              'num_iterations': 4000,
              'learning_rate': 0.03,
              'n_jobs': core}
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


lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (30, 80),
                                        'feature_fraction': (0.1, 0.6),
                                        'bagging_fraction': (0.5, 1),
                                        'max_depth': (7, 15),
                                        'subsample_for_bin': (3000, 6000),
                                        'lambda_l1': (0, 0.4),
                                        'lambda_l2': (0.05, 0.3),
                                        'min_split_gain': (0.001, 0.4),
                                        'min_child_weight': (3, 20)}, random_state=0)
logging.info('Bayesian Optimzation Start')
lgbBO.maximize(init_points=20, n_iter=100)
logging.info('Bayesian Optimzation End')


logging.info(lgbBO.max)
# =============================================================================
# #  Lightgbm for 1-gram-stat and original-60w-1k-mean-std
# {'target': -0.8775517086013884,
#  'params': {'bagging_fraction': 0.9874006283627135, 'feature_fraction': 0.12344707323658986,
#             'lambda_l1': 0.28670400862277234, 'lambda_l2': 0.4143312662108787,
#             'max_depth': 12.660069446914932, 'min_child_weight': 5.812082847291121,
#             'min_split_gain': 0.10677364238023246, 'num_leaves': 59.055367124956206,
#             'subsample_for_bin': 3066.562912041851}}
# =============================================================================

# =============================================================================
# model = LGBMClassifier(learning_rate=0.01,
#                       num_iterations=5000,
#                       num_class=11,
#                       bagging_fraction=.9874006283627135,
#                       feature_fraction=.12344707323658986,
#                       lambda_l1=.28670400862277234,
#                       lambda_l2=.4143312662108787,
#                       max_depth=12,
#                       min_child_weight=5,
#                       min_split_gain=.10677364238023246,
#                       num_leaves=59,
#                       subsample_for_bin=3066,
#                       objective='multiclass',
#                       n_jobs=core,
#                       seed=666)
# =============================================================================

model = LGBMClassifier(learning_rate=0.005,
                       num_iterations=5000,
                       num_class=11,
                       bagging_fraction=lgbBO.max['params']['bagging_fraction'],
                       feature_fraction=lgbBO.max['params']['feature_fraction'],
                       lambda_l1=lgbBO.max['params']['lambda_l1'],
                       lambda_l2=lgbBO.max['params']['lambda_l2'],
                       max_depth=int(lgbBO.max['params']['max_depth']),
                       min_child_weight=lgbBO.max['params']['min_child_weight'],
                       min_split_gain=lgbBO.max['params']['min_split_gain'],
                       num_leaves=int(lgbBO.max['params']['num_leaves']),
                       subsample_for_bin=int(
                           lgbBO.max['params']['subsample_for_bin']),
                       objective='multiclass',
                       n_jobs=core,
                       seed=666)


logging.info('Single model fit start')
cv_result = cross_val_score(model, new_train, df.y,
                            cv=5, scoring='neg_log_loss')
logging.info(f'Single model fit end -- {cv_result.mean()}')


logging.info('Calibration try')
count, nfolds = 0, 5
compare_mat = np.zeros([3, 5])
kf = KFold(n_splits=nfolds, shuffle=True, random_state=666)
for train_idx, valid_idx in kf.split(new_train, df.y):
    train_data = new_train[train_idx, :], df.y.values[train_idx]
    valid_data = new_train[valid_idx, :], df.y.values[valid_idx]
    logging.info('start train')

    model.fit(train_data[0], train_data[1])
    model_calib1 = CalibratedClassifierCV(model, method="sigmoid", cv=5)
    model_calib1.fit(train_data[0], train_data[1])
    model_calib2 = CalibratedClassifierCV(model, method="isotonic", cv=5)
    model_calib2.fit(train_data[0], train_data[1])

    lgb_result = log_loss(valid_data[1], model.predict_proba(valid_data[0]))
    calib_result1 = log_loss(
        valid_data[1], model_calib1.predict_proba(valid_data[0]))
    calib_result2 = log_loss(
        valid_data[1], model_calib2.predict_proba(valid_data[0]))

    compare_mat[:, count] = lgb_result, calib_result1, calib_result2
    count += 1
    logging.info(
        f'lgb: {round(lgb_result,5)}, calib_sig:{round(calib_result1,5)}, calib_iso:{round(calib_result2,5)}')
    gc.collect()
print(compare_mat.mean(axis=0))


# =============================================================================
# semi-supervised learning
# =============================================================================

# Single model
X_train, X_test, y_train, y_test = train_test_split(
    new_train, df.y.values, test_size=0.2, random_state=2048)
model.fit(X_train, y_train)
lgb_result = log_loss(y_test, model.predict_proba(X_test))
logging.info(f'lgb single mlogloss: {lgb_result}')
y_pred = model.predict(X_test)


model1 = LGBMClassifier(learning_rate=0.005,
                        num_iterations=5000,
                        num_class=11,
                        bagging_fraction=lgbBO.max['params']['bagging_fraction'],
                        feature_fraction=lgbBO.max['params']['feature_fraction'],
                        lambda_l1=lgbBO.max['params']['lambda_l1'],
                        lambda_l2=lgbBO.max['params']['lambda_l2'],
                        max_depth=int(lgbBO.max['params']['max_depth']),
                        min_child_weight=lgbBO.max['params']['min_child_weight'],
                        min_split_gain=lgbBO.max['params']['min_split_gain'],
                        num_leaves=int(lgbBO.max['params']['num_leaves']),
                        subsample_for_bin=int(
                            lgbBO.max['params']['subsample_for_bin']),
                        objective='multiclass',
                        n_jobs=core,
                        seed=666)

score, nfolds, count = 0, 5, 0
kf = KFold(n_splits=nfolds, shuffle=True, random_state=666)
for train_idx, valid_idx in kf.split(X_test, y_test):
    train_data = sparse.vstack([X_train, X_test[train_idx, :]], format='csr'),\
        np.concatenate([y_train, y_pred[train_idx]], axis=0)
    valid_data = X_test[valid_idx, :], y_test[valid_idx]
    model1.fit(train_data[0], train_data[1], eval_set=[
               (valid_data)], verbose=0, early_stopping_rounds=200)
    a = [x1 for x1, _ in model.evals_result_.items()][0]
    b = [x2 for _, x1 in model.evals_result_.items() for x2 in x1][0]
    score += model1.evals_result_[a][b][-1]/nfolds
    qq = round(model1.evals_result_[a][b][-1], 5)
    orig_score = log_loss(valid_data[1], model.predict_proba(valid_data[0]))
    logging.info(
        f'semi-{count} mlogloss: {qq}, single mlogloss: {round(orig_score,5)}')
    gc.collect()
logging.info(f'Final mlogloss: {round(score,5)}')


# =============================================================================
# Submit
# =============================================================================
logging.info('Write result 1')
model.fit(new_train, df.y.values)
submit.iloc[:, 1:] = model.predict_proba(new_test)
submit.iloc[:, 1:] = submit.iloc[:, 1:].astype('float32')
submit.sum(axis=1)
submit.to_csv('submit1_1222.csv', index=False)


logging.info('Write result 2')
logging.info('Stacking')
score, nfolds = 0, 5
kf = KFold(n_splits=nfolds, shuffle=True, random_state=666)
result = sparse.csr_matrix(np.zeros([5000, 10]))
for train_idx, valid_idx in kf.split(new_train, df.y):
    train_data = new_train[train_idx, :], df.y.values[train_idx]
    valid_data = new_train[valid_idx, :], df.y.values[valid_idx]
    logging.info('start train')
    model.fit(train_data[0], train_data[1], eval_set=[
              (valid_data)], verbose=0, early_stopping_rounds=200)
    a = [x1 for x1, _ in model.evals_result_.items()][0]
    b = [x2 for _, x1 in model.evals_result_.items() for x2 in x1][0]
    score += model.evals_result_[a][b][-1]/nfolds
    qq = round(model.evals_result_[a][b][-1], 5)
    logging.info(f'mlogloss: {qq}')
    result += sparse.csr_matrix((model.predict_proba(new_test)))/nfolds
    gc.collect()
logging.info(f'Final mlogloss: {round(score,5)}')

submit.iloc[:, 1:] = result.toarray()
submit.iloc[:, 1:] = submit.iloc[:, 1:].astype('float32')
submit.sum(axis=1)

submit.to_csv('submit2_1222.csv', index=False)
