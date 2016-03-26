
# coding: utf-8

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.grid_search import GridSearchCV   #Perforing grid search

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./input"]).decode("utf8"))

import sys
sys.stdout = open('grid_search_theta1.9.txt', 'w')
# print 'test'

# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

theta = 1.9
X_train = (1/theta)*np.arcsinh(X_train*theta)

X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)

# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }

# gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=3,
#  min_child_weight=6, gamma=0.1, subsample=0.85, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
#  param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(X_fit,y_fit)
# print "gsearch1.grid_scores_",gsearch1.grid_scores_
# print "gsearch1.best_params_", gsearch1.best_params_
# print "gsearch1.best_score_", gsearch1.best_score_

# g1_MAX_DEPTH = gsearch1.best_params_['max_depth']
# g1_MIN_CHILD_WEIGHT = gsearch1.best_params_['min_child_weight']

param_test2 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=3,
 min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_fit,y_fit)
print "gsearch2.grid_scores_",gsearch2.grid_scores_
print "gsearch2.best_params_", gsearch2.best_params_
print "gsearch2.best_score_", gsearch2.best_score_

# param_test3 = {
#  
#  }
# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch3.fit(X_fit,y_fit)
# print "gsearch3.grid_scores_",gsearch3.grid_scores_
# print "gsearch3.best_params_", gsearch3.best_params_
# print "gsearch3.best_score_", gsearch3.best_score_


# clf = None
# X_train = (1/theta)*np.arcsinh(X_train*theta)
# classifier
# clf = xgb.XGBClassifier(missing=np.nan,  
# 						learning_rate =0.09,
# 						n_estimators=5000,
# 						max_depth=3,
# 						min_child_weight=6,
# 						gamma=0.1,
# 						subsample=0.85,
# 						colsample_bytree=0.9,
# 						reg_alpha=5,
# 						objective= 'binary:logistic',
# 						nthread=4,
# 						scale_pos_weight=1,
# 						seed=27)


# # fitting
# clf.fit(X_fit, y_fit, early_stopping_rounds=25, eval_metric="auc", eval_set=[(X_eval, y_eval)])
# print 'Overall AUC:', roc_auc_score(y_fit, clf.predict_proba(X_fit)[:,1])

					

# # # predicting
# theta = 0.6
# X_test = (1/theta)*np.arcsinh(X_test*theta)
# y_pred= clf.predict_proba(X_test)[:,1]

# submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
# submission.to_csv("submission_26_03_param_tuning.csv", index=False)

# print('Completed!')

