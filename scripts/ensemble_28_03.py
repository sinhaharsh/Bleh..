from os import path

import numpy as np
import pandas as pd

from time import strftime
print strftime("%Y-%m-%d %H:%M:%S")

# import sys
# sys.stdout = open('grid_search_gbm.py', 'w')
# # print 'test'

# Any results you write to the current directory are saved as output.

from os import path

# load data
df_train = pd.read_csv(path.relpath("./input/train.csv"))
df_test = pd.read_csv(path.relpath("./input/test.csv"))

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




csv1 = pd.read_csv(path.relpath("./SubmissionCSV/submission_26_03_param_tuningWtheta.csv"))
csv2 = pd.read_csv(path.relpath("./SubmissionCSV/submission_26_03_param_tuning.csv"))
csv3 = pd.read_csv(path.relpath("./SubmissionCSV/submission_26_03_param_tuningWtheta2-5.csv"))
csv4 = pd.read_csv(path.relpath("./SubmissionCSV/submission_27_03_GBM_paramTuning.csv"))
csv5 = pd.read_csv(path.relpath("./SubmissionCSV/submission_24_03.csv"))
csv6 = pd.read_csv(path.relpath("./SubmissionCSV/submission_25_03_theta1-9.csv"))
csv7 = pd.read_csv(path.relpath("./SubmissionCSV/submission_29_03_Adaboost_paramTuning.csv"))

lbScores = [0.836782, 0.836466, 0.834351]
sum_lbScores = sum(lbScores)

weights = np.array(lbScores)/sum_lbScores
print "Distibuted Weights :", weights

# [[ 0.14433664  0.14428213  0.14391732  0.14246374  0.14215912  0.14188918
#    0.14095187]]

print len(csv1['TARGET'])

predMatrix = np.vstack([0.14433664*csv1['TARGET'], 0.14428213*csv2['TARGET'], 0.14391732*csv3['TARGET']])#, 0.14246374*csv4['TARGET'], 0.14215912*csv5['TARGET'], 0.14188918*csv6['TARGET'], 0.14095187*csv7['TARGET']]) 
print "Matrix Shape", np.shape(predMatrix)
print np.array(np.mat(weights))
print np.shape(np.array(np.mat(weights)))

y_pred = np.sum(predMatrix, axis=0)
print "Predictions Shape", np.shape(y_pred)


submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission_ensemble_28_03_1.csv", index=False)
