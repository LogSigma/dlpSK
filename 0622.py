import os
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df_t = pd.read_csv('~/git/dlp_2019/data/skhy_yield_prediction.csv', sep=",", encoding="CP949")

df_t["END_TM"] = df_t.END_TM.str.replace("오후","PM").str.replace("오전","AM")

df_t['END_TM'] = pd.to_datetime(df_t['END_TM'], format='%Y-%m-%d %p %H:%M:%S')
#df_t['END_TM_INT'] = (pd.to_numeric(df_t['END_TM']-df_t['END_TM'][4])/1000000000).astype(int)
df_t.fillna(df_t.mean(), inplace=True)
df_t = df_t.set_index('ID')
df_t = df_t.drop(['A','B','END_TM'], axis=1)

import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from unipy.stats.formula import from_formula
from unipy.stats import vif

def feature_selection_vif(data, thresh=5.0):
    
    assert isinstance(data, pd.DataFrame)

    # Create Dropped variable list
    dropped = pd.DataFrame(columns=['var', 'vif'])

    # Startswith 'drop = True'(Assume that some variables will be dropped)
    dropCondition = True

    # Calculate a VIF & Drop columns(variables)
    while dropCondition:

        # 1. Calculate a VIF
        vifDict = {col: vif(data.loc[:, col], data.loc[:, data.columns != col])
                   for col in data.columns}

        # Get the MAXIMUM VIF
        maxVar = max(vifDict, key=vifDict.get)
        maxVal = vifDict[maxVar]

        # 2. IF VIF values are over the threshold, THEN drop it
        if maxVal >= thresh:

            # Keep it
            dropped = dropped.append({'var': maxVar, 'vif': maxVal},
                                     ignore_index=True)

            # Drop it
            data = data.drop(maxVar, axis=1)

            # Print it
            print("Dropping '" + str(maxVar) + "' " + " VIF: " + str(maxVal))

            # Since a variable has been dropped, the assumption remains
            dropCondition = True

        else:

            # No variable dropped, the assumption has been rejected
            dropCondition = False

    # Print Massages
    remainsMsg = '# Remaining Variables '
    msgWrapper = '-' * (len(remainsMsg)+1)

    print('\n' + msgWrapper + '\n' + remainsMsg + '\n' + msgWrapper)
    print(list(data.columns))
    print('\n')

    droppedMsg = '# Dropped Variables '
    msgWrapper = '-' * (len(remainsMsg)+1)
    print('\n' + msgWrapper + '\n' + droppedMsg + '\n' + msgWrapper)
    print(list(dropped.loc[:, 'var']))
    print('\n')

    return data, dropped

feature_selection_vif(df_t.drop(['Y','Validation'], axis=1),)

drop_list = ['C2', 'C73', 'C55', 'C43', 'C79', 'C39', 'C66', 'C12', 'C42', 'C67', 'C26', 'C84', 'C90', 'C115', 'C9', 'C71', 'C28', 'C8', 'C35', 'C64', 'C68', 'C98', 'C108', 'C101', 'C25', 'C74', 'C99', 'C5', 'C117', 'C76', 'C11', 'C82']

df_t = df_t.drop(drop_list,axis=1)

df_t.shape

df_m1 = df_t[df_t['Validation']==0]
df_m1_y = df_m1['Y']

df_train = df_m1[:int(len(df_m1)*0.75)]
df_test = df_m1.drop(df_train.index.tolist())

df_train_y = df_train['Y']
df_test_y = df_test['Y']

df_train_x = df_train.drop(['Y','Validation'], axis=1)
df_test_x = df_test.drop(['Y','Validation'],axis=1)

model_gboost = GradientBoostingRegressor(loss='huber',n_estimators = 110, max_features='sqrt',
                                         max_depth=7, learning_rate=0.05, random_state=0)

model_gboost.fit(df_train_x, df_train_y)
gboost_pred = model_gboost.predict(df_test_x)

print('R2: ',r2_score(df_test_y, gboost_pred))
print('MSE: ',mean_squared_error(df_test_y, gboost_pred))

#R2:  0.5509361534179527
#MSE:  31.609030060134945
