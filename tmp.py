#%%

import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df_t = pd.read_csv('D:/DataSet_accuracy.csv', sep=",")

df_t['END_TM'] = pd.to_datetime(df_t['END_TM'], format='%Y-%m-%d %p %H:%M:%S')

df_t['END_TM_INT'] = (pd.to_numeric(df_t['END_TM']-df_t['END_TM'][4])/1000000000).astype(int)

#%%
#데이터 분석 로직

df_t.groupby('B')['Y'].mean()
df_t.groupby('B')['ID'].count()
df_t.groupby(['Validation','B'])['ID'].count()
#b column에 E,K 필요 없음

df_t.groupby('A')['Y'].mean()
df_t.groupby('A')['ID'].count()
df_t.groupby(['Validation','A'])['ID'].count()
#a column에 KD 필요 없음

for i in range(1,117):
	plt.boxplot(df_t['C{0}'.format(i)])
	plt.title('C{0}'.format(i))
	plt.show()

#C6 id 12728
#C8 id 12728

plt.rc('font',family='Malgun Gothic')
aa=sns.factorplot('A','Y',data=df_t,size=4,aspect=2)

plt.rc('font',family='Malgun Gothic')
aa=sns.factorplot('B','Y',data=df_t,size=4,aspect=2)

i=1
for i in range(1,117):
	plt.scatter(df_t['END_TM_INT'], df_t['C{0}'.format(i)])
    plt.title('C{0}'.format(i))
    plt.show()

#%%
#데이터 전처리
#df_t = df_t[df_t.B != 'E']
#df_t = df_t[df_t.B != 'K']

#df_t = df_t[df_t.A != 'KD']

df_t = df_t.drop(['C2','C20','C73','C113','C77','C39','C42','C43','C66','C22'],axis=1)
df_t = df_t[df_t.ID != 12728]
#df_t = df_t[df_t.ID != 2165]

df_t = df_t.fillna(df_t.median())

df_t = df_t.set_index('ID')

df_t = pd.get_dummies(df_t, columns=['A','B'])

df_m1 = df_t[df_t['Validation']==0]

df_m1 = df_m1.fillna(df_m1.median())

df_m1_y = df_m1['Y']

df_train = df_m1[:int(len(df_m1)*0.75)]
df_test = df_m1.drop(df_train.index.tolist())

df_train_y = df_train['Y']
df_test_y = df_test['Y']

df_train_x = df_train.drop(['Y','END_TM','Validation'], axis=1)
df_test_x = df_test.drop(['Y','END_TM','Validation'],axis=1)

#%%
#데이터 모델링
model_xgb = xgb.XGBRegressor(max_depth=7, n_estimators=110, learning_rate=0.05)

model_xgb.fit(df_train_x, df_train_y)
xgb_pred = model_xgb.predict(df_test_x)

print('R2: ',r2_score(df_test_y, xgb_pred))
print('MSE: ',mean_squared_error(df_test_y, xgb_pred))

#R2:  0.5522624907751074
#MSE:  31.464709224938723

#%%
model_gboost = GradientBoostingRegressor(loss='huber',n_estimators = 110, max_features='sqrt',
                                         max_depth=7, learning_rate=0.05)
model_gboost.fit(df_train_x, df_train_y)
gboost_pred = model_gboost.predict(df_test_x)

print('R2: ',r2_score(df_test_y, gboost_pred))
print('MSE: ',mean_squared_error(df_test_y, gboost_pred))

#R2:  0.5537570481212277
#MSE:  31.35967935510029

#%%
model_lgb = lgb.LGBMRegressor(objective = 'regression',n_estimators = 110, max_depth=7, learning_rate=0.05)
model_lgb.fit(df_train_x, df_train_y)
lgb_pred = model_lgb.predict(df_test_x)

print('R2: ',r2_score(df_test_y, lgb_pred))
print('MSE: ',mean_squared_error(df_test_y, lgb_pred))

#R2:  0.10503684525848689
#MSE:  62.89344727835004

#%%
model_svr = SVR(kernel='poly')
model_svr.fit(df_train_x, df_train_y)
svr_pred = model_svr.predict(df_test_x)

print('R2: ',r2_score(df_test_y, svr_pred))
print('MSE: ',mean_squared_error(df_test_y, svr_pred))



#%%
model_regr = RandomForestRegressor(max_depth=None, max_features=30, 
                                   n_estimators=60, n_jobs=2)
model_regr.fit(df_train_x, df_train_y)
regr_pred = model_regr.predict(df_test_x)

print('R2: ',r2_score(df_test_y, regr_pred))
print('MSE: ',mean_squared_error(df_test_y, regr_pred))

#R2:  0.4474328156780687
#MSE:  38.83160428535762
