#%%

import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import r2_score, mean_squared_error

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

for i in range(1,117):
	plt.scatter(df_t['END_TM_INT'], df_t['C{0}'.format(i)])
	plt.title('C{0}'.format(i))
	plt.show()

#%%
#데이터 전처리
df_t = df_t[df_t.B != 'E']
df_t = df_t[df_t.B != 'K']

df_t = df_t[df_t.A != 'KD']

df_t = df_t[df_t.ID != 12728]
df_t = df_t[df_t.ID != 2165]

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
gbm = xgb.XGBRegressor(max_depth=7, n_estimators=110, learning_rate=0.05)
gbm.fit(df_train_x, df_train_y)
predictions = gbm.predict(df_test_x)

print('R2: ',r2_score(df_test_y, predictions))
print('MSE: ',mean_squared_error(df_test_y, predictions))

## R2:  0.522950572277962
## MSE:  33.55045033901902
## 제출 결과 MSE:    44.61722
