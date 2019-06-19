import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('D:/DataSet_accuracy.csv', sep=",")

df_t = df
df_t['END_TM'] = pd.to_datetime(df_t['END_TM'], format='%Y-%m-%d %p %H:%M:%S')

df_t['END_TM'] = (pd.to_numeric(df_t['END_TM']-df_t['END_TM'][4])/1000000000).astype(int)

df_t = df_t.set_index('ID')

df_t = pd.get_dummies(df_t, columns=['A','B'])

df_m1 = df_t[df_t['Validation']==0]

#df_m1 = df_m1.T.fillna(df_m1.mean(axis=1)).T
df_m1 = df_m1.fillna(df_m1.median())

df_m1_y = df_m1['Y']

#df_train = df_m1.sample(frac=0.7, random_state=0, replace=False)
df_train = df_m1[:9430]
df_test = df_m1.drop(df_train.index.tolist())


df_train_y = df_train['Y']
df_test_y = df_test['Y']

df_train_x = df_train.drop(['Y','END_TM','Validation'], axis=1)
df_test_x = df_test.drop(['Y','END_TM','Validation'],axis=1)

#%%
model = GradientBoostingRegressor(n_estimators = 200, random_state=0, max_depth =5)
fit = model.fit(df_train_x, df_train_y)
g_df_test_pre = pd.DataFrame(fit.predict(df_test_x))
print('R2: ',r2_score(df_test_y, g_df_test_pre))
print('MSE: ',mean_squared_error(df_test_y, g_df_test_pre))

##R2:  -0.04904487606853736
##MSE:  68.31041275044969
