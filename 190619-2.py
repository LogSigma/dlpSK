#df_t = df_t[df_t.B != 'E']
#df_t = df_t[df_t.B != 'K']

#df_t = df_t[df_t.A != 'KD']

df_t = df_t.drop(['C2','C20','C73','C113','C77','C39','C42','C43','C66','C22'],axis=1)

df_t = df_t[df_t.ID != 12728]
#df_t = df_t[df_t.ID != 2165]

model_gboost = GradientBoostingRegressor(loss='huber',n_estimators = 110, max_features='sqrt',
                                         max_depth=7, learning_rate=0.05)
model_gboost.fit(df_train_x, df_train_y)
gboost_pred = model_gboost.predict(df_test_x)

print('R2: ',r2_score(df_test_y, gboost_pred))
print('MSE: ',mean_squared_error(df_test_y, gboost_pred))

#R2:  0.5537570481212277
#MSE:  31.35967935510029
