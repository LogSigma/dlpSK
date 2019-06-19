#df_t = df_t[df_t.B != 'E']
#df_t = df_t[df_t.B != 'K']

#df_t = df_t[df_t.A != 'KD']

df_t = df_t.drop(['C2','C20','C73','C113','C77','C39','C42','C43','C66','C22'],axis=1)

df_t = df_t[df_t.ID != 12728]
#df_t = df_t[df_t.ID != 2165]
