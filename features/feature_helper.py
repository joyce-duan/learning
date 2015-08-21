import pandas as pd

'''
def cap_value(x,cap):
    return cap if x > cap else x
'''
def fill_na(df, xnames, v = 0):
  for c in xnames:
    df[c].fillna(v, inplace=True)
  return df 
  
def cap_df(df, xnames, pct = 0.999, postfix=''):
   """
   cap specified columns xname to pctile
   na stays as na
   """
   df2 = pd.DataFrame()
   if postfix == '':
      postfix = '_capped'
   for c in xnames:
     cap = df[c].quantile(pct)
     df2[c+postfix]=df[c].apply(lambda x: cap if x > cap else x ) #cap_value(x, cap_at))
   return df2
