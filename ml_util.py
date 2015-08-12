import random 
import numpy as np
import pandas as pd

def sample_df(df, fname, n_samples, func_read):
	f_sample_name = fname.split('.')[0] + '_sample.' + fname.split('.')[1]

def random_sample(feature_lst, y, k=200):
	'''
	random sample k for features and y
		- INPUT:
			feature_list:  list of features n_data_rows x k_features  (can be string etc.)
			y:  
	'''
	print len(y), k
	lst_i = random.sample(xrange(len(y)), k)
	feature_lst = [feature_lst[i] for i in lst_i]
	y = [y[i] for i in lst_i]
	return feature_lst, y

def count_null(df):
   df_lng = pd.melt(df)
   null_var = df_lng.value.isnull()
   df_null_cnt = pd.crosstab(df_lng.variable, null_var)
   #print df_null_cnt.apply(lambda x: int(100.0 * x[0]/(x[1]+x[0])), axis = 1)
   df_null_cnt['pct_null'] = df_null_cnt.apply(lambda x: int(100.0 * x[1]/(x[1]+x[0])), axis = 1)
   df_null_cnt.columns = ['not_null', 'null', 'pct_null']
   #return pd.crosstab(df_lng.variable, null_var)
   return df_null_cnt

def describe_df(df):
	df_stats = df.describe(include='all')
	n_uniques = [[df[c].nunique()] for c in  df.columns]
	df_unique_cnts = pd.DataFrame(np.array(n_uniques).T, columns=df.columns, index = ['unique_counts'])
	df_stats = pd.concat([df_stats,df_unique_cnts])
	#print df_stats
	stats_names = ['count','unique_counts','min','max','mean','50%','std','25%','75%']
	return df_stats.ix[stats_names]

