import random 

def random_sampe(feature_lst, y, k=200):
	'''
	random sample k for features and y
		- INPUT:
			feature_list:  list of features n_data_rows x k_features  (can be string etc.)
			y:  
	'''
	lst_i = random.sample(range(len(y)), k)
	feature_lst = [feature_lst[i] for i in lst_i]
	y = [y[i] for i in lst_i]
	return feature_lst, y
