import pandas as pd
import numpy as np
#def score_grid_search(grid_search):
def get_grid_search_score_df(grid_search):
	lst_score = []
	gs_scores = grid_search.grid_scores_
	for grid_score in gs_scores:
		scores = grid_score.cv_validation_scores
		params =   grid_score.parameters
		lst_score.append([np.mean(scores), min(scores), max(scores)\
			, np.std(scores), params] )
	df_score = pd.DataFrame(lst_score, columns=['mean','min','max','std', 'param'])  
	df_score.sort(['mean'], inplace=True, ascending=False)
	return df_score

# Utility function to report best scores
def grid_search_report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def randomized_search(df, xnames, yname):
	# build a classifier
	clf = RandomForestClassifier(n_estimators=100)
	# specify parameters and distributions to sample from
	param_dist = {  #"max_depth": [3, 5, 7, 10, 15],
				   "max_depth": sp.stats.randint(3,15),
	              "max_features": sp.stats.randint(1, 11),
	              "min_samples_split": sp.stats.randint(1, 11),
	              "min_samples_leaf": sp.stats.randint(2, 11),
	              'class_weight':[None, {0:1, 1:1}, {0:1,1:5}, {0:1,1:10}, 'auto'],
	              "criterion": ["gini", "entropy"]}
	# run randomized search
	n_iter_search = 20
	grid_search = RandomizedSearchCV(clf, param_distributions=param_dist,
	                          verbose = 1, n_iter=n_iter_search, cv = 5, scoring= 'roc_auc',  n_jobs=1 )

	start = time.time()
	grid_search.fit(df[xnames], df[yname])
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings." % ((time.time() - start), n_iter_search))
	report(grid_search.grid_scores_, n_top = 5)
	return grid_search

def randomized_search_gbc(df, xnames, yname):
	# build a classifier
	n_features = len(xnames)
 	clf = GradientBoostingClassifier(n_estimators=200)
	# specify parameters and distributions to sample from
	param_dist = { 'learning_rate': sp.stats.uniform(0.01, 0.1),
				   "max_depth": sp.stats.randint(3,7),
	              #"max_features": sp.stats.uniform(0.1, 1.0),
	              'max_features': sp.stats.randint(1, 11),
	              "min_samples_split": sp.stats.randint(3, 20),
	              "min_samples_leaf": sp.stats.randint(3, 18),
		}
	# run randomized search
	n_iter_search = 20
	grid_search = RandomizedSearchCV(clf, param_distributions=param_dist,
	                          verbose = 1, n_iter=n_iter_search, cv = 5, scoring= 'roc_auc',  n_jobs=1 )

	start = time.time()
	grid_search.fit(df[xnames], df[yname])
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
	      " parameter settings." % ((time.time() - start), n_iter_search))
	report(grid_search.grid_scores_, n_top = 5)
	return grid_search

