'''
4 ways of building ensemble models:
1). blend using a blend_classifier and prob of submodels as features
2). weighed average of probs of submodels: 
        use CalibratedClassifierCV
        find weight by minimizing log_loss and get weighted average
        see eval_calibrated_clfs for example
3). class EnsembleClassifier: grid search on the weight for each submodel
4). get calibrated prob of each sub-models, then weighted average
'''
from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn import cross_validation
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np

'''
    clf_blend =  LogisticRegression()
    cv = cross_validation.KFold(len(X), n_folds=5) #, indices=False)  #use 2 for testrun
    results = blend_clfs2(X,y,cv,clfs)
    print results

    pred_proba = run_blend_predict(X, y, X_test,clfs)
    predicted_probs = [[index + 1, d[1]] for index, d in enumerate(pred_proba)]
'''
def blend_clfs2(X,y,cv,clfs, clf_blend = LogisticRegression()):
    '''
    INPUT:
        - clfs:  dictionary of classifiers, i.e.
            clfs = {'random forest gini': RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
             ,'extra tree gini':ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')}
    '''
    results = []
    n_clf = len(clfs.items())
    for train_val, test in cv:
        X2 = X[train_val]
        y2 = y[train_val]
        X_test = X[test]
        probs_test_blended_adj = run_blend_predict(X2, y2, X_test, clfs, clf_blend = clf_blend)
        results.append(logloss.llfun(y[test], probs_test_blended_adj[:,1]))
    return results 

def run_blend_predict(X_train_val, y_train_val, X_yunknown,clfs, clf_blend = LogisticRegression()):
    X2 = X_train_val
    y2 = y_train_val
    X_test = X_yunknown

    n_fold = 5 # testrun
    n_clf = len(clfs.items())

    cv_2nd = cross_validation.KFold(len(X2), n_folds=n_fold)
    probs_train_val = np.empty((X2.shape[0],n_clf))
    probs_test = np.empty((X_test.shape[0],n_clf))

    for i_clf, (des, clf) in enumerate(clfs.items()):
      probs_test_thisclf = np.zeros((X_test.shape[0],n_fold))  

      for i_fold, (train2, val2) in enumerate(cv_2nd):   
          clf.fit(X2[train2], y2[train2])    
          probs_train_val[val2,i_clf] = clf.predict_proba(X2[val2])[:,1]
          probs_test_thisclf[:,i_fold] = clf.predict_proba(X_test)[:,1]
      probs_test[:,i_clf] = probs_test_thisclf.mean(1)

    #clf_blend = LogisticRegression()
    print 'blend:', probs_train_val.shape, probs_test.shape, y2.shape, X2.shape   #debug

    clf_blend.fit(probs_train_val, y2)

    probs_test_blended = clf_blend.predict_proba(probs_test)
    probs_test_blended_adj = (probs_test_blended - probs_test_blended.min())/(probs_test_blended.max() - probs_test_blended.min())
    return probs_test_blended_adj


'''
example of build ensemble models
'''
    '''
    evaluate calibrated submodels using 3 weight scheme: a) same weight; b) top 3; c) optimized weights
    '''
    '''
    def init_calibrated_submodels(n_folds = 5):
        models = init_all_models()
        models_calibrated = [CalibratedClassifierCV(clf, cv=n_folds, method='isotonic') for clf in models] 
        model_names = [m.__class__.__name__ for m in models]
        return models_calibrated, model_names

    df, xnames, yname = load_df_from_pkl()

    models, model_names = init_calibrated_submodels()
    predictions_test, cv = build_ensemble_models(df, xnames, yname, models, model_names, n_folds = 5, cv_pkl=True, warmstart = False, w = None)
    
    weights = [[0.333,0.0,0.333,0.333]] * 5
    eval_ensemble_average_cv(predictions_test, df, cv, yname, weights)

    weights = find_weight_for_ensemble_cv(predictions_test, df, cv, yname)
    weights  = np.mean(weights, axis = 0)
    print weights
    weights = [list(weights)] * 5
    eval_ensemble_average_cv(predictions_test, df, cv, yname, weights)
    '''
def build_ensemble_models(df, xnames, yname, models, model_names, n_folds = 5, cv_pkl=True, warmstart = False, w = None, flag_print=True):
    '''
    build submodels, save as pickle files
    ensemble using weight w; only works for two classes 0/1
        INPUT
        - cv_pkl=True   read in cv from pkl file
        - warmstart = True   read in predicitons of submodels from pkl file
        - w: weight for each of the n_sub_models  
        - flag_print  write cv and test_predictions of each fold to pkl files
        OUTPUT
        - predictions_test  [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
        - cv
    '''
    n_models = len(models)
    run_name = 'eval_ensemble_models'
    if w is None:
        w = [1.0/n_models] * n_models
    w = np.array(w)

    cv_fname = 'data/kfold_cv.pkl'
    if cv_pkl:
        cv = pickle.load(open(cv_fname))
    else:
        cv = KFold(df.shape[0], n_folds= n_folds) 
        if flag_print:
            pickle.dump(cv,open(cv_fname, 'wb'))

    print model_names
    predictions_test = []  # 2d list of np array: i: cv; j: model j ;  np_array: n_sample x n_classes
    scores, scores_all, scores_all_train =  [], [], []  #scores of ensemble; scores of all test and train
    t00 = time.time() # time it     
    t0 = t00
    i = 0
    for i_cv, (train, test) in enumerate(cv):
        predictions_test.append([])
        predictions_train_thiscv = []
        X_train, y_train = df.iloc[train][xnames], df.iloc[train][yname]
        X_test, y_test = df.iloc[test][xnames], df.iloc[test][yname]
        X_train, scaler = scale_X(X_train, xnames)
        X_test = scale_X_transform(X_test , xnames, scaler)

        if len(train) + len(test) != df.shape[0]:
            print 'ERROR: K fold and dataset mis-match'
            return          
        scores_all.append([])
        scores_all_train.append([])     
        for j, clf in enumerate(models):
            model_name = model_names[j]
            probs, probs_train = get_predicted_prob(clf, i_cv, X_train, y_train, X_test, y_test, warmstart = warmstart, flag_print=flag_print, model_name = model_name)
            predictions_train_thiscv.append(probs_train)
            predictions_test[i_cv].append(probs)
            score_this_clf = roc_auc_score(y_test, probs[:, 1])
            print("testset score: %f" % score_this_clf)
            scores_all[i_cv].append(score_this_clf)
            scores_all_train[i_cv].append(roc_auc_score(y_train, probs_train[:,1]))
            t1 = time.time() # time it
            time_taken = (t1-t0)/60 
            print 'run %d %s time taken %.2fmin' % (i_cv, model_name, time_taken)
            t0 = t1 

        y_test_preds = np.array([p[:,1] for p in predictions_test[i_cv]])   
        y_test_ensemble = (y_test_preds.T).dot(w.T)
        y_test_ensemble = (y_test_ensemble - y_test_ensemble.min())/(y_test_ensemble.max() - \
        y_test_ensemble.min())

        y_train_preds = np.array([p[:,1] for p in predictions_train_thiscv])    
        y_train_ensemble = (y_train_preds.T).dot(w.T)
        y_train_ensemble = (y_train_ensemble - y_train_ensemble.min())/(y_train_ensemble.max() - \
        y_train_ensemble.min())     

        score_this_run = roc_auc_score(y_test, y_test_ensemble)
        score_this_run_train = roc_auc_score(y_train, y_train_ensemble)
        scores.append(score_this_run)
        scores_all[i_cv].append(score_this_run)
        scores_all_train[i_cv].append(score_this_run_train)     
        print "run %i combined score: %f" % (i_cv, scores[-1])
        print "       test: ", scores_all[i_cv]
        print "       train: ", scores_all_train[i_cv]
        print 
    print 
        
    scores_all = np.array(scores_all)
    scores_all_train = np.array(scores_all_train)
    t1 = time.time() # time it
    time_taken = (t1-t00)/60    
    print run_name,  ' finished in ', time_taken, ' minutes'
    print 'combined score: mean', (np.mean(scores), 'std:', np.std(scores))     
    print 'test set:', np.mean(scores_all, axis = 0)
    print 'train set:', np.mean(scores_all_train, axis = 0)
    return predictions_test, cv

def eval_ensemble_average_cv(predictions_test, df, cv, yname, weight, predictions_train = None):
    '''
    INPUT:
    - weight: n_fold * n_sub_models 
    - predictions_test: [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
    OUTPUT:
    - roc_test: numpy array: n_fold * (n_submodels + 1)
    - roc_train
    '''
    y_test_all = [] 
    roc_train = []
    for train, test in cv:
        y_test_all.append(df.iloc[test][yname])
    y_test_all = np.array(y_test_all)
    roc_test = eval_ensemble_kfold(predictions_test, y_test_all, weight)

    if predictions_train is not None:
        y_train_all = [] 
        for train, test in cv:
            y_train_all.append(df.iloc[train][yname])
        y_train_all = np.array(y_train_all)
        roc_train = eval_ensemble_kfold(predictions_test, y_test_all, weight)   
    return roc_test, roc_train  

def eval_ensemble_kfold(y_preds_nfold, y_actual_nfold, weight):
    '''
    INPUT:
        - y_preds_nfold: [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
        - y_actual_nfold: [i]: acutal of fold i
    '''
    roc = []
    for i_fold in xrange(len(y_preds_nfold)): #.shape[0]):
        roc.append([])
        y_actual = y_actual_nfold[i_fold]
        y_preds = np.array([p[:,1] for p in y_preds_nfold[i_fold]]) 
        w = np.array(weight)
        y_pred_ensemble = (y_preds.T).dot(w.T)
        y_pred_ensemble = (y_pred_ensemble - y_pred_ensemble.min())/(y_pred_ensemble.max() - \
        y_pred_ensemble.min())          
        roc[i_fold] = [roc_auc_score(y_actual,y_pred ) for y_pred in y_preds]
        roc[i_fold].append(roc_auc_score(y_actual, y_pred_ensemble))
        print roc[i_fold][-1],'         ', roc[i_fold]
    roc = np.array(roc)
    print np.mean(roc, axis = 0)
    return roc

def find_weight_for_ensemble_cv(predictions_test, df, cv, yname):
    '''
    INPUT: predictions_test: [i][j]: numpy array of [n, 1] : pred_proba for fold i and submodel j of class = 1
    '''
    weights = []
    for i, (train, test) in enumerate(cv):
        y_test = df.iloc[test][yname]
        y_test_preds = predictions_test[i]
        weights.append(find_weight_for_ensemble(y_test_preds, y_test))
    weights  = np.array(weights)
    print weights
    avg_weight  = np.mean(weights, axis = 0)
    print 'average weights to use:'
    return avg_weight, weights

def find_weight_for_ensemble(predictions, y_test):
    '''
    fing optimal weight to average predcitions
    INPUT:
        - predictions: n 
        - y_test
    OUTPUT:
        - list of k
    '''
    # source: https://www.kaggle.com/hsperr/otto-group-product-classification-challenge/finding-ensamble-weights
    #the algorithms need a starting value, right now we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    starting_values = [0.5]*len(predictions)

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight*prediction
        return log_loss(y_test, final_prediction)

    #adding constraints  and a different solver as suggested by user 16universe
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)

    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    return list(res['x'])


def grid_search_ensembleclassifier(iris):
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')

    params = {'logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200],}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
                % (mean_score, scores.std() / 2, params))

def tuning_weight_enseubleclassifier(clfs, X, y,voting='soft'):
    '''
    currently only works for 3 classifiers
    need to modify this. should do the prediction for each classifier once, then search for the best weights
    '''
    df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))

    i = 0
    for w1 in np.arange(1,3,0.5):
        for w2 in np.arange(1,3,0.5):
            for w3 in np.arange(1,3,0.5):                
                #if len(set((w1,w2,w3))) == 1: # skip if all weights are equal
                #    continue
                
                eclf = EnsembleClassifier(clfs=clfs, voting=voting, weights=[w1,w2,w3])
                scores = cross_validation.cross_val_score(
                                                estimator=eclf,
                                                X=X, 
                                                y=y, 
                                                cv=5, 
                                                scoring='accuracy',
                                                n_jobs=1)                
                df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
                i += 1                
    df = df.sort(columns=['mean', 'std'], ascending=False)
    return df

"""
modified from: 
http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
https://raw.githubusercontent.com/rasbt/mlxtend/master/mlxtend/classifier/ensemble.py
Soft Voting/Majority Rule classifier
This module contains a Soft Voting/Majority Rule classifier for
classification clfs.
"""
class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Soft Voting/Majority Rule classifier for unfitted clfs.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of classifiers.
      Invoking the `fit` method on the `VotingClassifier` will fit clones
      of those original classifiers that will be stored in the class attribute
      `self.clfs_`.

    voting : str, {'hard', 'soft'} (default='hard')
      If 'hard', uses predicted class labels for majority rule voting.
      Else if 'soft', predicts the class label based on the argmax of
      the sums of the predicted probalities, which is recommended for
      an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      Sequence of weights (`float` or `int`) to weight the occurances of
      predicted class labels (`hard` voting) or class probabilities
      before averaging (`soft` voting). Uses uniform weights if `None`.

    verbose : int, optional (default=0)
      Controls the verbosity of the building process.
        `verbose=0` (default): Prints nothing
        `verbose=1`: Prints the number & name of the clf being fitted
        `verbose=2`: Prints info about the parameters of the clf being fitted
        `verbose>2`: Changes `verbose` param of the underlying clf to
                     self.verbose - 2

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]

    clf : array-like, shape = [n_predictions]
      The unmodified input classifiers

    clf_ : array-like, shape = [n_predictions]
      Fitted clones of the input classifiers

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlxtend.sklearn import EnsembleClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = EnsembleClassifier(clfs=[clf1, clf2, clf3],
    ... voting='hard', verbose=1)
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """
    def __init__(self, clfs, voting='hard', weights=None, verbose=0):

        self.clfs = clfs
        self.named_clfs = {key: value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y):
        """ Fit the clfs.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.clfs_ = [clone(clf) for clf in self.clfs]

        if self.verbose > 0:
            print("Fitting %d classifiers..." % (len(self.clfs)))

        for clf in self.clfs_:

            if self.verbose > 0:
                i = self.clfs_.index(clf) + 1
                print("Fitting clf%d: %s (%d/%d)" %
                      (i, _name_estimators((clf,))[0][0], i, len(self.clfs_)))

            if self.verbose > 2:
                if hasattr(clf, 'verbose'):
                    clf.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((clf,))[0][1])

            clf.fit(X, self.le_.transform(y))
        return self


    def predict_all_clfs(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        pred_all_clfs = []
        clf_names = []
        for clf in self.clfs_:
            pred_all_clfs.append(clf.predict(X))
            clf_names.append(clf.__class__.__name__)
        maj = self.le_.inverse_transform(maj)
        return maj, pred_all_clfs, clf_names


    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """ Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def _predict_probas(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])

def try_calibrate(df, cv, predictions_test):
    n_folds = 5
    #df, cv, predictions_test = read_pred_cv(n_folds = n_folds)
    yname = 'dlqin2yrs' 

    print df.shape
    print df[yname].mean()
    xnames = df.columns.difference([yname, 'id'])
    print yname
    print xnames
    print df.describe().T
    print df.dtypes

    scores_all = []
    for i, (train, test) in enumerate(cv):
        preds_calibrated = np.zeros((len(test),5))
        scores_all.append([])
        y_test = df.iloc[test][yname]
        y_test_preds = predictions_test[i]
        for j, y_test_pred in enumerate(y_test_preds):
            score_this_clf = roc_auc_score(y_test, y_test_pred[:, 1])
            print("testset score: %f" % score_this_clf)
            scores_all[i].append(score_this_clf)    
            y_pred_scaled = (y_test_pred[:, 1] - y_test_pred[:, 1].min(axis=0)) \
              / (y_test_pred[:, 1].max(axis=0) - y_test_pred[:, 1].min(axis=0))
                    
            score_bin, empirical_prob = calibrate_prob(y_test, y_pred_scaled\
                , bins=200, normalize=False)
            preds_calibrated[:, j] = (sp.interp(y_pred_scaled, score_bin, empirical_prob)).T
            print roc_auc_score(y_test, preds_calibrated[:, j] )
        pred_ensemble = np.mean(preds_calibrated[:, [0,1,3,4]], axis=1)
        score_this_clf = roc_auc_score(y_test, pred_ensemble)
        scores_all[i].append(score_this_clf)
        print("testset score: %f" % score_this_clf)     
    print np.array(scores_all).mean(axis = 0)

'''
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import ensemble.EnsembleClassifier,  ensemble.tuning_weight
    print '\n\n ------ ensemble of different classifiers'
    models = [ linear_model.SGDClassifier(loss="log",penalty="elasticnet")  # hinge does not have prob
    , linear_model.LogisticRegression( tol= 0.001
    ,C=10
    ,class_weight= None)
    , SVC(kernel="linear", C=1,probability=True)
    #,MultinomialNB()
    ]
    print ensemble.tuning_weight(models, train_X, train_y)

    eclf = EnsembleClassifier(clfs=models,
     voting='soft', verbose=1
     , weights=[1.5,1.0,2.5])
    eclf = eclf.fit(train_X, train_y)

    test_y_pred_e, test_y_pred_all, clf_names = eclf.predict_all_clfs(test_X)
    print 'emsemble: ', accuracy_score(test_y, test_y_pred_e ) 
    for i, test_y_pred in enumerate(test_y_pred_all) :
        print clf_names[i], accuracy_score(test_y, test_y_pred )



'''