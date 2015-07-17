import sys
import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import isspmatrix
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

import pickle
import time
from collections import Counter
import random 

import pprint
import logging
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class ClassifierTestor(object):
    '''
    run all typical models using default setting and find the best 3
    Precondition:
        X, y
    '''
    metric_names = [
     'accuracy'
    ,'precision'
    ,'recall'
    ,'f1'
    ]

    def __init__ (self, estimators = None):
        if estimators == None:
            self.estimators = {
            'LogisticRegression':linear_model.LogisticRegression()
            , 'MultinomialNB':MultinomialNB()
            , 'SVC':SVC()
            , 'RF':RandomForestClassifier()
            , 'AdaBoostClassifier':AdaBoostClassifier() 
            , 'KNN':KNeighborsClassifier(5)
            ,'SVC_linear':SVC(kernel="linear", C=1)
            ,'SVC_rbf':SVC( kernel='rbf')
            , 'GaussianNB':GaussianNB()
            , 'GBC':GradientBoostingClassifier() 
            ,'LDA':LDA()
            ,'QDA':QDA()
            }
        else:
            self.estimators = estimators

    def __reorder(self, names):
        slow_methods = [ 'SVC_rbf', 'SVC', 'GBC']
        m_to_add = []
        for s in slow_methods:
            if s in names:
                i = names.index(s)
                del(names[i])
                m_to_add.append(s)
        names.extend(m_to_add)
        return names

    def fit(self, train_X, test_X, train_y, test_y):
        '''
        X: feature matrix
        '''
        # convert to full matrix, required for GBC

        estimator_names = self.estimators.keys()
        estimator_names = self.__reorder(estimator_names)

        print estimator_names
        print ', '.join(self.metric_names)

        if isspmatrix(train_X):
            train_X_full = train_X.todense()
            test_X_full = test_X.todense()
        else:
            train_X_full = train_X
            test_X_full = test_X

        metrics_all = []
        #for k, estimator in self.estimators.iteritems():

        for k in estimator_names:
            estimator = self.estimators[k]
            print '\n%s' % (k)
            t0 = time.time()
            try:
                if k in ['GaussianNB','GBC', 'LDA', 'QDA']:
                    train_X_f = train_X_full
                    test_X_f = test_X_full
                else:
                    train_X_f = train_X
                    test_X_f = test_X
                estimator = estimator.fit(train_X_f, train_y)
                test_y_pred = estimator.predict(test_X_f)
                metrics = [accuracy_score(test_y, test_y_pred) 
                ,precision_score(test_y, test_y_pred) #, average='binary')
                , recall_score(test_y, test_y_pred)
                , f1_score(test_y, test_y_pred)]
                str_metrics = ['%.3f' % (m) for m in metrics]
                print '%s %s'  %(k, str_metrics)
            except:
                print 'errror in model %s'  % (k)
                metrics = [np.nan] * len(metric_names)
            t1 = time.time() # time it
            metrics_all.append([k] + metrics + [(t1-t0)/60])
            print "finish in  %4.4fmin for %s " %((t1-t0)/60,k)
        self.df_score = pd.DataFrame(metrics_all, columns = ['model'] + self.metric_names + ['time'])
        print "\n"

    def score(self):
        self.df_score.sort('accuracy', ascending=False, inplace = True)
        return self.df_score

class ClassifierSelector(object):
    '''
    quick select of classifier hyper-parameter space
    features: X
    '''
    #   ['svm', 'rf','knn','lr','gbc']
    dict_model  = {'svm': SVC(),
    'rf': RandomForestClassifier()
    ,"knn": KNeighborsClassifier()
    ,'lr': linear_model.LogisticRegression()
    ,'gbc': GradientBoostingClassifier() 
    } 

    dict_params = {'svm':[
    {'clf__C': [1, 10], 'clf__kernel': ['linear']},
    {'clf__C': [1, 10]  # default gamma is 0.0 then 1/n_features
    , 'clf__kernel': ['rbf']},
    {'clf__kernel': ['poly'], 'clf__degree': [ 2, 3]}
    ], 
    'rf': [{"clf__n_estimators": [100, 250]}], 
    'knn': [{"clf__n_neighbors": [ 5, 10]}]
    , 'lr': [ {'clf__C': [1, 10]} ]
    , 'gbc': [{'clf__learning_rate': [ 0.1] # default 0.1
        , 'clf__n_estimators': [100, 300] #default 100
        }]
    }    

    def __init__(self, model_names, dict_params = {}):
        if model_names:
            self.models = dict([ (m, self.dict_model[m]) for m in model_names])
        else:
            self.models = self.dict_model 
        if dict_params:
            self.params = self.dict_params
        else:
            self.params = self.dict_params
        self.grid_searches = {}
        self.time_taken = {}

    def fit(self, x_train, y_train, cv=3, scoring=None,  refit=True, n_jobs=-1, verbose=1):
        print self.models.keys()

        if isspmatrix(x_train):
            train_X_full = x_train.todense()
        else:
            train_X_full = x_train

        for model_name in self.models:
            if model_name in ('gbc','lda', 'qda'):
                x_train_f = train_X_full
            else:
                x_train_f = x_train

            print '\n%s' % (model_name)
            print self.params[model_name]
            t0 = time.time()
            pipeline = Pipeline([
            ('clf', self.models[model_name])
            ]) 
            gs = GridSearchCV(pipeline, self.params[model_name]\
                , cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(x_train_f,y_train)
            print 'Best score  % .3f' % (gs.best_score_) , gs.best_params_,
            self.grid_searches[model_name] = gs    
            t1 = time.time() # time it
            self.time_taken[model_name] = (t1-t0)/60
        print 

    def score(self):
        lst_score = []
        for model_name, gs in  self.grid_searches.iteritems():
             gs_scores = gs.grid_scores_
             for grid_score in gs_scores:
                scores = grid_score.cv_validation_scores
                params =   grid_score.parameters
                lst_score.append([model_name, np.mean(scores), min(scores), max(scores), np.std(scores), params, self.time_taken[model_name]] )
        self.df_score = pd.DataFrame(lst_score, columns=['model','mean','min','max','std', 'param','minutes'])  
        self.df_score.sort('mean', inplace=True, ascending=False)
        return self.df_score 

    def score_predict(self, x_test, y_test):
        l = []
        if isspmatrix(x_test):
            test_X_full = x_test.todense()
        else:
            test_X_full = x_test

        for model_name in self.models:
            if model_name in ('gbc','lda', 'qda'):
                x_test_f = test_X_full
            else:
                x_test_f = x_test
        for model_name, gs in  self.grid_searches.iteritems():
            estimator = gs.best_estimator_
            #print estimator
            test_y_pred = estimator.predict(x_test_f)
            a = accuracy_score(y_test, test_y_pred)
            l.append([model_name, a])
        #print l
        return l

class ClassifierOptimizer(object):
    '''
    hyper parameter search for 1 classifier
    feature: can be text if added pipeline steps
    '''
    dict_model  = {'svm': SVC(),
    'rf': RandomForestClassifier(),
    "knn": KNeighborsClassifier(),
    'lr': linear_model.LogisticRegression()
    ,'gbc': GradientBoostingClassifier() 
    } # <== change here
    # pipeline parameters to automatically explore and tune
    
    dict_params = {'svm':[
    {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['linear']},
    {'clf__C': [1, 10, 100, 1000], 'clf__gamma': [0.1, 0.01, 0.001]  # default gamma is 0.0 then 1/n_features
    , 'clf__kernel': ['rbf']},
    {'clf__kernel': ['poly'], 'clf__degree': [1, 2, 3, 4]}
    ], 
    'rf': [{"clf__n_estimators": [250, 500, 1000]}], 
    'knn': [{"clf__n_neighbors": [1, 3, 5, 10, 20]}]
    , 'lr': [ {'clf__C': [0.0001, 0.001, 0.01, 0.5, 1, 10, 100, 1000],  # default: 1.0 inverse regularion strength
          'clf__class_weight': [None, 'auto'],
          'clf__tol': [ 1e-3, 1e-4, 1e-5, 1e-6]}]#, 1e-7] } ] # default 1e-4 0.0001
    , 'gbc': [{'clf__learning_rate': [0.8, 0.1, 0.05, 0.02, 0.01] # default 0.1
        , 'clf__max_depth': [3,6]  #default 3
        , 'clf__min_samples_leaf': [5, 10] #default 1
        , 'clf__max_features': [1.0, 0.3] #default None 1.0
        , 'clf__n_estimators': [300] #default 100
        }]
    }    

    #dict_params = {'svm':{'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']}}
    def __init__(self, clfname):
        self.clfname= clfname
        self.params = self.dict_params[self.clfname]
        self.clf_func = self.dict_model[self.clfname]
        self.parameters = None
        self.pipeline = None

    def get_clf(self):
        return self.clf_func

    def set_params(self, params):
        self.params = params

    def add_pipleline(self,lst_pipeline = [], params = None ):
        ''' 
        pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words = 'english', analyzer = analyzer11)),
        ('tfidf', TfidfTransformer()),
        ('chi2', SelectKBest(chi2)),
        ('clf', self.get_clf())
        ])   

        par_chi2 = {'chi2__k': [800, 1000, 1200, 1400, 1600, 1800, 2000]}
        '''
        self.pipeline = Pipeline(
            lst_pipeline + 
            [('clf', self.get_clf())])

        print("pipeline:", [name for name, _ in self.pipeline.steps])
        self.parameters = self.params

        if params:
            self.parameters = [ dict(d, **params) for d in self.params]
        print self.parameters

    def optimize(self, train_txt,  train_y):    
        '''
        train_txt: can be text if add_pipeline
        '''
        if self.parameters is None:
            self.parameters = self.params
        if self.pipeline is None:
            self.pipeline = Pipeline([
            ('clf', self.get_clf())])            

        print self.get_clf().__class__.__name__
        print("Performing grid search")
        print("parameters:")
        pprint.pprint(self.parameters)

        pnames = {}
        for p in self.parameters:
            for k in p:
                pnames[k] =1
        print pnames

        self.grid_search = GridSearchCV(self.pipeline, self.parameters, n_jobs=-1, verbose=1)

        t0 = time.time()
        self.grid_search.fit(train_txt, train_y)

        print("estimator Best score: %0.3f" % (self.grid_search.best_score_))
        print("Best parameters set:")
        best_parameters = self.grid_search.best_estimator_.get_params()
        #print best_parameters
        for param_name, v in best_parameters.iteritems():
            if param_name in pnames:
                print("\t%s: %r" % (param_name, v))
        #print self.grid_search.grid_scores_

        t1 = time.time() # time it
        print "finish in  %4.4fmin for %s " %((t1-t0)/60,'optimize')
        return self.grid_search

    def score_predict(self, test_txt, test_y):
        estimator = self.grid_search.best_estimator_
        test_y_pred = estimator.predict(test_txt)
        a = accuracy_score(test_y, test_y_pred)
        #print a
        return a, test_y_pred

    def get_best_estimator(self):
        '''
        dictionary of named steps in the pipeline
        '''
        return self.grid_search.best_estimator_.named_steps

    def get_best_classifier(self):
        '''
        instance of the classification method
        '''
        return self.grid_search.best_estimator_.named_steps['clf']
