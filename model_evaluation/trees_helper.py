import numpy as np
import matplotlib.pyplot as plt

colors = [('#d7191c', '#2c7bb6'), ('#fdae61', '#abd9e9')] # (test, train)
def plot_staged_score(est, X_test, X_train, y_test, y_train, func_score, ax=None, label='', ylabel ='auc', train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
    """
     plot n_estimators vs. score  for ``est``, use ``X_test`` and ``y_test`` for test error. 
    """
    n_estimators = len(est.estimators_)
    test_score, train_score  = np.empty(n_estimators), np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict_proba(X_test)):
       #test_score[i] = est.loss_(y_test, pred)
       test_score[i] = 1- func_score(y_test, pred[:,1])

    for i, pred in enumerate(est.staged_predict_proba(X_train)):
       #test_score[i] = est.loss_(y_test, pred)    
       train_score[i] = 1- func_score(y_train, pred[:,1])

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

    ax.plot(np.arange(n_estimators) + 1, test_score, color=test_color, label='Test %s' % label,
             linewidth=2, alpha=alpha)
    #ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color,
    #         label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, train_score, color=train_color,
             label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('n_estimators')
    #ax.set_ylim((0.5, 1.05))
    return test_score, ax

   

def plot_staged_score_rf(est, X_test, X_train, y_test, y_train, func_score, ax=None, label='', ylabel ='auc', train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
    """
     plot n_estimators vs. score  for ``est``, use ``X_test`` and ``y_test`` for test error. 
    """
    n_estimators = len(est.estimators_)
    test_score, train_score  = np.empty(n_estimators), np.empty(n_estimators)
    y_train_pred_cum , y_test_pred_cum = np.zeros((len(y_train), 2)), np.zeros((len(y_test), 2))

    for i, estimator in  enumerate(est.estimators_):
        y_train_pred_cum = y_train_pred_cum + estimator.predict_proba(X_train)
        pred_train = y_train_pred_cum / (i+1)
        train_score[i] = 1- func_score(y_train, pred_train[:,1])

        y_test_pred_cum = y_test_pred_cum + estimator.predict_proba(X_test)
        pred_test = y_test_pred_cum / (i+1)
        test_score[i] = 1- func_score(y_test, pred_test[:,1])      

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

    ax.plot(np.arange(n_estimators) + 1, test_score, color=test_color, label='Test %s' % label,
             linewidth=2, alpha=alpha)
    #ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color,
    #         label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, train_score, color=train_color,
             label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('n_estimators')
    #ax.set_ylim((0.5, 1.05))
    return test_score, ax
 