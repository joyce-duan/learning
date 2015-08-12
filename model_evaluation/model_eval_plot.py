import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

from metrics_helper import get_confusion_rates

def plot_feature_importance_gbc(clf, feature_names, topk = 25, figsize = (50,70) ):
    #topk = 25
    fig = plt.figure(figsize = figsize)
    importances = clf.feature_importances_ 
    sorted_idx = np.argsort(importances)[-topk:]
    #sorted_idx = sorted_idx[::-1]
    padding = np.arange(len(sorted_idx)) + 0.5
    #plt.barh(padding, importances[sorted_idx], align='center')
    plt.barh(padding, importances[sorted_idx],\
       color="b", alpha = 0.5, align="center")    
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.yticks(padding, feature_names[sorted_idx])
    #plt.show()
    return fig

def plot_feature_importance(rf, feature_names, topk = 25, errorbar=False, figsize = (50,70) ):
    #topk = 25
    fig = plt.figure(figsize = figsize)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)    
    sorted_idx = np.argsort(importances)[-topk:]
    padding = np.arange(len(sorted_idx)) + 0.5
    #plt.barh(padding, importances[sorted_idx], align='center')
    if errorbar: 
        plt.barh(padding, importances[sorted_idx],\
            color="b", alpha = 0.5, xerr=std[sorted_idx], align="center")   
    else:
        plt.barh(padding, importances[sorted_idx],\
        color="b", alpha = 0.5, align="center")  
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.yticks(padding, feature_names[sorted_idx])
    plt.show()
    #plt.plot()
    return fig

def profit_curve(y_actual, y_pred_lst, cb): 
    '''
        - INPUT
            y_actual: y is binary: n x 1
            y_pred_lst: output from k estimators  k x n 
            cb: cost beneift matrix
                avg_loss_per_fraud = -20  # false negative
                avg_cost_fp = -5  # false positive
                # cost - benefit matrix
                                     actual
                                positive,  negative
                    cb = np.array ([[0, avg_cost_fp],
                                    [avg_loss_per_fraud, 0]])
        - OUTPUT: figure, profits [(percentage, profit)]  k x 2
    '''
    pos = np.sum(y_actual == 1) / len(y_actual)
    neg = 1 - pos
    class_probs = np.array([pos, neg])
    fig = plt.figure()
    profits = []
    for probabilities  in y_pred_lst:
        indicies = np.argsort(probabilities)[::-1]

        profit = []
        for i in xrange(len(indicies)):
            pred_false = indicies[i:]
            y_predict = np.ones(len(indicies))
            y_predict[pred_false] = 0
            rates = get_confusion_rates(confusion_matrix(y_actual, y_predict))
            profit.append(np.sum(class_probs * rates * cb))
            
        print max(profit)
        i_max_profit2 = np.argmax(profit)
        probs = np.array(probabilities)
        print '%i max profit: %.2f  cutoff: %.2f ' % (i_max_profit2, profit[i_max_profit2], probs[i_max_profit2])

        percentages = np.arange(len(indicies)) / len(indicies) * 100
        plt.plot(percentages, profit, label= 'label')
        profits.append([percentages, profit])

    plt.legend(loc="lower right")
    plt.title("Profits of classifiers")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    #plt.ylim(20)
    #plt.show()
    return fig, profits

def plot_roc(y, y_pred_prob):
    '''
    for binary classification
    '''
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % ( roc_auc))    

def plot_roc_cv(classifier, X, y, n_folds, cv):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
