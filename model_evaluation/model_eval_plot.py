import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from metrics_helper import get_confusion_rates

def plot_feature_importance(rf, feature_names, topk = 25, figsize = (50,70) ):
    #topk = 25
    fig = plt.figure(figsize = figsize)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)    
    sorted_idx = np.argsort(-1.0* importances)[:topk]
    padding = np.arange(len(sorted_idx)) + 0.5
    #plt.barh(padding, importances[sorted_idx], align='center')
    plt.barh(padding, importances[sorted_idx],\
       color="b", alpha = 0.5, yerr=std[sorted_idx], align="center")    
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