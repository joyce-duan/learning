from configobj import ConfigObj
#config = ConfigObj('config')

#ml_home = config.get(
#    'ml_home', '/')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report

def get_confusion_rates(cm):
    '''
    - INPUT
        cm: confusion_matrix output from sklearn
    - OUTPUT:
                             actual
                          true     false    
            pred  true
                  false
    '''
    [[tn, fp], [fn, tp]] = cm

    N = fp + tn  # actual
    P = tp + fn

    tpr = tp / P
    fpr = fp / N
    fnr = fn / P
    tnr = tn / N

    return np.array([[tpr, fpr], [fnr, tnr]])


def print_classification_report(test_y, test_y_pred):
    print confusion_matrix(test_y, test_y_pred)
    print classification_report(test_y, test_y_pred, labels=[0, 1])

def print_top_features_pos_neg(clf_coefficients, features, k=20):
    sorted_idx = clf_coefficients.argsort() # coef: 2d array 
    print 'top %i features for positives:' %(k)
    print zip(features[sorted_idx[:-k:-1]], clf_coefficients[ sorted_idx[:-k:-1]])
    print 'bottom %i features for negatives:' %(k)
    print zip(features[sorted_idx[:k]], clf_coefficients[ sorted_idx[:k]])

def print_top_features(X, stats_func, features, label, k=20):
    '''
        X: 2d array
    '''
    stats = stats_func(X)
    print X.shape, stats.shape
    sorted_idx = stats.argsort() # 
    print 'top %i features for %s:' %(k, label)
    lst = zip(features[sorted_idx[:-k:-1]], stats[ sorted_idx[:-k:-1]])
    for e in lst:
        print e


def print_fp_fn_samples(test_y, test_y_pred, test_txt):

    i_lst_fp = [i for i in xrange(len(test_y)) if test_y[i] == 0 and test_y_pred[i] == 1]
    i_lst_fn = [i for i in xrange(len(test_y)) if test_y[i] == 1 and test_y_pred[i] == 0]
    print '\nfalse positive'
    for i in i_lst_fp[:20]:
        print i, test_y[i], ':', test_txt[i]
    print 'false negative'
    for i in i_lst_fn[:20]:
        print i, test_y[i], ':', test_txt[i]
