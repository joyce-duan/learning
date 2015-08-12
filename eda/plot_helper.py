'''
to-do:  
plot_ybinary_xnumerical: add mean_line
plot_cat_var: fix x axis
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ybinary_xnumerical(df_X, y_binary, n_per_row=2, nbins = 20, figsize=(15,15)):
   """ 
   y: binary. pandas series
   x: continuous  pandas dataframe
   # run binning of numerical x vs. binary y
   ncol: number of plots per row
   plot probablity_density (%counts) and mean
   """
   n_datarows = df_X.shape[0]
   cols = df_X.columns
   yname = y_binary.name
   nplots = len(cols)
   nrows = nplots // n_per_row
   if nplots % n_per_row > 0:
      nrows +=1
   i = 0
   fig1, ax = plt.subplots(nrows, n_per_row, figsize=figsize)
   if nrows == 1 and n_per_row == 1:
       axs = [ax]
   else:
        axs = ax.flatten()
   df = pd.concat([df_X, y_binary], axis=1)
   for icol_x in xrange(len(cols)):
            #x_range = df[cols[icol_x]].max() - df[cols[icol_x]].max()
            nbins_this =  min(nbins,df[cols[icol_x]].nunique())

            n_decimal = 1
            n_xtick_labels = min(5, nbins)
            step = int(1.0 * nbins_this / n_xtick_labels)

            '''
            if df3[cols[i-1]].nunique() < nbins:
                nbins_this = df3[cols[i-1]].nunique()
            '''
            df['bins'], bins =pd.cut(df[cols[icol_x]], nbins_this, labels=False, retbins=True)

            x_plot =  bins[0:nbins_this] + (bins[2]-bins[1])/2
            x_plot = [round(v, n_decimal) for v in x_plot]
            x_tick_labels = [x if i % step == 0 else '' for i, x in enumerate(x_plot)]

            prob_den = pd.value_counts(df['bins'])#/n_datarows
            s_cnt = pd.Series([0] * nbins_this)
            df_cnt = pd.concat([s_cnt, prob_den], axis = 1)
            df_cnt.iloc[:,0] = df_cnt.apply(lambda x: x[0] if np.isnan(x[1]) else x[1], axis = 1)
            prob_den = df_cnt.iloc[:, 0]
            '''
            print cols[icol_x], nbins_this
            print prob_den.shape
            print type(x_plot), len(x_plot)
            '''
            ymean = df.groupby('bins')[yname].mean()
            s_mean = pd.Series([0.0] * nbins_this)
            df_tmp = pd.concat([s_mean, ymean], axis = 1)
            df_tmp.iloc[:,0] = df_tmp.apply(lambda x: x[0] if np.isnan(x[1]) else x[1], axis = 1)
            ymean = df_tmp.iloc[:, 0]
            axs[icol_x].bar(prob_den.index, prob_den, color='steelblue') #color='lightgray')#,label='prob. density',color='lightgray' )

            ax2 = axs[icol_x].twinx()
#              ax[irow,icol].plot( ymean, label = '%delinquency',color='r' )
            ax2.plot(np.arange(len(ymean))+0.5, ymean, label = 'pct delinquency',color='r' )
            axs[icol_x].set_title(cols[icol_x])

            plt.xticks(np.arange(0.5, len(ymean)+0.5), x_tick_labels )
            #plt.xticks(np.arange(len(ymean))+0.5)#, x_tick_labels )
            if icol_x == 0:
                 axs[icol_x].legend(loc='upper right')
                 axs[icol_x].set_ylabel('count', color='k')  # black
                 ax2.legend(loc='upper right')
                 axs[icol_x].set_ylabel('counts')
                 ax2.set_ylabel('pct delinquency', color='r')
                 #ax2.right_ax.set_ylabel('% delinquency')
            #i +=1
   plt.tight_layout()
   return fig1

def plot_cat_var(df, catx_names, y_name, figsize=(16,4), n_per_row=3, title = ''):

    nplots = len(catx_names)
    nrows_plot = nplots // n_per_row
    if nplots % n_per_row > 0:
        nrows_plot = nrows_plot + 1
    fig, axs = plt.subplots(nrows_plot, n_per_row, figsize=figsize)
    axs_flat = axs.flatten()
    for i, cat in enumerate(catx_names):
        ax = axs_flat[i]
        ax2 = ax.twinx()
        avg_pos = df.groupby(cat)[y_name].mean()
        avg_pos.plot(kind='bar', ax=ax2, color='r', alpha = 0.5, position = 0)
        
        cnt = df.groupby(cat)[y_name].sum()

        cnt.plot(kind='bar', color= 'darkgray', ax = ax, position = 1)
        ax.set_title(cat)
        width = 0.4

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
 
def plot_cat_var(df, catx_names, y_name):
    fig, axs = plt.subplots(2, len(catx_names), figsize=(10,6))


    for i, cat in enumerate(catx_names):
        avg_pos = df.groupby(cat)[y_name].mean()
        avg_pos.plot(kind='bar', ax=axs[0,i], color='r', alpha = 0.5)
        cnt = df.groupby(cat)[y_name].sum()
        cnt.plot(kind='bar', ax=axs[1,i])
    plt.tight_layout()
    plt.suptitle('%Fraud and Total Counts')
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    source: http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html

    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




