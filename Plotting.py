import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import learning_curve


def plot_lc_curve(X, Y, title, i=None, clf=None):
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.

    if i != None:
        title = title + " \nnum of k: " + str(i)
    plot_learning_curve(clf, title, X, Y, ylim=(0.01, 0.6), n_jobs=-1)

    plt.show()


def metrics_results(y_test, y_predicted, model_name):
    print('/-------------------------------------------------------------------------------------------------------- /')
    print('RESULTS OF THE %s CLASSIFIER' % model_name)
    print('/-------------------------------------------------------------------------------------------------------- /')
    print('Accuracy is ', sklearn.metrics.accuracy_score(y_test, y_predicted, sample_weight='weighted'))
    print('Precision is ', precision_score(y_test, y_predicted, sample_weight='weighted'))
    print('Recall is ', recall_score(y_test, y_predicted, sample_weight='weighted'))
    print('F0-Score is ', f1_score(y_test, y_predicted, sample_weight='weighted'))
    print('AUC is ', roc_auc_score(y_test, y_predicted, sample_weight='weighted'))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
