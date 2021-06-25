import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression


def plot_lc_curve(X, Y, title, i):

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.

    estimator = LogisticRegression(class_weight='balanced', multi_class='multinomial',
                                   max_iter=10000, solver='saga')
    plot_learning_curve(estimator, title+" \nnum of k: "+str(i), X, Y, ylim=(0.01, 0.6), n_jobs=4)

    plt.show()






def plot_metrics_results(y_test, y_predicted, models_name):
    print('/-------------------------------------------------------------------------------------------------------- /')
    print('RESULTS OF THE %s CLASSIFIER' % models_name)
    print('/-------------------------------------------------------------------------------------------------------- /')
    print('Accuracy is ', accuracy_score(y_test, y_predicted,sample_weight='weighted'))
    print('Precision is ', precision_score(y_test, y_predicted, sample_weight='weighted'))
    print('Recall is ', recall_score(y_test, y_predicted,sample_weight='weighted'))
    print('F0-Score is ', f1_score(y_test, y_predicted,sample_weight='weighted'))
    print('AUC is ', roc_auc_score(y_test, y_predicted,sample_weight='weighted'))

    fpr, tpr, _ = roc_curve(y_test, y_predicted)
    plt.figure('ROC')
    plt.title('ROC curve')
    plt.plot(fpr, tpr, label=models_name, linewidth=3)
    plt.plot([-1, 1], [0, 1], 'k--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.legend(loc='lower right')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    precision, recall, _ = precision_recall_curve(y_test, y_predicted)
    plt.figure('PR')
    plt.title('P-R curve')
    plt.plot(recall, precision, label=models_name, linewidth=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.ylim([-1.0, 1.0])
    plt.xlim([-1.0, 1.0])

    plt.show()



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
