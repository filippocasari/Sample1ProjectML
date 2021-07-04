
from matplotlib import pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve


def evaluating_models( y_test, y_predicted, name):
    ########################################################################################################################
    # #######                                              EVALUATION                                              ####### #
    ########################################################################################################################

    precision = dict()
    recall = dict()
    for i in range(4):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_predicted[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve for "+name)
    plt.show()

    # roc curve
    fpr = dict()
    tpr = dict()

    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i],
                                      y_predicted[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve for "+name)
    plt.show()