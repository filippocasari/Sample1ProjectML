from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, f1_score, recall_score, precision_score, \
    accuracy_score, auc, RocCurveDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier


def valuating_models(models_name, y_test, y_predicted):
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
    plt.title("precision vs. recall curve for Neural network")
    plt.show()

    plt.show()