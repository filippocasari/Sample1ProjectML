from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, f1_score, recall_score, precision_score, \
    accuracy_score


def valuating_models(models_name, y_test, y_predicted):
    ########################################################################################################################
    # #######                                              EVALUATION                                              ####### #
    ########################################################################################################################
    for i in range(len(models_name)):
        print(
            '/-------------------------------------------------------------------------------------------------------- /')
        print('RESULTS OF THE %s CLASSIFIER' % models_name[i])
        print(
            '/-------------------------------------------------------------------------------------------------------- /')
        print('Accuracy is ', accuracy_score(y_test, y_predicted[i]))
        print('Precision is ', precision_score(y_test, y_predicted[i],average='macro'))
        print('Recall is ', recall_score(y_test, y_predicted[i],average='macro'))
        print('F1-Score is ', f1_score(y_test, y_predicted[i],average='macro'))
        print('AUC is ', roc_auc_score(y_test, y_predicted[i],average='macro'))

        fpr, tpr, _ = roc_curve(y_test, y_predicted[i])
        plt.figure('ROC')
        plt.title('ROC curve')
        plt.plot(fpr, tpr, label=models_name[i], linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        precision, recall, _ = precision_recall_curve(y_test, y_predicted[i])
        plt.figure('PR')
        plt.title('P-R curve')
        plt.plot(recall, precision, label=models_name[i], linewidth=4)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])

    plt.show()