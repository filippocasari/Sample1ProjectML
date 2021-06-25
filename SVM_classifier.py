import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_circles
from sklearn.metrics import classification_report
from sklearn.svm import SVC





def SVM_classifier(X_train, y_train, X_test,Y_test):

    rbf_svc = SVC(kernel='rbf', class_weight='balanced', max_iter=10000).fit(X_train, y_train)
    y_pred=rbf_svc.predict(X_test)
    print(classification_report(Y_test,y_pred))


