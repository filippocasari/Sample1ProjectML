from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def Logistic_regression(X_train, Y_train, X_test, Y_test):

    log_regr = LogisticRegression(class_weight='balanced', multi_class='multinomial',
                                  max_iter=10000, solver='saga')
    log_regr.fit(X=X_train, y=Y_train)
    y_pred = log_regr.predict(X=X_test)
    print("report with shape of X:"+str(X_train.shape))
    print(accuracy_score(Y_test, y_pred))
    return log_regr, accuracy_score(Y_test, y_pred)