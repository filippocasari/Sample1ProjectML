from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def Logistic_regression(X_train, Y_train, X_test, Y_test):

    log_regr = LogisticRegression(class_weight='balanced', multi_class='multinomial',
                                  max_iter=10000, solver='saga')
    log_regr.fit(X=X_train, y=Y_train)
    y_pred = log_regr.predict(X=X_test)
    print(classification_report(Y_test, y_pred))