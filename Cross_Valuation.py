from sklearn.model_selection import GridSearchCV


def make_cross_evaluation(X_train, Y_train, clf, parameters, model_name):
    ########################################################################################################################
    #                                         CROSS-VALIDATION FOR MODEL SELECTION                                         #
    ########################################################################################################################

    model = clf
    clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=5)
    clf.fit(X_train, Y_train)
    print(model_name)
    best_params = clf.best_params_

    print('Overall, the best values for parameters are ', str(best_params),
          ' since it leads to F1-score = ', clf.best_score_)
