from sklearn.model_selection import GridSearchCV



def make_cross_evaluation(X_train, X_test, Y_train, Y_test, clf, parameters, model_name):
    ########################################################################################################################
    #                                         CROSS-VALIDATION FOR MODEL SELECTION                                         #
    ########################################################################################################################

    model = clf
    clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=5)
    clf.fit(X_train, Y_train)
    print(model_name)
    print('Overall, the best value for parameter k is ', clf.best_params_.get('max_depth'),
          ' since it leads to F1-score = ', clf.best_score_)
