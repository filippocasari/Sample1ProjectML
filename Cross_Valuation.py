from sklearn.model_selection import GridSearchCV


def make_cross_evaluation(X_train, Y_train, clf, parameters, model_name):
    ########################################################################################################################
    #                                         CROSS-VALIDATION FOR MODEL SELECTION                                         #
    ########################################################################################################################

    model = clf
    grid_search_cv = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=5)
    grid_search_cv.fit(X_train, Y_train)
    print(model_name)
    best_params = grid_search_cv.best_params_

    print('Overall, the best values for parameters are ', str(best_params),
          ' since it leads to F1-score = ', grid_search_cv.best_score_)
    return best_params, grid_search_cv.best_estimator_