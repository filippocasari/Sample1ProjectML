from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def make_cross_validation(X_train, Y_train, clf, parameters, model_name):
    ########################################################################################################################
    #                                         CROSS-VALIDATION FOR MODEL SELECTION                                         #
    ########################################################################################################################

    model = clf
    grid_search_cv = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=5)
    grid_search_cv.fit(X_train, Y_train)
    print(model_name)
    best_params = grid_search_cv.best_params_

    if("KNN" in model_name):

        k_range = range(1, 101)

        # list of scores from k_range
        k_scores = []

        # 1. we will loop through reasonable values of k
        for k in k_range:
            # 2. run KNeighborsClassifier with k neighbours
            knn = KNeighborsClassifier(n_neighbors=k)
            # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
            scores = cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy')
            # 4. append mean of scores for k neighbors to k_scores list
            k_scores.append(scores.mean())
        # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
        plt.plot(k_range, k_scores)
        plt.xlabel('Value of K for '+model_name)
        plt.ylabel('Cross-Validated Accuracy')
        plt.show()
    #print(k_scores)
    print('Overall, the best values for parameters are ', str(best_params),
          ' since it leads to accuracy = ', grid_search_cv.best_score_)
    return best_params, grid_search_cv.best_estimator_