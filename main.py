import os
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, chi2
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import Cross_Validation
import Discretization
import EDA
import Plotting
import Evaluation
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from time import process_time, time

# ----------------------------------GLOBAL VARIABLES----------------------------------------------
count_features = False
discretization_bool = True
problem_is_binarized = False
normalization = False
standardization = False
preproc = True
np.random.seed(31415)

# Parameters for in Cross Validation
parameters_dt = {'max_depth': list(range(1, 75, 2)), "criterion": ["gini", "entropy"],
                 "splitter": ["best", "random"]}
parameters_knn = {'n_neighbors': list(range(1, 101, 2)), "weights": ["uniform", "distance"], "p": [1, 2]}


# ---------------------------------------------------------------------------------------------------
# used algorithms :  DecisionTree, KNN

# --------------------------------------------BEGIN OF CUSTOM FUNCTION------------------------------
def plot_metrics_for_each_features(df):
    # figures = [] # array to save plot
    X = df.drop(columns='Baselinehistological staging')  # get X
    names_cols = X.columns  # get columns names
    # try to create a new directory where save our plots
    try:
        os.makedirs("./plots")
    except FileExistsError:
        # directory already exists
        pass
    # for each features, plot a count plot
    for i in names_cols:
        sns.catplot(data=df, x=i, kind="count", hue='Baselinehistological staging')
        plt.gcf().subplots_adjust(bottom=0.10)
        plt.show()


# function to split easy
def splitting_train_test(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=0, train_size=0.8
    )
    return X_train, X_test, Y_train, Y_test


# function to find the best new array of X composed by "K" features
def select_best_features_with_kbest(X_train, X_test, Y_train, Y_test, title, clf):
    indexes = []
    accuracy_scores = []
    max_accuracy = 0
    k_migliore = 0
    # start trying performance of the model with "k" features
    for i in range(12, len(X_train.columns)):
        title = "Learning Curves with " + title
        selector = SelectKBest(chi2, k=i)
        selector.fit(X_train, Y_train)
        # scores = selector.scores_
        # scores /= scores.max()
        X_train_new = selector.transform(X_train)
        X_test_new = selector.transform(X_test)
        indexes.append(i)
        accuracy = accuracy_score(Y_test, (clf.fit(X_train_new, Y_train)).predict(X_test_new))
        if (accuracy > max_accuracy):  # get model with max accuracy
            max_accuracy = accuracy
            k_migliore = i
            temp1 = selector.transform(X_train)
            temp2 = selector.transform(X_test)

        accuracy_scores.append(accuracy)
        print(str(accuracy) + " with " + str(
            i) + " features")
        print(" X with selection K BEST \n" + str(X.columns.values[selector.get_support()]))
        # Plotting.plot_lc_curve(X_train_new, Y_train, title, i, clf) # plot learning curve

    # plot function Accuracy = f(K)
    sns.barplot(x=indexes, y=accuracy_scores)
    plt.xlabel("K features")
    plt.ylim(0, 0.4)
    plt.ylabel("accuracy score")
    plt.show()
    X_train_new_final = temp1
    X_test_new_final = temp2
    return k_migliore, max_accuracy, X_train_new_final, X_test_new_final


# feature selection with Variance Threshold
def feature_selection_varince(X_train, X_test):
    constant_filter = VarianceThreshold(threshold=0.0)
    constant_filter.fit(X_train)
    constant_columns = [column for column in X_train.columns
                        if column not in
                        X_train.columns[constant_filter.get_support()]]
    X_train = constant_filter.transform(X_train)
    X_test = constant_filter.transform(X_test)
    for column in constant_columns:
        print("Removed ", column)
    return X_train, X_test


# another method to select "x" features, starting from the model
def select_from_model(X, Y, clf, title):
    feature_names = X.columns
    X_train, X_test, Y_train, Y_test = splitting_train_test(X, Y)
    clf.fit(X_train, Y_train)
    model = SelectFromModel(clf, prefit=True)
    mask = model.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    X_new = pd.DataFrame(data=model.transform(X), columns=new_features)
    X_train, X_test, Y_train, Y_test = splitting_train_test(X_new, Y)
    print(" X with selection from model \n" + str(X_new))
    title = "Learning Curves with" + title + "with select from model)"
    Plotting.plot_lc_curve(X_train, Y_train, title)
    return X_new


# dumb function to count target features
def counting_target_features(Y):
    if count_features:
        # inizio conteggio per ogni classe, per vedere se è bilanciato
        count_class_0 = 0
        count_class_1 = 0
        count_class_2 = 0
        count_class_3 = 0

        for i in Y:
            # print(i)
            if i == 0:
                count_class_0 += 1
            if i == 1:
                count_class_1 += 1
            if i == 2:
                count_class_2 += 1
            if i == 3:
                count_class_3 += 1
        print("samples of class 0: " + str(count_class_0))
        print("samples of class 1: " + str(count_class_1))
        print("samples of class 2: " + str(count_class_2))
        print("samples of class 3: " + str(count_class_3))


# Standard Scaler to standardize X
def standardization_(X):
    min_max_scaler = MinMaxScaler()
    names_cols = X.columns  # nomi delle colonne

    X_scale = pd.DataFrame(min_max_scaler.fit_transform(X[names_cols]), columns=names_cols)
    return X_scale


# the problem could be binarized, Cyrrosis, Many Septa==> class 1, Portal fibrosis, Few septa==> class 0
def binarizing_problem(i):
    if i == 1 or i == 2:
        i = 0
    if i == 3 or i == 4:
        i = 1
    return i


# MinMax Scaler to Normalize X
def normalization_(X):
    scaler = MinMaxScaler()
    names_cols = X.columns  # nomi delle colonne
    X_std = pd.DataFrame(scaler.fit_transform(X[names_cols]), columns=names_cols)
    return X_std


# I tried random forest, but it is not used for my problem
def ensamble_random_forest(X_train, Y_train, X_test, y_test):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    scores = cross_val_score(random_forest, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    # Y_train = pd.DataFrame(data=label_binarize(Y_train, classes=[1,2,3,4]))
    random_forest.fit(X_train, Y_train)
    print(scores.mean())

    y_predicted = random_forest.predict(X_test)
    print('Accuracy of Ensamble for is ', accuracy_score(y_test, y_predicted))

    # sns.barplot()


# Plot confusion matrix
def confusionMatrix(y_test, y_pred, title):
    f, axes = plt.subplots(1, 1, figsize=(10, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test,
                                                   y_pred))
    disp.plot(ax=axes, values_format='.3g')
    disp.ax_.set_title(title)
    disp.im_.colorbar.remove()
    plt.subplots_adjust(wspace=0.20, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()


# Drop features with variance 0.0, Variance threshold is used
def dropping_features_close_zero_variance(X):
    # X=X.drop(columns=['RNA Base'], axis=1)
    '''
    'RNA 4', 'ALT 36', 'ALT 1', 'ALT 4', 'ALT 12', 'ALT 24', 'ALT 36', 'ALT 48', 'AST 1','Plat'
    '''
    selector = VarianceThreshold(threshold=0)
    X_not_change = X.copy()
    selector.fit(X)
    constant_columns = [column for column in X.columns
                        if column not in
                        X.columns[selector.get_support()]]
    X = selector.transform(X)
    for column in constant_columns:
        print("Removed ", column)
    return pd.DataFrame(data=X, columns=X_not_change.columns[selector.get_support(indices=True)])


# Ensamble
def ensamble_bagging(X_train, Y_train, X_test, y_test, clf, title):
    clf_bagging = BaggingClassifier(base_estimator=clf, n_estimators=100, n_jobs=-1, random_state=0)

    score = cross_val_score(clf_bagging, X_train, Y_train, scoring='accuracy',
                            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10))
    print(score.mean())
    clf_bagging.fit(X_train, Y_train)
    y_pred = clf_bagging.predict(X_test)
    print("Accuracy of Ensamble with " + title + " is: " + str(accuracy_score(y_test, y_pred)))
    return accuracy_score(y_test, y_pred)


# main, here we go
if __name__ == '__main__':

    # Dati di input
    input_file = "./HCV-Egy-Data/HCV-Egy-Data.csv"
    df = pd.read_csv(input_file, header=0)
    print("Starting EDA...")
    # EDA.analysis_dataset(df)  # start eda to evaluate data

    print("EDA finished")

    X = df.drop(columns='Baselinehistological staging')  # X is df without target feature
    X_not_discret = X.copy()  # get a copy of X not discretized
    # EDA.clustering(X_not_discret, "evaluation without discretization")

    if discretization_bool:  # work with X discretized
        X = Discretization.discr_fun(X)  # call the function
        X = Discretization.converting_to_0_and_1(X)  # converting from [1,2] to [0,1], more clear
        print("X: \n", X)
        # EDA.clustering(X, "evaluation with discretization")
    Y = df['Baselinehistological staging']
    if problem_is_binarized:
        Y = Y.apply(binarizing_problem)  # see theory explanation

    df = pd.concat([X, Y], axis=1)  # rebuild the dataset with X discretized
    # print("Y: \n", Y)

    plot_metrics_for_each_features(df)
    name_columns = X.columns
    # print("X:\n" + str(X))

    print("DF after preprocessing: \n" + str(df))
    # counting_target_features(Y)  # conto le features

    # print(names_cols + "\n" + str(len(names_cols)))
    # plot images, not preprocessing
    # plot_metrics_for_each_features(names_cols, X, "_not_preprocessing")

    # plot and save images, with min max scaler
    # plot_metrics_for_each_features(names_cols, X_scale, "min_max_scaler")
    X = pd.DataFrame(data=X, columns=name_columns)  # create a dataframe

    # STARTING SPLITTING
    X_train_no_preproc, X_test_no_preproc, Y_train_no_preproc, Y_test_no_preproc = train_test_split(
        X, Y, random_state=0, train_size=0.80
    )
    X = dropping_features_close_zero_variance(X)  # Delete features with 0 variance

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=0, train_size=0.80
    )
    # print("New X is:\n", X)
    # print(X.shape)

    # START PREPROCESSING
    # ------------------------------Standardization or Normalization---------------------------------
    # ------------------------------------------NOT USED----------------------------------------------
    X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(
        standardization_(X), Y, random_state=0, train_size=0.80)
    # plot_metrics_for_each_features(names_cols, X_std, "_standardized")
    X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(
        normalization_(X), Y, random_state=0, train_size=0.80
    )
    # X_train, X_test,Y_train, Y_test=X_train_norm,X_test_norm,Y_train_norm, Y_test_norm
    # ----------------------------------------DEFINITION OF MODELS------------------------------------

    clf_KNN_no_feat_sel = KNeighborsClassifier()
    clf_DT_no_feat_sel = DecisionTreeClassifier()
    # Features selection
    k_migliore_knn, accuracy_max_knn, X_train_new_knn, X_test_new_knn = select_best_features_with_kbest(X_train,
                                                                                                        X_test,
                                                                                                        Y_train,
                                                                                                        Y_test,
                                                                                                        "KNN",
                                                                                                        clf_KNN_no_feat_sel)
    k_migliore_dt, accuracy_max_dt, X_train_new_dt, X_test_new_dt = select_best_features_with_kbest(X_train,
                                                                                                    X_test,
                                                                                                    Y_train,
                                                                                                    Y_test,
                                                                                                    "Decision Tree",
                                                                                                    clf_DT_no_feat_sel)

    print("Ho selezionato col KNN un numero di feature 'k': ", k_migliore_knn, "\n con accuracy: ", accuracy_max_knn)
    print("Ho selezionato col DT un numero di feature 'k': ", k_migliore_dt, "\n con accuracy: ", accuracy_max_dt)
    # ________________________________ Scelti i K best params___________________________________
    # select_k_best_KNN = SelectKBest(chi2, k=k_migliore_knn)
    # select_k_best_DT = SelectKBest(chi2, k=k_migliore_dt)

    # ----------------------------------Best Iperparameters-------------------------

    # clf_KNN = make_pipeline(select_k_best_KNN, clf_KNN_no_feat_sel)
    # clf_DT = make_pipeline(select_k_best_DT, clf_DT_no_feat_sel)
    # X_train_new_knn=X_train
    # X_test_new_knn = X_test
    # X_train_new_dt=X_train
    # X_test_new_dt=X_test
    BEST_PARAMS_KNN_PREPROC, best_estimator_knn = Cross_Validation.make_cross_validation(X_train_new_knn, Y_train,
                                                                                         clf_KNN_no_feat_sel,
                                                                                         parameters_knn,
                                                                                         "KNN with preprocessing")
    BEST_PARAMS_DT_PREPROC, best_estimator_dt = Cross_Validation.make_cross_validation(X_train_new_dt, Y_train,
                                                                                       clf_DT_no_feat_sel,
                                                                                       parameters_dt,
                                                                                       "Decision Tree with preprocessing")

    # ----------------------------------------Models with best iperpar--------------------------
    knn_model = KNeighborsClassifier(**BEST_PARAMS_KNN_PREPROC)
    decis_tree = DecisionTreeClassifier(**BEST_PARAMS_DT_PREPROC)
    # -----------------------------------------Final Models Preprocessing---------------------
    # clf_KNN_final = make_pipeline((MinMaxScaler()), best_estimator_knn )
    # clf_DT_final = make_pipeline(MinMaxScaler().fit(), best_estimator_dt)

    # -----------------------------------------END PREPROCESSING--------------------------------

    # --------------------------------------------------------------------------------------------
    # sns.displot(data=Y_test, x=Y_test.classes_)
    # plt.show()
    print("Preprocessing applied? " + str(preproc))
    print("Analysis with Discretization: " + str(discretization_bool))
    print("Problem is Binarized ?: " + str(problem_is_binarized))
    print("Standardization applied ? : " + str(standardization))
    print("Normalization applied? : " + str(normalization))

    # ---------------------------------PREDICTION------------------------------------------------
    clf_KNN_final = best_estimator_knn
    clf_DT_final = best_estimator_dt
    t1_start = time()
    ypred_preproc_knn = clf_KNN_final.predict(X_test_new_knn)
    t1_stop = time()
    print("process time for KNN Preprocessing:  %.20f", t1_stop - t1_start)
    t_start = time()
    ypred_preproc_dt = clf_DT_final.predict(X_test_new_dt)
    t_stop = time()
    print("process time for DT preprocessing:  %.20f", t_stop - t_start)
    # clf_KNN_final.fit(X_train_norm, Y_train_norm)
    # clf_DT_final.fit(X_train_norm, Y_train_norm)
    # ypred_preproc_knn = clf_KNN_final.predict(X_test_norm)
    # ypred_preproc_dt = clf_DT_final.predict(X_test_norm)

    # -----------------------------------------NO PREPROCESSING NOR CROSS VALIDATION----------------------------------
    t2_start = time()
    clf1 = KNeighborsClassifier().fit(X_train_no_preproc, Y_train_no_preproc)
    y_KNN_without_preproc = clf1.predict(X_test_no_preproc)
    t2_stop = time()
    print("Time for KNN NO PREPROCESSING:  %.20f" % (t2_stop - t2_start))

    clf2 = DecisionTreeClassifier().fit(X_train_no_preproc, Y_train_no_preproc)
    t3_start = time()
    y_DT_without_preproc = clf2.predict(X_test_no_preproc)
    t3_stop = time()
    print("Time for DT NO PREPROCESSING: %.20f" % (t3_stop - t3_start))
    # -------------------------------------------------------------------------------------------
    # NO PREPROCESSING, WITH CROSS VALIDATION
    BEST_PARAMS_KNN_NO_PREPROC, best_estimator_knn = Cross_Validation.make_cross_validation(X_train_no_preproc,
                                                                                            Y_train_no_preproc,
                                                                                            clf_KNN_no_feat_sel,
                                                                                            parameters_knn,
                                                                                            "KNN without preprocessing")
    BEST_PARAMS_DT_NO_PREPROC, best_estimator_dt = Cross_Validation.make_cross_validation(X_train_no_preproc,
                                                                                          Y_train_no_preproc,
                                                                                          clf_DT_no_feat_sel,
                                                                                          parameters_dt,
                                                                                          "Decision Tree without preprocessing")
    y_KNN_without_preproc_CV = KNeighborsClassifier().fit(X_train_no_preproc,
                                                          Y_train_no_preproc).predict(
        X_test_no_preproc)
    y_DT_without_preproc_CV = DecisionTreeClassifier().fit(X_train_no_preproc,
                                                           Y_train_no_preproc).predict(
        X_test_no_preproc)

    # LIST OF Y PREDICTED
    y_predicted = [ypred_preproc_knn, ypred_preproc_dt, y_KNN_without_preproc, y_DT_without_preproc]
    models_name = ["KNN PREPROCESSING", "DT PREPROCESSING", "KNN NO PREPROCESSING", "DT NO PREPROCESSING"]
    print("WITH KNN PREPROCESSING:\n", classification_report(Y_test, ypred_preproc_knn))
    print("WITH DT PREPROCESSING:\n", classification_report(Y_test, ypred_preproc_dt))
    print("WITH KNN NO PREPROCESSING:\n", classification_report(Y_test, y_KNN_without_preproc))
    print("WITH DT NO PREPROCESSING:\n", classification_report(Y_test, y_DT_without_preproc))
    print("WITH KNN NO PREPROCESSING WITH CV:\n", classification_report(Y_test_no_preproc, y_KNN_without_preproc_CV))
    print("WITH DT NO PREPROCESSING WITH CV:\n", classification_report(Y_test_no_preproc, y_DT_without_preproc_CV))

    # LET'S GET ACCURACY SCORES
    accuracy_score_KNN_preproc = accuracy_score(Y_test, ypred_preproc_knn)
    accuracy_score_DT_preproc = accuracy_score(Y_test, ypred_preproc_dt)
    accuracy_score_KNN_no_preproc = accuracy_score(Y_test, y_KNN_without_preproc)
    accuracy_score_DT_no_preproc = accuracy_score(Y_test, y_DT_without_preproc)
    accuracy_score_KNN_no_preproc_cv = accuracy_score(Y_test_no_preproc, y_KNN_without_preproc_CV)
    accuracy_score_DT_no_preproc_cv = accuracy_score(Y_test_no_preproc, y_DT_without_preproc_CV)
    accuracy_scores = [accuracy_score_KNN_preproc, accuracy_score_KNN_no_preproc, accuracy_score_KNN_no_preproc_cv]
    accuracy_index = ["KNN with\n Preprocessing and CV", "KNN with \nno Preprocessing",
                      "KNN with \nno Preprocessing, with CV"]
    # PLOT ACCURACY FOR EACH MODELS
    plot = sns.barplot(y=accuracy_index, x=accuracy_scores)
    plt.tight_layout()
    plot.set(xlabel="accuracy")
    plt.show()
    # I have to binarized 'cause I want plot ROC curve and Precision vs Recall
    le = LabelBinarizer()
    # print(Y_test.shape)
    # print(ypred_preproc_knn.shape)
    y_test_bin = le.fit_transform(Y_test)
    y_predicted_bin_KNN_with_preproc = le.fit_transform(ypred_preproc_knn)
    # print("Y after binarization:\n", y_test_bin, "\n", ypred_preproc_knn)

    # Plot just 2 ROC curves and 2 Precision Recall
    Evaluation.evaluating_models(y_test_bin, y_predicted_bin_KNN_with_preproc, "KNN with Preprocessing")
    y_predicted_bin_DT_with_preproc = le.fit_transform(ypred_preproc_dt)
    y_predicted_bin_DT_no_preproc = le.fit_transform(y_DT_without_preproc)
    y_predicted_bin_KNN_no_preproc = le.fit_transform(y_KNN_without_preproc)
    Evaluation.evaluating_models(y_test_bin, y_predicted_bin_DT_with_preproc,
                                "Decison Tree with Preprocessing")

    # create and Plot Confusion matrix
    confusionMatrix(Y_test, ypred_preproc_knn, "Confusion Matrix with KNN preprocessing")
    confusionMatrix(Y_test, y_KNN_without_preproc, "Confusion Matrix with KNN no preprocessing")
    confusionMatrix(Y_test, ypred_preproc_dt, "Confusion Matrix with DT preprocessing")
    confusionMatrix(Y_test, y_DT_without_preproc, "Confusion Matrix with DT no preprocessing")
    # ensamble_random_forest(X_train_new_dt, Y_train, X_test_new_dt, Y_test)

    # Get the Ensemble
    accuracy_score_KNN_processing_ensemble = ensamble_bagging(X_train_new_knn, Y_train, X_test_new_knn, Y_test,
                                                              best_estimator_knn,
                                                              "KNN with preprocessing")
    accuracy_score_DT_processing_ensemble = ensamble_bagging(X_train_new_dt, Y_train, X_test_new_dt, Y_test,
                                                             best_estimator_dt,
                                                             "DT with preprocessing")

    # fig, ax = plt.subplots()

#   plot_decision_regions(X[['RNA 12', 'RNA EF']].values, Y.values, best_estimator_knn, ax=ax)
# ax.set_xlabel('RNA 12')
# ax.set_ylabel('RNA EF')
# fig.suptitle('DT plot')
# plt.show()
# fig = plt.figure(figsize=(30, 20))
# tree.plot_tree(clf_DT_final, X.columns, 'Baselinehistological staging')
# tree.plot_tree(decision_tree=clf_DT_final, feature_names=X.columns, class_names='Baselinehistological staging')
# plt.show()
