import os

import numpy as np
from mlxtend.plotting import plot_decision_regions
from numpy import mean, std
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, \
    precision_recall_curve, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold, \
    cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.tree import DecisionTreeClassifier

import Cross_Valuation
import EDA
import Plotting
import Valuation
from Features_Selection import feature_selection_kbest
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer, MinMaxScaler, label_binarize
from Logistic_Regression import Logistic_regression
from SVM_classifier import SVM_classifier

count_features = False
discretization_bool = True
problem_is_binarized = False
normalization = False
standardization = True
preproc = True
np.random.seed(31415)
BEST_PARAMS_KNN_PREPROC = {'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}
BEST_PARAMS_DT_PREPROC = {'criterion': 'gini', 'max_depth': 9, 'splitter': 'best'}
BEST_PARAMS_KNN_NO_PREPROC = {'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}
BEST_PARAMS_DT_NO_PREPROC = {'criterion': 'gini', 'max_depth': 9, 'splitter': 'best'}


# used algorithms : Logistic Regression, DecisionTree, Clustering (K-means for evaluate number of classes)


def plot_metrics_for_each_features(X, name_png=None):
    figures = []
    names_cols = X.columns
    try:
        os.makedirs("./plots")
    except FileExistsError:
        # directory already exists
        pass
    fig, axes = plt.subplots(5, 6)
    index = 0
    j = 1
    for i in X.columns:
        '''
        figure = sns.displot(X, x=i)
        figures.append(figure)
        figure.savefig("./plots/" + str(i) + name_png)
        plt.close()  # plot close per chiudere la finestra di plot, onde evitare troppi  (>20)\
        # ed avere un errore a Runtime'''
        j += 1
        sns.catplot(data=X, x=i, kind="count")
        # axes[index].set_title(i)

    plt.show()


def splitting_train_test(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=0, train_size=0.7
    )
    return X_train, X_test, Y_train, Y_test


def select_best_features_with_kbest(X_train, X_test, Y_train, Y_test, title, clf):
    indexes = []
    accuracy_scores = []
    max_accuracy = 0
    k_migliore = 0

    for i in range(3, 29):
        title = "Learning Curves with " + title
        selector = SelectKBest(chi2, k=i)
        selector.fit(X_train, Y_train)
        X_indices = np.arange(X.shape[-1])
        scores = selector.scores_
        scores /= scores.max()
        X_train_new = selector.transform(X_train)
        X_test_new = selector.transform(X_test)
        indexes.append(i)
        accuracy = accuracy_score(Y_test, (clf.fit(X_train_new, Y_train)).predict(X_test_new))
        if (accuracy > max_accuracy):
            max_accuracy = accuracy
            k_migliore = i
            X_train_new_final = selector.transform(X_train)
            X_test_new_final = selector.transform(X_test)

        accuracy_scores.append(accuracy)
        print(str(accuracy) + " with " + str(
            i) + " features")
        # print(X_new.shape)
        print(" X with selection K BEST \n" + str(X.columns.values[selector.get_support()]))
        # Plotting.plot_lc_curve(X_train_new, Y_train, title, i, clf)

    sns.barplot(x=indexes, y=accuracy_scores)
    plt.xlabel("K features")
    plt.ylim(0, 0.4)
    plt.ylabel("accuracy score")
    plt.show()
    return k_migliore, max_accuracy, X_train_new_final, X_test_new_final


def feature_selection_varince(X_train, X_test):
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    constant_columns = [column for column in X_train.columns
                        if column not in
                        X_train.columns[constant_filter.get_support()]]
    X_train = constant_filter.transform(X_train)
    X_test = constant_filter.transform(X_test)
    for column in constant_columns:
        print("Removed ", column)
    return X_train, X_test


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


def discretization_Age(i):
    if i in range(0, 33):
        i = 0
    elif i in range(33, 38):
        i = 1
    elif i in range(38, 43):
        i = 2
    elif i in range(43, 48):
        i = 3
    elif i in range(48, 53):
        i = 4
    elif i in range(53, 58):
        i = 5
    else:
        i = 6

    return i


def discretization_BMI(i):
    if i in range(0, 18):
        i = 0
    elif i in range(18, 25):
        i = 1
    elif i in range(25, 30):
        i = 2
    elif i in range(30, 35):
        i = 3
    elif i in range(35, 40):
        i = 4
    elif i in range(53, 58):
        i = 5

    return i


def discretization_WBC(i):
    if i in range(0, 4000):
        i = 0
    elif i in range(4000, 11000):
        i = 1
    elif i in range(11000, 12102):
        i = 2

    return i


def discretization_RBC(i):
    if 0 <= i < 3000000:
        i = 0
    elif 3000000 <= i < 5000000:
        i = 1
    elif 5018452 > i >= 5000000:
        i = 2

    return i


def discretization_Plat(i):
    if 93013 <= i < 100000:
        i = 0
    elif 100000 <= i < 225000:
        i = 1
    elif 225000 <= i < 226465:
        i = 2

    return i


def discretization_AST_ALT(i):
    if 0 <= i < 20:
        i = 0
    elif 20 <= i <= 40:
        i = 1
    elif 40 < i <= 128:
        i = 2

    return i


def discretization_HGB(df):
    print(df.loc[df.Gender == 1, 'HGB'])


def discretization_RNA(i):
    if 0 <= i <= 5:
        i = 0
    elif i > 5:
        i = 1

    return i


def discr_male_HGB(i):
    if 2 <= i < 14:
        i = 0
    elif 14 <= i <= 17:
        i = 1
    elif 17 < i <= 20:
        i = 2
    return i


def discr_female_HGB(i):
    if 2 <= i < 12:
        i = 0
    elif 12 <= i <= 15:
        i = 1
    elif 15 < i <= 20:
        i = 2
    return i


def discr_HGB(df):
    df['HGB'] = df['HGB'].apply(lambda x: discr_male_HGB(x) if x == 1 else discr_female_HGB(x))
    print(df['HGB'])
    return df


def discr_fun(X):
    X = discr_HGB(X)
    X['Age'] = X['Age'].apply(discretization_Age)
    X['BMI'] = X['BMI'].apply(discretization_BMI)
    X['WBC'] = X['WBC'].apply(discretization_WBC)
    X['RBC'] = X['RBC'].apply(discretization_RBC)
    X['Plat'] = X['Plat'].apply(discretization_Plat)
    X['AST 1'] = X['AST 1'].apply(discretization_AST_ALT)
    X['ALT 1'] = X['ALT 1'].apply(discretization_AST_ALT)
    X['ALT 4'] = X['ALT 4'].apply(discretization_AST_ALT)
    X['ALT 12'] = X['ALT 12'].apply(discretization_AST_ALT)
    X['ALT 24'] = X['ALT 24'].apply(discretization_AST_ALT)
    X['ALT 36'] = X['ALT 36'].apply(discretization_AST_ALT)
    X['ALT 48'] = X['ALT 48'].apply(discretization_AST_ALT)
    X['RNA Base'] = X['RNA Base'].apply(discretization_RNA)
    X['RNA 4'] = X['RNA 4'].apply(discretization_RNA)
    X['RNA 12'] = X['RNA 12'].apply(discretization_RNA)
    X['RNA EOT'] = X['RNA EOT'].apply(discretization_RNA)
    X['RNA EF'] = X['RNA EF'].apply(discretization_RNA)

    return X


# TODO Target encoder sulle features
def converting_to_0_and_1(X):
    le = LabelBinarizer()  # instanza che converte dal range [1,2,3,4] a [0,1,2,3]
    # i valori variano e possono essere 1 o 2. Li converto in 0 e 1 per maggior praticità
    X['Gender'] = le.fit_transform(X['Gender'])
    # print("Gender array: \n"+str(X['Gender']))
    X['Nausea or Vomiting'] = le.fit_transform(X['Nausea or Vomiting'])
    X['Headache '] = le.fit_transform(X['Headache '])
    X['Diarrhea '] = le.fit_transform(X['Diarrhea '])
    X['Fatigue & generalized bone ache '] = le.fit_transform(X['Fatigue & generalized bone ache '])
    X['Jaundice '] = le.fit_transform(X['Jaundice '])
    X['Epigastric pain '] = le.fit_transform(X['Epigastric pain '])

    return X


def counting_features(Y):
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


def standardization_(X):
    min_max_scaler = MinMaxScaler()
    names_cols = X.columns  # nomi delle colonne

    X_scale = pd.DataFrame(min_max_scaler.fit_transform(X[names_cols]), columns=names_cols)
    return X_scale


def binarizing_problem(i):
    if i == 1 or i == 2:
        i = 0
    if i == 3 or i == 4:
        i = 1
    return i


def normalization_(X):
    scaler = StandardScaler()
    names_cols = X.columns  # nomi delle colonne
    X_std = pd.DataFrame(scaler.fit_transform(X[names_cols]), columns=names_cols)
    return X_std


def ensamble(X_train, Y_train, X_test, y_test):
    bgclassfier = BaggingClassifier(base_estimator=KNeighborsClassifier(**BEST_PARAMS_KNN_PREPROC))
    crossvalidation = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    # Y_train = pd.DataFrame(data=label_binarize(Y_train, classes=[1,2,3,4]))
    bgclassfier.fit(X_train, Y_train)

    y_predicted = bgclassfier.predict(X_test)
    print('Accuracy is ', accuracy_score(y_test, y_predicted))

    # sns.barplot()


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


if __name__ == '__main__':

    # Dati di input
    input_file = "./HCV-Egy-Data/HCV-Egy-Data.csv"
    df = pd.read_csv(input_file, header=0)
    print("Starting EDA...")
    EDA.analysis_dataset(df)

    print("EDA finished")
    # df describe, descrive il dataset, inizio EDA

    X = df.drop(columns='Baselinehistological staging')
    X_not_discret = X.copy()
    EDA.clustering(X_not_discret, "evaluation without discretization")
    if discretization_bool:
        X = discr_fun(X)
        EDA.clustering(X, "evaluation with discretization")

    Y = df['Baselinehistological staging']
    Y = Y.astype(int)  # converto in type int
    X = converting_to_0_and_1(X)

    df = pd.concat([X, Y], axis=1)
    # plot_metrics_for_each_features(df, "ciao")
    name_columns = X.columns
    print("X:\n" + str(X))
    # discretization_HGB(X) # TODO da rivedere

    # stesso preprocessing per l'array di output
    # Y = le.fit_transform(Y)

    print("DF after preprocessing: \n" + str(df))
    counting_features(Y)  # conto le features

    # TODO countplot, displot, pieplot, barplot, violin plot, pairplot
    # countplot ==> mette a confronto della classe target, feature più rilevante

    # print(names_cols + "\n" + str(len(names_cols)))
    # plot and save images, not preprocessing
    # plot_metrics_for_each_features(names_cols, X, "_not_preprocessing")

    # plot and save images, with standardizationount_class_0 = 0

    # plot_metrics_for_each_features(names_cols, X_std, "_standardized")

    # plot and save images, with min max scaler
    # plot_metrics_for_each_features(names_cols, X_scale, "min_max_scaler")

    if problem_is_binarized:
        Y = Y.apply(binarizing_problem)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=0, train_size=0.80
    )
    # START PREPROCESSING
    # -----------------------Standardization or Normalization---------------------------------

    X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(
        standardization_(X), Y, random_state=0, train_size=0.80)

    X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(
        normalization_(X), Y, random_state=0, train_size=0.80
    )

    clf_KNN_no_feat_sel = KNeighborsClassifier()
    clf_DT_no_feat_sel = DecisionTreeClassifier()
    k_migliore_knn, accuracy_max_knn, X_train_new_knn, X_test_new_knn = select_best_features_with_kbest(X_train_std,
                                                                                                        X_test_std,
                                                                                                        Y_train_std,
                                                                                                        Y_test_std,
                                                                                                        "KNN",
                                                                                                        clf_KNN_no_feat_sel)
    k_migliore_dt, accuracy_max_dt, X_train_new_dt, X_test_new_dt = select_best_features_with_kbest(X_train_std,
                                                                                                    X_test_std,
                                                                                                    Y_train_std,
                                                                                                    Y_test_std,
                                                                                                    "Decision Tree",
                                                                                                    clf_DT_no_feat_sel)
    print("Ho selezionato col KNN un numero di feature 'k': ", k_migliore_knn, "\n con accuracy: ", accuracy_max_knn)
    print("Ho selezionato col DT un numero di feature 'k': ", k_migliore_dt, "\n con accuracy: ", accuracy_max_dt)
    # ________________________________ Scelti i K best params___________________________________
    select_k_best_KNN = SelectKBest(k=k_migliore_knn)
    select_k_best_DT = SelectKBest(k=k_migliore_dt)
    # ----------------------------------Selezione migliori iperparametri-------------------------
    parameters_dt = {'max_depth': list(range(1, 100, 2)), "criterion": ["gini", "entropy"],
                     "splitter": ["best", "random"]}
    parameters_knn = {'n_neighbors': list(range(1, 10, 2)), "weights": ["uniform", "distance"], "p": [1, 2]}

    # Cross_Valuation.make_cross_evaluation(X_train_new_knn, Y_train, clf_KNN_no_feat_sel, parameters_knn, "KNN with preprocessing")
    # Cross_Valuation.make_cross_evaluation(X_train_new_dt, Y_train, clf_DT_no_feat_sel, parameters_dt,
    #          "Decision Tree with preprocessing")

    # ----------------------------------------Modelli coi best iperpar--------------------------
    knn_model = KNeighborsClassifier(**BEST_PARAMS_KNN_PREPROC)
    decis_tree = DecisionTreeClassifier(**BEST_PARAMS_DT_PREPROC)
    # -----------------------------------------Modello finale Preprocessing---------------------
    clf_KNN_final = make_pipeline(select_k_best_KNN, knn_model)
    clf_DT_final = make_pipeline(select_k_best_DT, decis_tree)
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
    clf_KNN_final.fit(X_train_new_knn, Y_train)
    clf_DT_final.fit(X_train_new_dt, Y_train)
    ypred_preproc_knn = clf_KNN_final.predict(X_test_new_knn)
    ypred_preproc_dt = clf_DT_final.predict(X_test_new_dt)

    # -----------------------------------------NO PREPROCESSING----------------------------------
    y_KNN_without_preproc = clf_KNN_no_feat_sel.fit(X_train, Y_train).predict(X_test)
    y_DT_without_preproc = DecisionTreeClassifier().fit(X_train, Y_train).predict(X_test)
    # -------------------------------------------------------------------------------------------

    y_predicted = [ypred_preproc_knn, ypred_preproc_dt, y_KNN_without_preproc, y_DT_without_preproc]
    models_name = ["KNN PREPROCESSING", "DT PREPROCESSING", "KNN NO PREPROCESSING", "DT NO PREPROCESSING"]
    print("WITH KNN PREPROCESSING:\n", classification_report(Y_test, ypred_preproc_knn))
    print("WITH DT PREPROCESSING:\n", classification_report(Y_test, ypred_preproc_dt))
    print("WITH KNN NO PREPROCESSING:\n", classification_report(Y_test, y_KNN_without_preproc))
    print("WITH DT NO PREPROCESSING:\n", classification_report(Y_test, y_DT_without_preproc))
    le = LabelBinarizer()
    print(Y_test.shape)
    print(ypred_preproc_knn.shape)
    y_test_bin = le.fit_transform(Y_test)
    y_predicted_bin = le.fit_transform(ypred_preproc_knn)
    print("Y after binarization:\n", y_test_bin, "\n", ypred_preproc_knn)
    Valuation.valuating_models(models_name, y_test_bin, y_predicted_bin)

    confusionMatrix(Y_test, ypred_preproc_knn, "Confusion Matrix with KNN preprocessing")
    confusionMatrix(Y_test, y_KNN_without_preproc, "Confusion Matrix with KNN no preprocessing")
    confusionMatrix(Y_test, ypred_preproc_dt, "Confusion Matrix with DT preprocessing")
    confusionMatrix(Y_test, y_DT_without_preproc, "Confusion Matrix with DT no preprocessing")
    ensamble(X_train_new_knn, Y_train, X_test_new_knn, Y_test)
    fig, ax = plt.subplots()
    X_std = StandardScaler().fit_transform(X[['RNA 12', 'RNA EOT']])
    decis_tree.fit(X_not_discret[['RNA 12', 'RNA EF']], Y)
    plot_decision_regions(X_not_discret[['RNA 12', 'RNA EF']].values, Y.values, decis_tree, ax=ax)
    ax.set_xlabel('RNA 12')
    ax.set_ylabel('RNA EF')
    fig.suptitle('DT plot')

    plt.show()
