from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from Features_Selection import feature_selection_kbest
from sklearn import preprocessing
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer, MinMaxScaler
from Logistic_Regression import Logistic_regression


def plot_metrics_for_each_features(names_cols, X, name_png):
    figures = []
    for i in names_cols:
        figure = sns.displot(X, x=i)
        figures.append(figure)
        figure.savefig("./plots/" + str(i) + name_png)
        plt.close()  # plot close per chiudere la finestra di plot, onde evitare troppi  (>20)\
        # ed avere un errore a Runtime


def splitting_train_test(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=0, train_size=0.66
    )
    return X_train, X_test, Y_train, Y_test


def select_best_feauteres_with_kbest(X, Y):
    for i in range(2, 28):
        X_new=feature_selection_kbest(X, Y, i)
        X_train, X_test,Y_train, Y_test= splitting_train_test(X_new,Y)
        Logistic_regression(X_train, Y_train, X_test, Y_test)
def select_from_model(X,Y):
    svc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    X_new.shape

if __name__ == '__main__':
    # Dati di input
    input_file = "./HCV-Egy-Data/HCV-Egy-Data.csv"
    df = pd.read_csv(input_file, header=0)

    # df describe, descrive il dataset, inizio EDA
    print(df.describe())
    print(df.info())  # mi dice se ci sono tipi come 'object' e se qualche sample è nullo
    # drop della colonna corrispondente al target output
    X = df.drop(columns='Baselinehistological staging')
    print("X:\n" + str(X))
    Y = df['Baselinehistological staging']
    Y = Y.astype(int)  # converto in type int
    le = LabelEncoder()  # instanza che converte dal range [1,2,3,4] a [0,1,2,3]
    # i valori variano e possono essere 1 o 2. Li converto in 0 e 1 per maggior praticità
    X['Gender'] = le.fit_transform(X['Gender'])
    # print("Gender array: \n"+str(X['Gender']))
    X['Nausea or Vomiting'] = le.fit_transform(X['Nausea or Vomiting'])
    X['Headache '] = le.fit_transform(X['Headache '])
    X['Diarrhea '] = le.fit_transform(X['Diarrhea '])
    X['Fatigue & generalized bone ache '] = le.fit_transform(X['Fatigue & generalized bone ache '])
    X['Jaundice '] = le.fit_transform(X['Jaundice '])
    X['Epigastric pain '] = le.fit_transform(X['Epigastric pain '])
    # stesso preprocessing per l'array di output
    Y = le.fit_transform(Y)
    print("Y:\n" + str(Y))

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

    # ----------------------------------PREPROCESSING----------------------------------------

    names_cols = X.columns  # nomi delle colonne
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X[names_cols]), columns=names_cols)

    min_max_scaler = MinMaxScaler()
    X_scale = pd.DataFrame(min_max_scaler.fit_transform(X[names_cols]), columns=names_cols)
    print(names_cols + "\n" + str(len(names_cols)))
    # plot and save images, not preprocessing
    plot_metrics_for_each_features(names_cols, X, "_not_preprocessing")

    # plot and save images, with standardizationount_class_0 = 0
    count_class_1 = 0
    count_class_2 = 0
    count_class_3 = 0
    plot_metrics_for_each_features(names_cols, X_std, "_standardized")

    # plot and save images, with min max scaler
    plot_metrics_for_each_features(names_cols, X_scale, "min_max_scaler")

    print("samples of class 0: " + str(count_class_0))
    print("samples of class 1: " + str(count_class_1))
    print("samples of class 2: " + str(count_class_2))
    print("samples of class 3: " + str(count_class_3))
    # ------------------------------------END PREPROCESSING-------------------------------------

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=0, train_size=0.66
    )
    X_train_std, X_test_std, Y_train, Y_test = train_test_split(
        X_std, Y, random_state=0, train_size=0.66
    )
    X_train_minmax, X_test_minmax, Y_train, Y_test = train_test_split(
        X_scale, Y, random_state=0, train_size=0.66
    )

    select_best_feauteres_with_kbest(X, Y)
    select_from_model(X,Y)
    print("Logistic regression without preprocessing:\n")
    Logistic_regression(X_train, Y_train, X_test, Y_test)
    print("Logistic regression with Standardization:\n")
    Logistic_regression(X_train_std, Y_train, X_test_std, Y_test)
    print("Logistic regression with Min Max normalization:\n")
    Logistic_regression(X_train_minmax, Y_train, X_test_minmax, Y_test)
