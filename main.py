from time import sleep

from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

def distplot(feature, frame, color='g'):
    plt.figure()
    plt.title("Distribution for {}".format(feature))
    ax = sns.histplot(frame[feature], color=color)

if __name__ == '__main__':
    input_file = "./HCV-Egy-Data/HCV-Egy-Data.csv"
    df = pd.read_csv(input_file, header=0)


    # df describe
    print(df.describe())
    print(df.info()) # mi dice se ci sono tipi come 'object' e se qualche sample è nullo
    X=df.drop(columns='Baselinehistological staging')
    print("X:\n"+str(X))

    Y = df['Baselinehistological staging']
    Y = Y.astype(int)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    print("Y:\n" + str(Y))
    count_class_0 = 0
    count_class_1 = 0
    count_class_2 = 0
    count_class_3 = 0
    for i in Y:
        # print(i)
        if (i == 0):
            count_class_0 += 1
        if (i == 1):
            count_class_1 += 1
        if (i == 2):
            count_class_2 += 1
        if (i == 3):
            count_class_3 += 1

    names_cols=X.columns



    print(names_cols+"\n"+str(len(names_cols)))
    figures=[]
    #TODO scaler boolean values
    for i in names_cols:

        figure=sns.displot(X, x=i)
        figures.append(figure)
        figure.savefig("./plots/"+str(i)+"_not_preprocessing")
        plt.close()
    scaler = StandardScaler().fit(X[names_cols])
    X_std=pd.DataFrame(scaler.transform(X[names_cols]), columns=names_cols)
    for i in names_cols:

        figure=sns.displot(X_std, x=i)
        figures.append(figure)
        figure.savefig("./plots/"+str(i)+"_not_preprocessing_standardized")
        plt.close()

    print("samples of class 0: " + str(count_class_0))
    print("samples of class 1: " + str(count_class_1))
    print("samples of class 2: " + str(count_class_2))
    print("samples of class 3: " + str(count_class_3))
    log_regr = LogisticRegression(penalty='l2', class_weight='balanced')
    X_train, X_val_and_test, Y_train, \
    Y_val_and_test = \
        train_test_split(X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split \
        (X_val_and_test, Y_val_and_test, test_size=0.5)

    #log_regr.fit(X_train, Y_train)
    #y_pred = log_regr.predict(X_test)
    #print(classification_report(Y_test, y_pred))