from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler, StandardScaler


input_file = "CAR/car.data"
df = pd.read_csv(input_file, header=0)

dataset = df.values
print(df.info())
print(df.describe())
X = dataset[:, 0:6]
Y = dataset[:, 6]
print("X:\n", X)
print("Y:\n", Y)
switch_buying = {'vhigh': 0,
                 'high': 1,
                 'med': 2,
                 'low': 3
                 }
switch_maint = {'vhigh': 0,
                'high': 1,
                'med': 2,
                'low': 3
                }
switch_doors = {'2': 0,
                '3': 1,
                '4': 2,
                '5more': 3
                }

switch_persons = {'2': 0,
                  '4': 1,
                  'more': 2
                  }
switch_lug_boot = {
    'small': 1,
    'med': 2,
    'big': 3
}
switch_safety = {
    'low': 1,
    'med': 2,
    'high': 3
}
switch = {
    0: switch_buying,
    1: switch_maint,
    2: switch_doors,
    3: switch_persons,
    4: switch_lug_boot,
    5: switch_safety
}
switch_Y = {
    'unacc': 1,
    'acc': 2,
    'good': 3,
    'vgood': 4
}
X_final = np.empty(shape=np.shape(X), dtype='int64')
for i in range(len(X)):

    for j in range(len(X[i])):
        switch_temp = switch[j]
        X_final[i][j] = switch_temp[X[i][j]]

print("Original X:\n",X)
print("X:\n",X_final)
'''
for i in range(len(Y)):
    Y[i]=switch_Y[Y[i]]

'''
print("X : \n", X)
# --------------------------Pre-processing--------------------------------
X_normalized = MinMaxScaler().fit_transform(X_final)
print("X normalized: \n", X_normalized)
X_standardized = StandardScaler().fit_transform(X_final)
print("X standardized: \n", X_standardized)
# ------------------------------------------------------------------------
le = LabelEncoder()
le.fit_transform(np.unique(Y))
print(list(le.classes_))
# y_dense = LabelBinarizer().fit_transform(Y)
# ------------------------------------------------------------------------
# No preprocessing
X_train, X_test, Y_train, \
Y_test = \
    train_test_split(X_final, Y, test_size=0.3)

# ------------------------------------------------------------------------
# Yes preprocessing
# -----------------------NORMALIZATION_SPLIT-------------------------------
X_train_NORM, X_test_NORM, Y_train_NORM, \
Y_test_NORM = \
    train_test_split(X_normalized, Y, test_size=0.3)
# -----------------------NORMALIZATION_SPLIT-------------------------------
X_train_STAND, X_test_STAND, Y_train_STAND, \
Y_test_STAND = \
    train_test_split(X_standardized, Y, test_size=0.3)
# _____________________________________
# y_sparse = sparse.csr_matrix(y_dense)
# print(y_sparse)
# _________________CALCULATING FREQUENCY___________________________

class1_num_y_train = 0
class2_num_y_train = 0
class3_num_y_train = 0
class4_num_y_train = 0
for i in Y_train:
    if (i == 'unacc'):
        class1_num_y_train += 1
    elif (i == 'acc'):
        class2_num_y_train += 1
    elif (i == 'good'):
        class3_num_y_train += 1
    elif (i == 'vgood'):
        class4_num_y_train += 1
print(np.shape(Y_train))
weight_class_1 = class1_num_y_train / 1208
weight_class_2 = class2_num_y_train / 1208
weight_class_3 = class3_num_y_train / 1208
weight_class_4 = class4_num_y_train / 1208
# --------------------------------STARTING FIT_______________________________________________
# BALANCING WITH WEIGHT, NO PREPROCESSING
gnb_weighted_no_preprocess = GaussianNB(priors=[weight_class_1, weight_class_2, weight_class_3, weight_class_4]).fit(
    X_train, Y_train)
# NO WEIGHT, NO PREPROCESSING
gnb_not_balanced = GaussianNB().fit(X_train, Y_train)
# BALANCING WITH WEIGHT,PREPROCESSING, NORMAL
gdb_weighted_preprocc_norm = GaussianNB().fit(
    X_train_NORM, Y_train_NORM)
# BALANCING WITH WEIGHT,PREPROCESSING, STAND----priors=[weight_class_1, weight_class_2, weight_class_3, weight_class_4]
gdb_weighted_preprocc_stand = GaussianNB().fit(
    X_train_STAND, Y_train_STAND)

# --------------------PREDICTIONS---------------------------------

gnb_predictions = gnb_weighted_no_preprocess.predict(X_test)
gnb_predictions_no_balancing = gnb_not_balanced.predict(X_test)
gnb_predictions_norm = gdb_weighted_preprocc_norm.predict(X_test_NORM)
gnb_predictions_stand = gdb_weighted_preprocc_stand.predict(X_test_STAND)

# accuracy on X_test
target_names = ['unacc', 'acc', 'good', 'vgood']
# print("Score GNB with balancing: ", gnb_predictions.score(X_test, Y_test))
# print("Score GNB with no balancing: ", gnb_predictions_no_balancing.score(X_test, Y_test))

# ---------------------------------PRINTING-----------------------------------------

print("-----------------------WITH-BALANCING-Normalization-Weighted----------------------------")
print(classification_report(Y_test_NORM, gnb_predictions_norm, labels=target_names))
print("-----------------------WITH-BALANCING-Standardization-Weighted----------------------------")
print(classification_report(Y_test_STAND, gnb_predictions_stand, labels=target_names))
print("-----------------------WITH NO BALANCING, NO PREPROCESSING----------------------------")
print(classification_report(Y_test, gnb_predictions_no_balancing, labels=target_names))
print("-----------------------WITH BALANCING WEIGHTS, NO PREPROCESSING----------------------------")
print(classification_report(Y_test, gnb_predictions, labels=target_names))
Num_of_instances = {'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0}
class1_num_y_test = 0
class2_num_y_test = 0
class3_num_y_test = 0
class4_num_y_test = 0
for i in Y_test:
    if (i == 'unacc'):
        class1_num_y_test += 1
    elif (i == 'acc'):
        class2_num_y_test += 1
    elif (i == 'good'):
        class3_num_y_test += 1
    elif (i == 'vgood'):
        class4_num_y_test += 1
print("-----------------------TEST----------------------------")
print("Instances of class 'unacc' ", class1_num_y_test)
print("Instances of class 'acc' ", class2_num_y_test)
print("Instances of class 'good' ", class3_num_y_test)
print("Instances of class 'vgood' ", class4_num_y_test)

print("-----------------------TRAIN----------------------------")
print("Instances of class 'unacc' ", class1_num_y_train)
print("Instances of class 'acc' ", class2_num_y_train)
print("Instances of class 'good' ", class3_num_y_train)
print("Instances of class 'vgood' ", class4_num_y_train)
# creating a confusion matrix
cm = confusion_matrix(Y_test, gnb_predictions)
precision = dict()
recall = dict()
n_classes=4
gnb_predictions=LabelBinarizer().fit_transform(gnb_predictions)
print(gnb_predictions)
Y_test=LabelBinarizer().fit_transform(Y_test)
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        gnb_predictions[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()
# roc curve
fpr = dict()
tpr = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i],
                                  gnb_predictions[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.show()
