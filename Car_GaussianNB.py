from scipy import sparse
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.utils import class_weight

input_file = "CAR/car.data"
df = pd.read_csv(input_file, header=0)

dataset = df.values

X = dataset[:, 0:6]
Y = dataset[:, 6]
print("X:\n", X)
print("Y:\n", Y)
switch_buying = {'vhigh': 1,
                 'high': 2,
                 'med': 3,
                 'low': 4
                 }
switch_maint = {'vhigh': 1,
                'high': 2,
                'med': 3,
                'low': 4
                }
switch_doors = {'2': 1,
                '3': 2,
                '4': 3,
                '5more': 5
                }

switch_persons = {'2': 2,
                  '4': 3,
                  'more': 4
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
X_final = np.empty(shape=np.shape(Y), dtype=int)
for i in range(len(X)):

    for j in range(len(X[i])):
        switch_temp = switch[j]
        X[i][j] = switch_temp[X[i][j]]

print(X)
'''
for i in range(len(Y)):
    Y[i]=switch_Y[Y[i]]

'''
#--------------------------Pre-processing--------------------------------
X_normalized = MinMaxScaler().fit_transform(X)
X_standardized = StandardScaler().fit_transform(X)
#------------------------------------------------------------------------
le = LabelEncoder()
le.fit_transform(np.unique(Y))
print(list(le.classes_))
y_dense = LabelBinarizer().fit_transform(Y)
#------------------------------------------------------------------------
# No preprocessing
X_train, X_test, Y_train, \
Y_test = \
    train_test_split(X, Y, test_size=0.3)
#------------------------------------------------------------------------
#Yes preprocessing
#-----------------------NORMALIZATION_SPLIT-------------------------------
X_train_NORM, X_test_NORM, Y_train_NORM, \
Y_test_NORM = \
    train_test_split(X, Y, test_size=0.3)
#-----------------------NORMALIZATION_SPLIT-------------------------------
X_train_STAND, X_test_STAND, Y_train_STAND, \
Y_test_STAND = \
    train_test_split(X, Y, test_size=0.3)
#_____________________________________
# y_sparse = sparse.csr_matrix(y_dense)
# print(y_sparse)
#_________________CALCULATING FREQUENCY___________________________

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
weight_class_1 = class1_num_y_train / (1208)
weight_class_2 = class2_num_y_train / (1208)
weight_class_3 = class3_num_y_train / (1208)
weight_class_4 = class4_num_y_train / (1208)
#--------------------------------STARTING FIT_______________________________________________
# BALANCING WITH WEIGHT, NO PREPROCESSING
gnb_weighted_no_preprocess = GaussianNB(priors=[weight_class_1, weight_class_2, weight_class_3, weight_class_4]).fit(X_train, Y_train)
# NO WEIGHT, NO PREPROCESSING
gnb_not_balanced = GaussianNB().fit(X_train, Y_train)
# BALANCING WITH WEIGHT,PREPROCESSING, NORMAL
gdb_weighted_preprocc_norm=GaussianNB(priors=[weight_class_1, weight_class_2, weight_class_3, weight_class_4]).fit(X_train_NORM, Y_train_NORM)
# BALANCING WITH WEIGHT,PREPROCESSING, STAND
gdb_weighted_preprocc_stand=GaussianNB(priors=[weight_class_1, weight_class_2, weight_class_3, weight_class_4]).fit(X_train_STAND, Y_train_STAND)


#--------------------PREDICTIONS---------------------------------

gnb_predictions = gnb_weighted_no_preprocess.predict(X_test)
gnb_predictions_no_balancing = gnb_not_balanced.predict(X_test)
gnb_predictions_norm=gdb_weighted_preprocc_norm.predict(X_test)
gnb_predictions_stand=gdb_weighted_preprocc_stand.predict(X_test)

# accuracy on X_test
target_names = ['unacc', 'acc', 'good', 'vgood']
# print("Score GNB with balancing: ", gnb_predictions.score(X_test, Y_test))
# print("Score GNB with no balancing: ", gnb_predictions_no_balancing.score(X_test, Y_test))

#---------------------------------PRINTING-----------------------------------------

print("-----------------------WITH-BALANCING-Normalization-Weighted----------------------------")
print(classification_report(Y_test, gnb_predictions_norm, target_names=target_names))
print("-----------------------WITH-BALANCING-Standardization-Weighted----------------------------")
print(classification_report(Y_test, gnb_predictions_stand, target_names=target_names))
print("-----------------------WITH NO BALANCING, NO PREPROCESSING----------------------------")
print(classification_report(Y_test, gnb_predictions_no_balancing, target_names=target_names))
print("-----------------------WITH BALANCING WEIGHTS, NO PREPROCESSING----------------------------")
print(classification_report(Y_test, gnb_predictions, target_names=target_names))
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
