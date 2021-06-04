from scipy import sparse
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils import class_weight

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

def trasform_matrix(x):

    for i in range(len(x)):

        for j in range(len(x[i])):
            switch_temp = switch[j]
            x[i][j] = switch_temp[x[i][j]]

    print(x)
    return x



input_file = "CAR/car.data"
df = pd.read_csv(input_file, header=0)

dataset = df.values

X = dataset[:, 0:6]
Y = dataset[:, 6]
print("X:\n", X)
print("Y:\n", Y)


switch_Y = {
    'unacc': 1,
    'acc': 2,
    'good': 3,
    'vgood': 4
}
X_final = np.empty(shape=np.shape(Y), dtype=int)





X = trasform_matrix(x=X)
'''
for i in range(len(Y)):
    Y[i]=switch_Y[Y[i]]

'''
le = LabelEncoder()
le.fit_transform(Y)

X_train, X_test, Y_train, \
Y_test = \
    train_test_split(X, Y, test_size=0.3)
print(list(le.classes_))
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(Y_train),
                                                  Y_train)
y_dense = LabelBinarizer().fit_transform(Y)

y_sparse = sparse.csr_matrix(y_dense)
print(y_sparse)

# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)
# X_test_minmax =min_max_scaler.fit_transform(X_test)
gnb = GaussianNB().fit(X_train, Y_train)

gnb_predictions = gnb.predict(X_test)

# accuracy on X_test
accuracy = accuracy_score(Y_test, gnb_predictions, normalize=True)
print("Accuracy :" ,accuracy)

# creating a confusion matrix
cm = confusion_matrix(Y_test, gnb_predictions)
x_sample1 = np.array([['low', 'low', '5more', 'more', 'big', 'high']])
x_sample2=np.array([['vhigh','vhigh','2','2','med','low']])
x_sample1 = trasform_matrix(x=x_sample2)
y_sample1_predict = gnb.predict(x_sample2)
print(y_sample1_predict)
accuracy_sample_1 = accuracy_score(['unacc'], y_sample1_predict)
print(accuracy_sample_1)
print(cm)
exit()