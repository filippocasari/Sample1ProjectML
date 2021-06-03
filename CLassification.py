from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

input_file = "./HCV-Egy-Data/HCV-Egy-Data.csv"
df = pd.read_csv(input_file, header=0)

dataset = df.values

X = dataset[:, 0:28]
for i in X:
    for j in i:
        j = np.where(j == '?', 0, j)

# print(X)
Y = dataset[:, 28]
Y = Y.astype(int)
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
oftr_model = LogisticRegression(class_weight='balanced',
                                multi_class='multinomial',
                                solver='saga')
min_max_scaler = preprocessing.StandardScaler()
X_scale = min_max_scaler.fit_transform(X)

print("X:\n", X)
print("Y: \n", Y)
X_train, X_val_and_test, Y_train, \
Y_val_and_test = \
    train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split \
    (X_val_and_test, Y_val_and_test, test_size=0.5)
print("Shape of whole dataset : " + str(dataset.shape))
min_max_scaler.fit(X_train)
X_train = min_max_scaler.transform(X_train)
X_test = min_max_scaler.transform(X_test)
oftr_model.fit(X_train, Y_train)
y_pred = oftr_model.predict(X_test)
target_names = ['1', '2', '3', '4']
print(classification_report(Y_test, y_pred, target_names=target_names))
