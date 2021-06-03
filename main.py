from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn import preprocessing
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    input_file = "./HCV-Egy-Data/HCV-Egy-Data.csv"
    df = pd.read_csv(input_file, header=0)

    dataset = df.values

    X = dataset[:, 0:28]
    X=np.where(X=='?', 0, X)

    # print(X)
    Y = dataset[:, 28]
    Y = Y.astype(int)
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    '''
    switch = {1: np.array([1, 0, 0, 0]),
              2: np.array([0, 1, 0, 0]),
              3: np.array([0, 0, 1, 0]),
              4: np.array([0, 0, 0, 1])
              }
    Y_final = np.empty(shape=(1385, 4), dtype=int)
    for i in range(len(Y)):
        Y_final[i] = switch[Y[i]]
    print(Y_final)
    '''
    param_grid = [
        {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
                (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
                (17,), (18,), (19,), (20,), (21,)
            ]
        }
    ]
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
    # print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
    print("Shape of X train: " + str(X_train.shape) + "\nShape of Y train: " + str(Y_train.shape))
    print("Shape of X valuation: " + str(X_val.shape) + "\nShape of Y valuation: " + str(Y_val.shape))
    print("Shape of X test: " + str(X_test.shape) + "\nShape of Y test: " + str(Y_test.shape))

    clf = MLPClassifier()
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    target_names = ['1', '2', '3', '4']
    print(classification_report(Y_test, y_pred, target_names=target_names))
    # print('Accuracy on the test set is: ' + str(np.round(accuracy_score(Y_test, y_pred) * 100, 2)) + '%')
    print(y_pred)
    # print(y_pred - Y_test)
    '''
    model = Sequential([
        Dense(2, activation='relu', input_shape=(28,)),
        Dense(2, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(X_train, Y_train,
                     batch_size=2, epochs=10,
                     validation_data=(X_val, Y_val))
    model.evaluate(X_test, Y_test)[1]
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')

    plt.show()
    
    applicazione senza preprocessing dei dataset
    applicazione con preprocessing
    vedere quella migliore
    unsambal
    plottare le prestazioni
    sezione dei risultati==> tabelle e plot==> 15/20 pagine
    early stopping, weight decay
    '''
