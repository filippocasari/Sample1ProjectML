from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def feature_selection_kbest(X, y, K):
    selector = SelectKBest(chi2, k=K)
    selector.fit(X, y)

    X_new = selector.transform(X)

    # text format
    var = X.columns[selector.get_support(indices=True)]
    # vector format
    vector_names = list(X.columns[selector.get_support(indices=True)])

    print(vector_names)
    return X_new
