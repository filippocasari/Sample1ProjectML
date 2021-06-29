import itertools

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from numpy import square
from sklearn.cluster import KMeans

from main import discretization_bool, discr_fun


def clustering(X, title):
    inertia = []  # Squared Distance between Centroids and data points
    for n in range(1, 11):
        algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, random_state=111,
                            algorithm='elkan'))
        algorithm.fit(X)
        inertia.append(algorithm.inertia_)

    plt.figure()
    plt.plot(np.arange(1, 11), inertia, 'o')
    plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
    plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
    plt.title(title)
    plt.show()
    algorithm_final = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111,
                             algorithm='elkan')

    X3 = X[['Age', 'RNA EOT', 'RNA EF']].iloc[:, :].values
    fig = plt.figure()
    algorithm_final.fit(X3)
    labels4 = algorithm_final.labels_
    # print(labels3)
    X['label4'] = labels4
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X['Age'], ys=X['RNA EOT'], zs=X['RNA EF'], marker='o', s=300,
               c=X['label4'])
    ax.set_xlabel('Age')
    ax.set_ylabel('RNA EOT')
    ax.set_zlabel('RNA EF')
    plt.title('Clusters '+title)

    plt.show()


def analysis_dataset(df):
    # EDA starting...
    # print(df)
    # df = discr_fun(df)
    # sns.set(style="ticks", color_codes=True)
    # plt.hist(df['Baselinehistological staging'])
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, square=False )
    plt.figure(figsize=(20, 20))
    #plt.scatter(df['Total day minutes'], df['Total night minutes'])
    plt.show()
    print(df.info())
    print(df.isnull())
    print(df.describe())
    print('Duplicate Values: ', len(df)-len(df.drop_duplicates()))
    columns_to_show = ['Age', 'RNA 12', 'RNA EOT']
    print(df.groupby(['Baselinehistological staging'])[columns_to_show].describe(percentiles=[]))
    X = df.drop(columns='Baselinehistological staging')
    X_not_discret = X.copy()
    if discretization_bool:
        X = discr_fun(X)
        print("X discretizzata...")
        gr = sns.FacetGrid(data=df, row='Age', col="RNA EF", hue="Baselinehistological staging", height=3.5)
        gr.map(plt.scatter, "1", "2", alpha=0.6)
        gr.add_legend()


    if discretization_bool:
        clustering(X, "with data not continues")

    clustering(X_not_discret, "with data continues")
    # show the balanced dataset
    hue = 'Baselinehistological staging'
    sns.displot(data=df['Baselinehistological staging'])
    #plt.pie(x=hu ,data=df['Baselinehistological staging'], shadow=True)
    plt.legend()

    plt.title("RNA EF and Classes")

    sns.countplot(x='RNA 12', hue=hue, data=df)
    plt.show()
    '''
    df.plot.scatter(x='RNA 12', y='RNA EF', c='Baselinehistological staging', logy=True, cmap='summer')
    plt.show()
    df.plot.scatter(x='RNA 12', y='RNA EOT', c='Baselinehistological staging', logy=True, cmap='autumn')
    plt.show()
    print(pd.crosstab(df['RNA 12'], df['Baselinehistological staging'], margins=True))
    sns.countplot(x='RNA 12', hue='RNA EOT', data=df)
    plt.show()
    sns.boxplot(x='RNA EF', data=df)
    plt.show()
    '''
    # print(df['Baselinehistological staging'])


    # sns.pairplot(df_chosen, hue=hue)
    df_chosen = df[['RNA 12', 'RNA EOT', 'RNA EF', hue]]
    mks = itertools.cycle(["o", "s", "D", "X", "v"])
    markers = [next(mks) for i in df[hue].unique()]
    #g = sns.pairplot(df_chosen, hue=hue, markers=markers, palette=['red', 'green', 'black', 'yellow'])
    corr_df = df.corr()
    print("The correlation DataFrame is:")
    print(corr_df, "\n")

    # list_corr=corr_df.abs().nlargest(28, Y)['Baselinehistological staging'].index
    plt.figure(figsize=(40, 30))
    matrix = np.triu(corr_df, k=1)
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', square=True, linewidth=0.1, mask=matrix)

    plt.show()
