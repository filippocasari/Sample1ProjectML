import itertools

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from numpy import square
from sklearn.cluster import KMeans

from main import discretization_bool
from Discretization import discr_fun


def clustering(X, title):
    inertia = []  # Squared Distance between Centroids and data points
    for n in range(1, 15):
        algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, random_state=111,
                            algorithm='elkan'))
        algorithm.fit(X)
        inertia.append(algorithm.inertia_)

    plt.figure()
    plt.plot(np.arange(1, 15), inertia, 'o')
    plt.plot(np.arange(1, 15), inertia, '-', alpha=0.5)
    plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
    plt.title(title)
    plt.show()
    algorithm_final = KMeans(n_clusters=4, init='k-means++', n_init=10,  random_state=111,
                             algorithm='elkan')

    X3 = X[['Baseline histological Grading', 'RNA EOT', 'RNA EF']].iloc[:, :].values
    fig = plt.figure()
    algorithm_final.fit(X3)
    labels4 = algorithm_final.labels_
    # print(labels3)

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X['Baseline histological Grading'], ys=X['RNA EOT'], zs=X['RNA EF'], marker='o', s=300,
               c=labels4)
    ax.set_xlabel('Baseline histological Grading')
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
    plt.subplots(figsize=(30, 30))
    sns.heatmap(corr_matrix, square=False, linewidths=.5,annot_kws={"fontsize":15} )

    plt.show()
    print(df.info())
    print(df.isnull())
    print(df.describe())
    print('Duplicate Values: ', len(df)-len(df.drop_duplicates()))
    columns_to_show = ['Baseline histological Grading', 'RNA 12', 'RNA EOT']
    print(df.groupby(['Baselinehistological staging'])[columns_to_show].describe(percentiles=[]))
    X = df.drop(columns='Baselinehistological staging')


    X = discr_fun(X)
    print("X discretizzata...")
    #gr = sns.FacetGrid(data=df, row='Age', col="RNA EF", hue="Baselinehistological staging", height=3.5)
    #gr.map(plt.scatter, "1", "2", alpha=0.6)
    #gr.add_legend()
    plt.show()



    # show the balanced dataset
    hue = 'Baselinehistological staging'
    sns.displot(data=df['Baselinehistological staging'])

    plt.legend()
    #sns.countplot(x='RNA 12', hue=hue, data=df)
    plt.show()
    df=discr_fun(df)
    df.plot.scatter(x='RNA 12', y='RNA EF', c='Baselinehistological staging', logy=True, cmap='summer')
    plt.show()
    df.plot.scatter(x='RNA 12', y='RNA EOT', c='Baselinehistological staging', logy=True, cmap='autumn')
    plt.show()
    print(pd.crosstab(df['RNA 12'], df['Baselinehistological staging'], margins=True))
    sns.countplot(x='RNA 12', hue='RNA EOT', data=df)
    plt.show()
    sns.boxplot(x='RNA EF', data=df)
    plt.show()

    # print(df['Baselinehistological staging'])


    # sns.pairplot(df_chosen, hue=hue)
    #df_chosen = df[['RNA 12', 'RNA EOT', 'RNA EF', hue]]
   # mks = itertools.cycle(["o", "s", "D", "X", "v"])
    #markers = [next(mks) for i in df[hue].unique()]
    #g = sns.pairplot(df_chosen, hue=hue, markers=markers, palette=['red', 'green', 'black', 'yellow'])
    corr_df = df.corr()
    print("The correlation DataFrame is:")
    print(corr_df, "\n")

    # list_corr=corr_df.abs().nlargest(28, Y)['Baselinehistological staging'].index

    matrix = np.triu(corr_df, k=1)
    #sns.heatmap(corr_df, annot=True, cmap='coolwarm', square=True, linewidth=0.1, mask=matrix)
   # plt.show()
    return 0
