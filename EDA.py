
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

def analysis_dataset(df):
    # EDA starting...
    #print(df)
    # df = discr_fun(df)
    sns.set(style="ticks", color_codes=True)
    # plt.hist(df['Baselinehistological staging'])

    # show the balanced dataset
    sns.displot(data=df['Baselinehistological staging'])
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
    sns.pairplot(df, hue='Baselinehistological staging')
    plt.show()


