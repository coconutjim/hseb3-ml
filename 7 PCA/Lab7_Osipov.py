__author__ = 'Lev Osipov'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Task 1
data = pd.read_csv('stock_prices.csv')

# Task 2
prices = data.iloc[:, 1:]
correlation_matrix = prices.corr()
correlations = []
for i in range(correlation_matrix.shape[0]):
    values = correlation_matrix.icol(i).values[i + 1:]
    for j in range(len(values)):
        correlations.append(values[j])
plt.hist(correlations)
plt.title("Correlation histogram")
plt.show()

# Task 3
pca = PCA(1)
reduction = pca.fit_transform(prices)
print "First component", pca.explained_variance_ratio_[0]
series = pd.Series(reduction[:, 0], data['Date'])
series.plot(title='PCA dynamic')
plt.show()

# Task 4
dji = pd.read_csv('dji.csv')
series = pd.Series(dji['Close'].values, dji['Date'])
series.plot(title='Dow Jones index dynamic')
plt.show()