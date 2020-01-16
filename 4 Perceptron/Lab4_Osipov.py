__author__ = 'Lev Osipov'

import numpy as np
import pandas as pd
import pylab as pl


def f(wghts, vec):
    result = np.dot(wghts, vec)
    if result > 0:
        return 1
    elif result < 0:
        return -1
    else:
        return 0


def iteration(wghts, lrn_rt, tr_classes, tr_data):
        for j in range(0, 10):
            i = np.random.randint(tr_data.shape[0])
            wghts += lrn_rt * (tr_classes[i] - f(wghts, tr_data.iloc[i])) * tr_data.iloc[i]


def test(wghts, data, classes):
    result = 0
    for i in range(0, data.shape[0]):
        if f(wghts, data.iloc[i]) == classes[i]:
            result += 1

    return result / float(len(data))


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_classes = train_data['class']
train_data = train_data.drop('class', axis=1)

test_classes = test_data['class']
test_data = test_data.drop('class', axis=1)

learning_rate = 1
max_iterations = 100

weights = np.zeros(train_data.shape[1])
results = []
local_max = 0
local_max_index = -1
for i in range(0, max_iterations):
    if local_max_index + 50 <= i:
        break
    iteration(weights, learning_rate, train_classes, train_data)
    learning_rate *= 0.95
    success_rate = test(weights, test_data, test_classes)
    if local_max < success_rate:
        local_max = success_rate
        local_max_index = i
    results.append([int(i + 1), success_rate])
results = np.array(results)
pl.plot(results[:, 0], results[:, 1])
pl.show()

