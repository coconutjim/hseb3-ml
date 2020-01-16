__author__ = 'Lev Osipov'

import numpy as np
import matplotlib.pyplot as plt
from heapq import nsmallest


def cosine(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# Task 1
points = 100
dimension = 200
coordinates = []
results = []
print "Wait, please.."
for dim in xrange(1, dimension):
    coordinates.append(np.random.rand(points, dim) * 2 - 1)
    dim_results = []
    for i in xrange(points):
        cos = []
        for j in xrange(points):
            if i != j:
                cos.append(cosine(coordinates[dim - 1][i], coordinates[dim - 1][j]))
        dim_results.append([np.min(cos), np.max(cos)])
    results.append(np.mean(dim_results, axis=0))
np_results = np.array(results)
plt.plot(np_results[:, 0], label="Mean minimum cosine value")
plt.plot(np_results[:, 1], label="Mean maximum cosine value")
plt.xlabel('Dimension')
plt.ylabel('Cosine value')
plt.legend()
plt.show()

# Task 2
points = 1000
dimension = 1000
coordinates = []
function = []
print "Wait, please.."
for dim in xrange(1, dimension):
    coordinates.append(np.random.rand(points, dim) * 2 - 1)
    l2 = []
    l1 = []
    l_inf = []
    for i in xrange(points):
        l2.append(np.linalg.norm(coordinates[dim - 1][i]))
        l1.append(np.linalg.norm(coordinates[dim - 1][i], 1))
        l_inf.append(np.linalg.norm(coordinates[dim - 1][i], np.inf))
    l2_max = np.max(l2)
    l1_max = np.max(l1)
    l_inf_max = np.max(l_inf)
    l2_min = np.min(l2)
    l1_min = np.min(l1)
    l_inf_min = np.min(l_inf)
    function.append([(l2_max - l2_min) / l2_max, (l1_max - l1_min) / l1_max, (l_inf_max - l_inf_min) / l_inf_max])
np_function = np.array(function)
plt.plot(np_function[:, 0], label="l_2")
plt.plot(np_function[:, 1], label="l_1")
plt.plot(np_function[:, 2], label="l_infinity")
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.legend()
plt.show()

# Task 3
features_file = open("features.txt")
feature_list = []
for feature in features_file:
    id, name = feature.split('\t')
    feature_list.append(name)
features_file.close()

restaurants_file = open("new_york.txt")
restaurants = {}  # dictionary
for restaurant in restaurants_file:
    id, name, features = restaurant.split('\t')
    current_features = np.zeros(len(feature_list))
    current_features[[int(current_feature) for current_feature in features.split()]] = 1  # matching existing features
    restaurants[name] = current_features
restaurants_file.close()

fl = np.array(feature_list)


def find_similar(restaurant_name, all_restaurants, features_array):
    cf = all_restaurants[restaurant_name]  # current features
    del all_restaurants[restaurant_name]  # deleting current restaurant
    similar_ones = nsmallest(10, all_restaurants.items(), lambda item: np.linalg.norm(item[1] - cf))  # finding
    print "Restaurant: ", restaurant_name, features_array[np.nonzero(cf)]
    print "Similar:"
    for similar in similar_ones:
        print similar[0], features_array[np.nonzero(similar[1])]


# Tests
find_similar("Sazerac House", restaurants.copy(), fl)
find_similar("Isabella's", restaurants.copy(), fl)
find_similar("Lucky Strike", restaurants.copy(), fl)

