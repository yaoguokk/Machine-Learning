import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import pandas as pd
import random
import csv





dic = {}
with open('synthetic1_test.csv', 'r') as f:
    tmp = csv.reader(f)
    for t in tmp:
        dic.setdefault(t[2], []).append([t[0], t[1]])

print(dic)


style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]


def k_nearest_neighbour(data ,predict, k=3):
    if len(data) >= k:
        print("k should be less than the total  voting group")
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]



    return vote_result

result = k_nearest_neighbour(dataset ,new_features, k=3)
print(result)


for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s= 100, color='b')

plt.scatter(new_features[0],new_features[1],s=50,color='r')
plt.show()