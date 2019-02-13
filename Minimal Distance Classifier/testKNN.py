import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import csv
import random
import operator
from plotDecBoundaries import plotDecBoundaries

# with open('iris.data', 'r') as csvfile:
# 	lines = csv.reader(csvfile)
# 	for row in lines:
# 		print (', '.join(row))

def loadDataset(length, trainfilename,testfilename, trainingSet=[], testSet=[]):
    with open(trainfilename, 'r') as csvfile1:
        lines = csv.reader(csvfile1)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(length):
                dataset[x][y] = float(dataset[x][y])
                trainingSet.append(dataset[x])

    with open(testfilename, 'r') as csvfile2:
        lines = csv.reader(csvfile2)
        dataset1 = list(lines)
        for x in range(len(dataset1) - 1):
            for y in range(length):
                dataset1[x][y] = float(dataset1[x][y])
                testSet.append(dataset1[x])



def centerPoint(trainSet):
    x1sum=0
    y1sum=0
    x2sum=0
    y2sum=0
    for x in range(len(trainSet)):
        if (trainingSet[x][2]=='1'):
            x1sum+=trainSet[x][0]
            y1sum+=trainSet[x][1]
        else:
            x2sum += trainSet[x][0]
            y2sum += trainSet[x][1]
    x1mean=x1sum/len(trainSet)
    y1mean=y2sum/len(trainSet)
    x2mean = x2sum / len(trainSet)
    y2mean = y2sum / len(trainSet)
    centerpoint=[[x1mean,y1mean,'1'],[x2mean,y2mean,'2']]
    return centerpoint

#  plot the trainset
trainingSet=[]
testSet=[]
loadDataset(2, 'synthetic1_train.csv', 'synthetic1_test.csv', trainingSet, testSet)
print("Train:"+ repr(len(trainingSet)))#the number of train set
print("Test:"+ repr(len(testSet)))#the number of test set

dic={1:'b', 2:'r'}
dic1={1:'>',2:'<'}

# plot the central point
for i in trainingSet:
    plt.scatter(i[0],i[1], s= 10, color=dic.get(int(i[2])))
centralpoint  = centerPoint(trainingSet)
for i in centralpoint:
    plt.scatter(i[0],i[1], s=100, color='cyan')
for i in testSet:
    plt.scatter(i[0],i[1], s= 50,color=dic.get(int(i[2])))





#difine the euclidieanDistance of two set, return the distance
def euclidieanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

#test euclidieanDistance function
data1= [3,3,4,'a']
data2= [4,4,5,'b']
data3= [5,6,7,'b']
# distance = euclidieanDistance(data1, data2, 4)
# print("Distance: "+ repr(distance))



#return the first k point which is cloeast to the testInstance point

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclidieanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors #return the train point which cloest to the testpoint



def getNeighbors1(centerpoint, testInstance):
    distances = []
    length = len(testInstance)-1
    for x in range(len(centerpoint)):
        dist = euclidieanDistance(testInstance, centerpoint[x], length)
        distances.append((centerpoint[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    neighbors.append(distances[0][0])
    return neighbors #return the train point which cloest to the testpoint

# test getNeighborsighbors funciton
# trainset = [data1,[4,4,5,'b'],data3]
#
# testInstance = [5,5,5]
# k = 1
# print(centralpoint)
# neighbors1 = getNeighbors1(centralpoint, testInstance)
# neighbors = getNeighbors(trainset, testInstance,1)
# print(neighbors1)

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]# [5, 6, 7, 'b'] vote feature as the last items 'b'
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]# the sortedVotes is like [('b', 2), ('a', 1)] with feature 'b' and votes 2

#test getResponse
# neighbors = [data1,data2,data3]
# response = getResponse(neighbors)
# print(response)   # b




def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0 # return percentage in 100 scale

#test getAccuracy
# testSet = [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
# predictions  = ['a','a','a']
# accuracy = getAccuracy(testSet, predictions)
# print(accuracy)


def main():
    # prepare data
    trainingSet = []
    testSet = []
    loadDataset(2,'synthetic1_train.csv','synthetic1_test.csv', trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

def main1():
    # prepare data
    trainingSet = []
    testSet = []
    loadDataset(2,'synthetic1_train.csv','synthetic1_test.csv', trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    cPoint  = centerPoint(trainingSet)
    for x in range(len(testSet)):
        neighbors = getNeighbors1(cPoint, testSet[x])
        result = getResponse(neighbors)
        predictions.append(result)
        if(testSet[x][-1]==result):
            plt.scatter(testSet[x][0],testSet[x][1],s=10,marker='x',color = 'green')
        else:
            plt.scatter(testSet[x][0], testSet[x][1], s=10, marker='D',color ='black')
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    plt.show()

main1()
# style.use('fivethirtyeight')
#
# dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
# new_features = [5,7]
#
#
# def k_nearest_neighbour(data ,predict, k=3):
#     if len(data) >= k:
#         print("k should be less than the total  voting group")
#     distances=[]
#     for group in data:
#         for features in data[group]:
#             euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
#             distances.append([euclidean_distance, group])
#
#     votes = [i[1] for i in sorted(distances)[:k]]
#     print(Counter(votes).most_common(1))
#     vote_result = Counter(votes).most_common(1)[0][0]
#
#
#
#     return vote_result
#
# result = k_nearest_neighbour(dataset ,new_features, k=3)
# print(result)
#
#
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1], s= 100, color=i)
#
# plt.scatter(new_features[0],new_features[1],s=50,color='b')
# plt.show()