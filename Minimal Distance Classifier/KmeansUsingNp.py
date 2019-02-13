import numpy as np
import math
import matplotlib.pyplot as plt

import csv
import operator
from plotDecBoundaries import plotDecBoundaries


def loadDataset(length, trainfilename,testfilename, trainingSet=[], testSet=[]):
    with open(trainfilename, 'r') as csvfile1:
        lines = csv.reader(csvfile1)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(length):
                dataset[x][y] = float(dataset[x][y])
            trainingSet.append(dataset[x])

    with open(testfilename, 'r') as csvfile2:
        lines = csv.reader(csvfile2)
        dataset1 = list(lines)
        for x in range(len(dataset1)):
            for y in range(length):
                dataset1[x][y] = float(dataset1[x][y])
            testSet.append(dataset1[x])



def centerPoint(trainSet):
    x1sum=0
    y1sum=0
    x2sum=0
    y2sum=0
    count1=0
    count2=0
    for x in range(len(trainSet)):
        if (trainSet[x][2]==1):
            x1sum+=trainSet[x][0]
            y1sum+=trainSet[x][1]
            count1=count1+1
        else:
            x2sum += trainSet[x][0]
            y2sum += trainSet[x][1]
            count2=count2+1
    x1mean=x1sum/count1
    y1mean=y1sum/count2
    x2mean = x2sum / count1
    y2mean = y2sum / count2
    centerpoint=[[x1mean,y1mean,1],[x2mean,y2mean,2]]
    return centerpoint


def prepareforExplot(trainingSet):
    train_set = np.array(trainingSet)
    trainxy = np.array(train_set[:, 0:2])
    trainlable = np.array([int(i) for i in train_set[:, 2:3]])  # must be int
    mean = centerPoint(trainingSet)
    mean = np.array(mean, dtype=np.float32)  # transfer all mean[] to float
    mean = mean[:, :2]
    # plot the boundary with function
    plotDecBoundaries(trainxy, trainlable, mean)







#difine the euclidieanDistance of two set, return the distance
def euclidieanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


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



def nearest(centerpoint, testInstance):
    distances = []
    length = len(testInstance)-1
    for x in range(len(centerpoint)):
        dist = euclidieanDistance(testInstance, centerpoint[x], length)
        distances.append((centerpoint[x], dist))
    distances.sort(key=operator.itemgetter(1))
    nearest = []

    nearest.append(distances[0][0][2])#[([0.5539213899999996, -0.04058025600000003, '2'], 6.721239790654441), ([0.0, -0.04058025600000003, '1'], 7.099820372176822)]
    return nearest #return the train point which cloest to the testpoint




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




def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0 # return percentage in 100 scale




def main():
    # prepare data
    count= 0
    trainingSet = []
    testSet = []
    loadDataset(3,'synthetic1_train.csv','synthetic1_test.csv', trainingSet, testSet)
    # generate predictions
    cPoint  = centerPoint(trainingSet)
    prepareforExplot(trainingSet)
    set = trainingSet
    for x in range(len(set)):
        result = nearest(cPoint, set[x])
        if set[x][-1]==int(result[0]):
            plt.scatter(set[x][0],set[x][1],s=10,marker='x',color = 'red')
            count=count+1
        else:
            plt.scatter(set[x][0], set[x][1], s=10, marker='D',color ='black')

    accuracy = count/len(set)
    print('Error rate: ' + repr(1-accuracy) )
    # plt.show()

main()