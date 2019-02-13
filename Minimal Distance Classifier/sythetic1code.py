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
    for x in range(len(trainSet)):
        if (trainSet[x][2]=='1'):
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



def prepareforExplot():
    trainingSet = []
    testSet = []
    # loadDataset 3 is to transfer all data to float
    loadDataset(3, 'synthetic1_train.csv', 'synthetic1_test.csv', trainingSet, testSet)
    train_set = np.array(trainingSet)
    trainxy = np.array(train_set[:, 0:2])
    trainlable = np.array([int(i) for i in train_set[:, 2:3]])  # must be int
    print(train_set[1])
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


def main1():
    # prepare data
    count= 0
    trainingSet = []
    testSet = []
    loadDataset(2,'synthetic1_train.csv','synthetic1_test.csv', trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    cPoint  = centerPoint(trainingSet)
    set = testSet
    for x in range(len(set)):
        result = nearest(cPoint, set[x])
        if set[x][-1]==result[0]:
            plt.scatter(set[x][0],set[x][1],s=10,marker='x',color = 'green')
            count=count+1
        else:
            plt.scatter(set[x][0], set[x][1], s=10, marker='D',color ='black')
        print('> predicted=' + repr(result[0]) + ', actual=' + repr(testSet[x][-1]))
    accuracy = count/len(set)
    print('Accuracy: ' + repr(accuracy) + '%')
    plt.show()
    prepareforExplot()

main1()