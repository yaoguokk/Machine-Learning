import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import operator
from plotDecBoundaries import plotDecBoundaries

def loadDataset(feature1,feature2,trainfilename,testfilename, trainingSet=[], testSet=[]):
    with open(trainfilename, 'r') as csvfile1:
        lines = csv.reader(csvfile1)
        # temp = []
        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [float(dataset[x][feature1]), float(dataset[x][feature2]), int(dataset[x][-1])]
            trainingSet.append(temp)


    with open(testfilename, 'r') as csvfile2:
        lines = csv.reader(csvfile2)
        # temp = []
        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [float(dataset[x][feature1]), float(dataset[x][feature2]), int(dataset[x][-1])]
            testSet.append(temp)



def centerPoint(trainSet):
    x1sum=0
    y1sum=0
    x2sum=0
    y2sum=0
    x3sum=0
    y3sum=0
    count1=0
    count2=0
    count3=0
    for x in range(len(trainSet)):
        if (trainSet[x][2]==1):
            count1=count1+1
            x1sum+=trainSet[x][0]
            y1sum+=trainSet[x][1]
        elif(trainSet[x][2]==2):
            count2=count2+1
            x2sum += trainSet[x][0]
            y2sum += trainSet[x][1]
        elif(trainSet[x][2]==3):
            count3=count3+1
            x3sum += trainSet[x][0]
            y3sum += trainSet[x][1]
    x1mean=x1sum/count1
    y1mean=y1sum/count1
    x2mean = x2sum / count2
    y2mean = y2sum / count2
    x3mean = x3sum / count3
    y3mean = y3sum / count3
    centerpoint=[[x1mean,y1mean,1],[x2mean,y2mean,2],[x3mean,y3mean,3]]
    return centerpoint


trainingSet=[]
testSet=[]
# # loadDataset 3 is to transfer all data to float
loadDataset(0,6,'wine_train.csv','wine_test.csv', trainingSet, testSet)
center=centerPoint(trainingSet)

def prepareforExplot(trainingSet):
    train_set = np.array(trainingSet)
    trainxy = np.array(train_set[:, 0:2])
    trainlable = np.array([int(i) for i in train_set[:, 2:3]])  # must be int
    mean = centerPoint(trainingSet)
    mean = np.array(mean)  # transfer all mean[] to float, dtype=np.float32
    mean = mean[:, :2]
    # plot the boundary with function
    plotDecBoundaries(trainxy, trainlable, mean)

# prepareforExplot(trainingSet)

# print the 1 2 3 spots with its mean
def printpot(trainingSet,center):
    for i in trainingSet:
        if(i[2]==1):
            l1= plt.scatter(i[0], i[1], s=20,color='r')
        if (i[2] == 2):
            l2=plt.scatter(i[0], i[1], s=20,color='b')
        if (i[2] == 3):
            l3=plt.scatter(i[0], i[1], s=20,color='black')
    for i in center:
        if (i[2] == 1):
            l4=plt.scatter(i[0], i[1], color='red',s=100,marker='D',label='class 1 mean')
        if (i[2] == 2):
            l5=plt.scatter(i[0], i[1], color='skyblue',s=100,marker='D',label='class 2 mean')
        if (i[2] == 3):
            l6=plt.scatter(i[0], i[1], color='saddlebrown',s=100,marker='D',label='class 3 mean')

    plt.legend((l1, l2, l3,l4,l5,l6),
               ('class 1', 'class 2', 'class 3','class 1 mean','class 2 mean','class 3 mean'),
               scatterpoints=1,
               loc='upper left',
               ncol=3,
               fontsize=7)

    plt.show()

# printpot(trainingSet,center)

#difine the euclidieanDistance of two set, return the distance
def euclidieanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)





#return the first k point which is cloeast to the testInstance point

def getNeighbors(trainingSet, centerpoint, k):
    distances = []
    length = len(centerpoint)-1
    for x in range(len(trainingSet)):
        dist = euclidieanDistance(centerpoint, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors #return the train point which cloest to the testpoint



def nearest(centerpoint, testInstance):
    distances = []
    length = len(testInstance)-1
    len1 = len(centerpoint)
    for x in range(len(centerpoint)):
        dist = euclidieanDistance(testInstance, centerpoint[x], length)
        distances.append((centerpoint[x], dist))
    distances.sort(key=operator.itemgetter(1))
    nearest=[]

    nearest.append(distances[0][0][-1])
    return nearest #return the train point which cloest to the testpoint



def main(a,b):
    # prepare data
    count= 0
    trainingSet = []
    testSet = []
    loadDataset(a,b,'wine_train.csv','wine_test.csv', trainingSet, testSet)
    # generate predictions
    cPoint  = centerPoint(trainingSet)
    set = testSet
    for x in range(len(set)):
        result = nearest(cPoint, set[x])
        if set[x][-1]==result[0]:
            count=count+1

        # print('> predicted=' + repr(result[0]) + ', actual=' + repr(set[x][-1]))
    accuracy = count/len(set)
    # print(accuracy)
    # print('>error rate of feature '+str(a)+' and feature '+str(b) +" is " +str((round(1-accuracy,2))))
    # plt.show()
    return [round(1-accuracy,3),a+1,b+1]

print(main(0,1))

def getaccuracyrank():
    list1 = []
    for i in range(13):
        for j in range(i,13):
            if i != j:
                list1.append(main(i, j))

    list1 = sorted(list1, key=operator.itemgetter(0))
    for i in range(len(list1)):
        if (i%11==0 or i==0) or i ==len(list1)-1:
            print(list1[i])
    writefile(list1)


def writefile(list):
    # csvfile=open("output.csv","a",newline='')
    # write=csv.writer(csvfile,dialect='excel')
    writer = csv.writer(open('output.csv', 'w',newline=""),dialect='excel')
    writer.writerow([ "error rate","feature 1", "feature 2"])
    for i in list:
        writer.writerow(i)

# getaccuracyrank()