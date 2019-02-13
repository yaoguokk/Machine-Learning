import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import operator
from plotDecBoundaries import plotDecBoundaries
from plotDecBoundaries1vsR import plotDecBoundaries

def loadDataset(feature1,feature2,trainfilename,testfilename, trainingSet=[], testSet=[]):
    with open(trainfilename, 'r') as csvfile1:
        lines = csv.reader(csvfile1)

        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [float(dataset[x][feature1]), float(dataset[x][feature2]), int(dataset[x][-1])]
            trainingSet.append(temp)


    with open(testfilename, 'r') as csvfile2:
        lines = csv.reader(csvfile2)

        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [float(dataset[x][feature1]), float(dataset[x][feature2]), int(dataset[x][-1])]
            testSet.append(temp)



def centerPoint(trainSet,a=0):
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
    list0 = [[x1mean, y1mean, 1], [x2mean, y2mean, 2], [x3mean, y3mean, 3]]
    x12mean = (x1sum + x2sum) / (count2 + count1)
    y12mean = (y1sum + y2sum) / (count1 + count2)
    x23mean = (x2sum + x3sum) / (count2 + count3)
    y23mean = (y2sum + y3sum) / (count3 + count2)
    x13mean = (x1sum + x3sum) / (count1 + count3)
    y13mean = (y1sum + y3sum) / (count1 + count3)

    list01 = [x23mean, y23mean,-1]
    list02 = [x13mean, y13mean,-2]
    list03 = [x12mean, y12mean,-3]

    if a==1:
        return list01
    elif a==2:
        return list02
    elif a==3:
        return list03
    elif a==-1:
        return list0+[list01]+[list02]+[list03]
    else:
        return list0


trainingSet=[]
testSet=[]
# loadDataset 3 is to transfer all data to float
loadDataset(0,1,'wine_train.csv','wine_test.csv', trainingSet, testSet)
center=centerPoint(trainingSet,-1)
# print(center)

def prepareforExplot1vsR(trainingSet):
    train_set = np.array(trainingSet)
    trainxy = np.array(train_set[:, 0:2])
    trainlable = np.array([int(i) for i in train_set[:, 2:3]])  # must be int
    mean = centerPoint(trainingSet,-1)
    mean = np.array(mean)  # transfer all mean[] to float, dtype=np.float32
    mean = mean[:, :2]#[[1st feature],[2nd feature]]
    # plot the boundary with function
    plotDecBoundaries(trainxy, trainlable, mean)

# prepareforExplot1vsR(trainingSet)


def prepareforExplot(trainingSet,a=0):
    train_set = np.array(trainingSet)
    trainxy = np.array(train_set[:, 0:2])
    trainlable = np.array([int(i) for i in train_set[:, 2:3]])  # must be int
    list1=[1,2,3]
    if(a in list1):
        list1.remove(a)
        for i in range(len(trainlable)):
            if  trainlable[i] in list1:
                trainlable[i]= 0
    mean = centerPoint(trainingSet,a)

    mean = np.array(mean)  # transfer all mean[] to float, dtype=np.float32
    mean = mean[:, :2]#[[1st feature],[2nd feature]]
    # plot the boundary with function
    plotDecBoundaries(trainxy, trainlable, mean,a)

# prepareforExplot(trainingSet,2)


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

# difine the euclidieanDistance of two set, return the distance
def euclidieanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)



def nearest1(centerpoint, testInstance):
    length = len(testInstance)-1

    dist3 = euclidieanDistance(testInstance, centerpoint[2], length)  # ->3
    dist03 = euclidieanDistance(testInstance, centerpoint[5], length)  # ->!3
    dist2 = euclidieanDistance(testInstance, centerpoint[1], length)  # ->2
    dist02 = euclidieanDistance(testInstance,centerpoint[4], length)  # ->!2
    dist1 = euclidieanDistance(testInstance, centerpoint[0], length)  # ->1
    dist01 = euclidieanDistance(testInstance, centerpoint[3], length)  # ->!1

    if dist1 < dist01 and dist2 > dist02 and dist3 > dist03:
        return [1]
    elif dist2 < dist02 and dist1 > dist01 and dist3 > dist03:
        return [2]
    elif dist3 < dist03 and dist2 > dist02 and dist1 > dist01:
        return [3]
    else:
        return [0]
     #return the train point which cloest to the testpoint



def main(a,b):
    # prepare data
    count= 0
    trainingSet = []
    testSet = []
    loadDataset(a,b,'wine_train.csv','wine_test.csv', trainingSet, testSet)
    # generate predictions
    cPoint  = centerPoint(trainingSet,-1)
    print(cPoint)
    set = testSet
    for x in range(len(set)):
        result = nearest1(cPoint, set[x])
        if set[x][-1]==result[0]:
            count=count+1
        #     plt.scatter(set[x][0], set[x][1], s=20, color='r')
        # else:
        #     plt.scatter(set[x][0], set[x][1], s=20, color='b')
        # print('> predicted=' + repr(result[0]) + ', actual=' + repr(set[x][-1]))
    accuracy = count/len(set)
    print("the accuracy of testset is ",end="")
    print(accuracy)
    # print('>error rate of feature '+str(a)+' and feature '+str(b) +" is " +str((round(1-accuracy,2))))
    # plt.show()
    # return [round(1-accuracy,3),a+1,b+1]
    set = trainingSet
    lenth = len(set)
    count = 0
    for x in range(lenth):
        result = nearest1(cPoint, set[x])
        if set[x][-1] == int(result[0]):
            count = count + 1
        #     plt.scatter(set[x][0], set[x][1], s=20, color='r')
        # else:
        #     plt.scatter(set[x][0], set[x][1], s=20, color='b')
        # print('> predicted=' + repr(result[0]) + ', actual=' + repr(set[x][-1]))
    accuracy = count / len(set)
    print("the accuracy of trainset is ", end="")
    print(accuracy)

main(0,1)
