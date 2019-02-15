import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import random
from plotDecBoundaries1vsR import plotDecBoundaries

def loadDataset(feature1,feature2,trainfilename,testfilename, trainingSet=[], testSet=[]):
    with open(trainfilename, 'r') as csvfile1:
        lines = csv.reader(csvfile1)

        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [float(dataset[x][feature1]), float(dataset[x][feature2]), 1]
            trainingSet.append(temp)


    with open(testfilename, 'r') as csvfile2:
        lines = csv.reader(csvfile2)

        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [float(dataset[x][feature1]), float(dataset[x][feature2]), 1]
            testSet.append(temp)

def loadLabel(trainfilename,testfilename,trainingSet=[], testSet=[]):
    with open(trainfilename, 'r') as csvfile1:
        lines = csv.reader(csvfile1)
        dataset = list(lines)

        for x in range(len(dataset)):
            temp=int(dataset[x][-1])
            trainingSet.append(temp)

    with open(testfilename, 'r') as csvfile2:
        lines = csv.reader(csvfile2)
        dataset = list(lines)
        for x in range(len(dataset)):
            temp = int(dataset[x][-1])
            testSet.append(temp)

#test load
trainingSet=[]
testSet=[]
trainlabel=[]
testlabel=[]
# loadDataset(0,1,'feature_train.csv','feature_test.csv',trainingSet,testSet)
# loadLabel('label_train.csv','label_test.csv',trainlabel,testlabel)
loadDataset(0,1,'synthetic2_train.csv','synthetic2_test.csv',trainingSet,testSet)
loadLabel('synthetic2_train.csv','synthetic2_test.csv',trainlabel,testlabel)
# loadDataset(0,1,'synthetic1_train.csv','synthetic1_test.csv',trainingSet,testSet)
# loadLabel('synthetic1_train.csv','For Synthetic 2, the the smallest error For Synthetic 2, the the smallest error synthetic1_test.csv',trainlabel,testlabel)


def shuffle(DataSet):
    random.seed=[12]
    length = len(DataSet)
    shuffle = [i for i in range(length)]
    shufflelist = random.sample(shuffle, length)
    return shufflelist




def GD(trainSet,trainlabel,sl):
    x=np.array(trainSet)#trainset nparray
    label=np.array(trainlabel) # label nparray
    w=np.array([[0.1,0.1,0.1]]) # initial w

    factor=1
    maxrange=100
    for m in range(1,maxrange):
        count = 0
        Temp = np.array([-100000])
        for n in range(len(trainSet)):
            i = (m-1)*len(trainSet)+n

            if(label[sl[n]]==1):
                zn=1
            else:
                zn=-1
            judge = np.dot(x[sl[n]], w[i].T)*zn
            # print("judge =",judge)
            if (judge > 0):
                count+=1
                w = np.append(w, [w[i]], axis=0)
            else:
                wnew = w[n] + factor * zn * x[sl[n]]

                w = np.append(w, [wnew], axis=0)
                if(m==maxrange-1):
                    if Temp<judge:
                        Temp = judge
                        store = wnew


        # print("count is ",count)
        if(count==len(trainSet)):
            print("jump out! at iteration ",m)
            # print(w)
            return w[i]

    return store


w=GD(trainingSet,trainlabel,shuffle(trainingSet))

print(w)

def errorRate(trainSet,label,w):
    w=np.array(w)
    x=np.array(trainSet)
    l=np.array(label)
    count=0
    for i in range(len(x)):
        if (l[i] == 1):
            zn = 1
        else:
            zn = -1
        judge=np.dot(x[i],w.T)*zn
        if(judge<0):
            count+=1
    # print(count)
    return (float(count/len(trainSet)))

print ("testSet error rate ",errorRate(testSet,testlabel,w))
print ("TrainSet error rate ",errorRate(trainingSet,trainlabel,w))

train = np.array(trainingSet)
label = np.array(trainlabel)
plotDecBoundaries(train,label,w)


# x=np.array(trainingSet)#trainset nparray
# sl=shuffle(trainingSet)
# label=np.array(trainlabel) # label nparray
# w=np.array([[0.1,0.1,0.1]]) # initial w
# k=np.array([[3,5,4]])
# print(np.dot(w.T, k))
# # for m in range(1,1001):

# w=np.array([-1])
# m=np.array([0])
# print(w<m)



