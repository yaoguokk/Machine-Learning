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


# loadDataset 3 is to transfer all data to float
# loadDataset(0,1,'wine_train.csv','wine_test.csv', trainingSet, testSet)
# center=centerPoint(trainingSet,-1)
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


def printpot(trainingSet,label):
    print(label)
    for i in range(len(label)):
        if(label[i]==1):
            # print("it is 1")
            plt.scatter(trainingSet[i][0], trainingSet[i][1], s=50,color='r')
        elif (label[i] == 2):
            # print("it is 2")
            plt.scatter(trainingSet[i][0], trainingSet[i][1], s=20,color='b')

    plt.show()

# printpot(testSet,testlabel)


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

# main(0,1)
