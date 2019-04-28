import csv
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


def readfile(filename,dataL50k,dataS50k,rawdata):
    with open(filename,"r") as csvfile:
        raw = csv.reader(csvfile)
        raw = list(raw)
        row = []
        for row in raw:
            rawdata.append(row)
            if "1" == row[-1]:
                dataL50k.append(row)
            elif "-1" == row[-1]:
                dataS50k.append(row)

def readfile_InTwoSet(filename1,datapos,dataneg,exclude_list):
    with open(filename1,"r") as csvfile:
        raw = csv.reader(csvfile)
        raw = list(raw)
        temp1 = []
        count = 1
        for row in range(1,len(raw)):
            # temp= []
            # for col in range(len(raw[0])):
            #     if col not in exclude_list:
            #         temp.append(raw[row][col])
            # temp1.append(temp)

            if "pos" in raw[row][0]:
                raw[row][0]="1"
                tempPos= []
                for col in range(len(raw[0])):
                    if col not in exclude_list:
                        tempPos.append(raw[row][col])
                datapos.append(tempPos)
                temp1.append(tempPos)
            if "neg" in raw[row][0]:
                raw[row][0]="-1"
                tempNeg= []
                for col in range(len(raw[0])):
                    if col not in exclude_list:
                        tempNeg.append(raw[row][col])
                dataneg.append(tempNeg)
                temp1.append(tempNeg)
    return temp1

def transfertoFloat(rawdata,feature,type="float",size=1):
    rawdata= np.array(rawdata)
    age = rawdata[:,feature-1:feature]
    #transfer to 1*10320 1D array
    cpage= np.array([x[0] for x in age])
    #transfer char type to float
    if(type == "float"):
        cpage = cpage.astype(np.float)
    elif(type == "int"):
        cpage = cpage.astype(np.int)
    # reshape the row into column size
    if(size==1):
        cpage = cpage.reshape(len(cpage),1)
    else:
        cpage = cpage.reshape(1,len(cpage))
        cpage=cpage[0]
    return cpage

#inpout the rawdata and the index of feature and outpot the standarized column
def standardize(rawdata,feature,type):
    cpage= transfertoFloat(rawdata,feature,"float")
    scaler =  StandardScaler()
    scaler.fit(cpage)
    cpage = scaler.transform(cpage)

    return cpage

#input the column(frature) you want to onehotcoding 
def oneHot(work):
    hotcode = OneHotEncoder()
    hotcode.fit(work)
    a = hotcode.transform(work).toarray()
    return a



def outputfile(finename,matrix):
    filename1="after_onehot.csv"
    df1 = pd.DataFrame(matrix)
    df1.to_csv(filename1, sep=',',index = False,header=False)

def naivebayes(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    predic = clf.predict(X_test)
    confuse_matrix  = confusion_matrix(y_test,predic,labels=[1,-1])
    return accuracy_score(y_test,predic) , confuse_matrix
    print('accuracy is %.2f' %accuracy_score(y_test,predic))

def svm1(X_train, X_test, y_train, y_test,C,gamma):
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    # X = np.concatenate((X_test,X_train),axis=0)
    # y = np.concatenate((y_train,y_test),axis=0)
    # scores = cross_val_score(clf, X, y, cv=5)

    #after smote
    X_train,y_train=SMOTE1(X_train,y_train)
    clf.fit(X_train,y_train)

    predic = clf.predict(X_test)
    confuse_matrix  = confusion_matrix(y_test,predic,labels=[1,-1])
    print(confuse_matrix)
    print('accuracy is %.2f' %accuracy_score(y_test,predic))

def Perceptron1(X_train, X_test, y_train, y_test):
    clf = Perceptron(tol = 1e-3, random_state=0)
    clf.fit(X_train,y_train)
    predic = clf.predict(X_test)
    confuse_matrix  = confusion_matrix(y_test,predic,labels=[1,-1])
    return accuracy_score(y_test,predic) , confuse_matrix
    print('accuracy is %.2f' %accuracy_score(y_test,predic))

def linear_svc(X_train, X_test, y_train, y_test):
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X_train, y_train)
    predic = clf.predict(X_test)
    confuse_matrix  = confusion_matrix(y_test,predic,labels=[1,-1])

    return accuracy_score(y_test, predic), confuse_matrix




def SMOTE1(X,label):
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(X,label)
    return X_smo, y_smo


def compareNaicebayes_before_after_smoth(X,label):

    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.30, random_state=42)
    print("no smote")
    naivebayes(X_train,X_test,y_train,y_test)
    print("after smote")
    x,y = SMOTE1(X,label)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    naivebayes(X_train,X_test,y_train,y_test)

def generateRandomIndexList(data_length):
    length = random.randint(1,data_length)
    a = [i for i in range(data_length)]

    select_list = random.sample(a,length)
    select_list.sort()
    return select_list

def feature_select(data_train,data_label,num):
    result = []
    length = len(data_train[0])
    for i in range(num):
        newlist = generateRandomIndexList(length);
        feature_list = np.array(newlist)
        list_len = len(feature_list)
        train = data_train[:,feature_list]
        # X_train, X_test, y_train, y_test = train_test_split(train, data_label, test_size=0.30, random_state=42)
        a,matrix = Perceptron1(train, train, data_label, data_label)
        result.append([a,list_len,newlist,matrix])
    return result

def test_list(data_train,data_label):
    newlist=[0, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 29, 31, 32, 33, 34, 35, 36, 37, 42, 43, 44, 47, 48, 50, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 94, 95, 98, 99, 100, 102, 103, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117]
    train = data_train[:,np.array(newlist)]
    a,matrix = Perceptron1(train, train, data_label, data_label)
    print(a)
    print(matrix)



def main1():
    file_label = "washed_data_label_52.csv"
    file_train = "washed_data_train_52.csv"
    data_label = np.genfromtxt(file_label, delimiter=',')
    data_train = np.genfromtxt(file_train, delimiter=',')
    data_length = len(data_train[0])

    #choose the times you want to run 
    num = 100
    result = feature_select(data_train,data_label,num)
    result.sort(key = lambda ele: ele[0],reverse = True)

    with open("result_feature_select_Perceptron.csv","w") as csvfile:
        csvwrite = csv.writer(csvfile)
        for i in result:
            csvwrite.writerow(i)
    print("total "+str(num)+" times ")
    
    test_list(data_train,data_label)

import time
start = time.time()
main1()
end = time.time()
print("time is ",end-start)