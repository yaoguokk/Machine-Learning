
import csv
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
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

filename = "change50k.csv"
dataL50k=[]
dataS50k=[]
rawdata=[]

#read from files to two dataset 
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


# split the i th column of dataset and return this column 
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

    # print(intCol)
    # print(np.mean(cpage,axis=0))
    # print(np.var(cpage,axis=0))
    # print(cpage.shape)
    return cpage

#input the column(frature) you want to onehotcoding 
def oneHot(work):
    hotcode = OneHotEncoder(handle_unknown='ignore')
    hotcode.fit(work)
    a = hotcode.transform(work).toarray()
    return a



def naivebayes(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    predic = clf.predict(X_test)
    return accuracy_score(y_test,predic)

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
    confuse_matrix = confusion_matrix(y_test,predic,labels=[1,-1])
    return accuracy_score(y_test,predic)
    # print('accuracy is %.2f' %accuracy_score(y_test,predic))

def linear_svc(X_train, X_test, y_train, y_test):
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X_train, y_train)
    predic = clf.predict(X_test)
    confuse_matrix = confusion_matrix(y_test,predic,labels=[1,-1])
    return accuracy_score(y_test, predic)


def SMOTE1(X,label):
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(X,label)
    return X_smo, y_smo



# for test the test dataset 
def readTestData(filename):
    dataL50k=[]
    dataS50k=[]
    rawdata=[]
    readfile(filename,dataL50k,dataS50k,rawdata)

    stand_age=standardize(rawdata,1,"float")
    stand_working_hours = standardize(rawdata,-3,"float")

    work = transfertoFloat(rawdata, 2,"d")
    work_oneHot= oneHot(work)

    Relationship = transfertoFloat(rawdata,8,"ds")
    Relationship_oneHot = oneHot(Relationship)

    education = transfertoFloat(rawdata,4,"ds")
    education_oneHot=oneHot(education)

    occupation = transfertoFloat(rawdata,7,"ds")
    occu_onehot = oneHot(occupation)
    # print("occupation train",np.unique(occupation))

    race= transfertoFloat(rawdata,9,"ds")
    race_onehot = oneHot(race)


    label = transfertoFloat(rawdata,15,"int",2)
    X = np.concatenate((stand_age,stand_working_hours,Relationship_oneHot,work_oneHot,education_oneHot),axis=1)
    return X, label



# input a list, return its all permutations 
def get_power_set(s):
    #https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    #thanks to newacct, i learn how to produce permutation on a given number of list with the help of his python code
    power_set=[[]]
    for elem in s:
    # iterate over the sub sets so far
        for sub_set in power_set:
      # add a new subset consisting of the subset at hand added elem
            power_set=power_set+[list(sub_set)+[elem]]
    return power_set

# input a select list,combine the feature to form a feature pairs,  and implement specific ML model resuturn its accurcay
def feature_select(select_list,data_list,label):
    if(len(select_list)==1):
        X = data_list[select_list[0]]
    else:
        X = data_list[select_list[0]]
        for i in range(1,len(select_list)):
            X = np.concatenate((X,data_list[select_list[i]]),axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.30, random_state=42)
    result = Perceptron1(X,X,label,label)
    return result
    
# the main function, perform feature selection for a specific model with all possible pairs
def using_feature_get_best_combination():
    filename = "change50k.csv"
    dataL50k=[]
    dataS50k=[]
    rawdata=[]
    readfile(filename,dataL50k,dataS50k,rawdata)

    stand_age=standardize(rawdata,1,"float")


    work = transfertoFloat(rawdata, 2,"d")
    work_oneHot= oneHot(work)


    education = transfertoFloat(rawdata,4,"ds")
    education_oneHot=oneHot(education)

    education_number = standardize(rawdata,5,"float")

    marital_status = transfertoFloat(rawdata,6,"ds")
    marital_oneHot = oneHot(marital_status)

    occupation = transfertoFloat(rawdata,7,"ds")
    occu_onehot = oneHot(occupation)

    Relationship = transfertoFloat(rawdata,8,"ds")
    Relationship_oneHot = oneHot(Relationship)

    race= transfertoFloat(rawdata,9,"ds")
    race_onehot = oneHot(race)

    stand_working_hours = standardize(rawdata,-3,"float")

    sex = transfertoFloat(rawdata,10,"ds")
    sex_onehot = oneHot(sex)

    country = transfertoFloat(rawdata,14,"ds")
    country_onehot= oneHot(country)

    label = transfertoFloat(rawdata,15,"int",2)
    
    data_list = [stand_age,work_oneHot,education_oneHot,education_number,marital_oneHot,occu_onehot,Relationship_oneHot,race_onehot,sex_onehot,stand_working_hours,country_onehot]

    data_list_name = ["Age","Workclass","Education","Education-num","Marital-status","Occupation","Relationship","Race","Sex","Working_hours","Country"]

    data_list_name1 = ["age","working_hours","workclass","Relationship","education_oneHot","occu_onehot","race_onehot","sex_onehot"]

    a = [i for i in range(len(data_list))]

    select_list = get_power_set(a)
    result  = []
    for i in range(1,len(select_list)):
        a = feature_select(select_list[i],data_list,label)
        result.append([a,select_list[i]])

    result.sort(key = lambda ele: ele[0],reverse = True)

    name = result[0][1]

    with open("result_feature_10feature_Perceptron.csv","w") as csvfile:
        csvwrite = csv.writer(csvfile)
        for i in result:
            csvwrite.writerow(i)

            
    # manual select
    # data = np.concatenate((work_oneHot,education_number,marital_oneHot),axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.30, random_state=42)
    # result = Perceptron1(X_train,X_test,y_train,y_test)
    # print(data.shape)
    # print(result)

#    select_list = [2,5,6,8,10]
#    X = data_list[2]
#    for i in select_list:
#        print(data_list_name[i])
#        
#    for i in range(1,len(select_list)):
#        X = np.concatenate((X,data_list[select_list[i]]),axis=1)
#    result = Perceptron1(X,X,label,label)
#    with open("X.csv","w") as csvfile:
#        csvwrite = csv.writer(csvfile)
#        for i in X:
#            csvwrite.writerow(i)


import time
start = time.time()
using_feature_get_best_combination()

end = time.time()
print("time is ",end-start)
