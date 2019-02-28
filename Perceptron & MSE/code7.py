
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import random
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def loadDataset(feature1,feature2,trainfilename,testfilename, trainingSet=[], testSet=[]):
    with open(trainfilename, 'r') as csvfile1:
        lines = csv.reader(csvfile1)

        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [1,float(dataset[x][feature1]), float(dataset[x][feature2])]
            trainingSet.append(temp)


    with open(testfilename, 'r') as csvfile2:
        lines = csv.reader(csvfile2)

        dataset = list(lines)
        for x in range(len(dataset)):
            temp = [1,float(dataset[x][feature1]), float(dataset[x][feature2])]
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


def loadDataLabel(filename):
    data=[]
    label=[]
    with open(filename,newline='') as csvfile:
        oneline =  csv.reader(csvfile)
        for row in oneline:
            data.append([float(s) for s in row[:-1]])
            label.append(int(row[-1]))
    return (np.array(data),np.array(label))


(data_train, label_train)=loadDataLabel('wine_train.csv')
(data_test, label_test)=loadDataLabel('wine_test.csv')


# mean_train = np.mean(data_train,axis = 0)
def problemb():
    # before standardizing
    print('before standardizing\n')

    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('train data column means \n',np.mean(data_train,axis = 0))
    print('train data column standard deviation \n',np.std(data_train,axis = 0))
    print('test data column means \n',np.mean(data_test,axis = 0))
    print('test data column standard deviation \n',np.std(data_test,axis = 0))

    copy_data_train= np.array([x for x in data_train])#deep copy
    copy_data_test = np.array([x for x in data_test])#deep copy

    # to standardize the training data
    print('after standardize')
    scaler = StandardScaler()
    scaler.fit(copy_data_train)#standard factor is from train data

    data_standard_train = scaler.transform(copy_data_train)
    print('train data column means \n',np.mean(data_standard_train,axis = 0))
    print('train data column standard deviation \n',np.std(data_standard_train,axis = 0))

    data_standard_test = scaler.transform(copy_data_test)
    print('train data column means \n',np.mean(data_standard_test,axis = 0))
    print('train data column standard deviation \n',np.std(data_standard_test,axis = 0))

# problemb()
# c.
def problemc():
    scaler = StandardScaler()

    copy_data_train = np.array([x for x in data_train])  # deep copy
    copy_data_test = np.array([x for x in data_test])#deep copy
    scaler.fit(copy_data_train)
    data_standard_train = scaler.transform(copy_data_train)
    data_standard_test = scaler.transform(copy_data_test)

    scaler.fit(copy_data_train)
    ppn = Perceptron()
    ppn.fit(data_standard_train, label_train)
    print('the final weight vectors for nonaugmented w is',ppn.coef_)
    print('the final weight vectors for w0 is ',ppn.intercept_)
    test_predit = ppn.predict(data_standard_test)
    print('accuracy is %.2f' %accuracy_score(label_test,test_predit))

# problemc()
#d

copy_data_train = np.array([x for x in data_train])  # deep copy
copy_data_test = np.array([x for x in data_test])  # deep copy
scaler = StandardScaler()
scaler.fit(copy_data_train)
data_standard_train = scaler.transform(copy_data_train)
data_standard_test = scaler.transform(copy_data_test)

def problemD():
    print('one vs. rest')
    print('for 1st and 2nd feature')
    index=0
    weight = {1:1,2:2,3:3}
    ovr=Perceptron(tol=1e-3, random_state=index,class_weight=weight)
    #using standard train to train model
    ovr.fit(data_standard_train[:,:2],label_train)
    print('the final weight vectors for nonaugmented w is ',ovr.coef_)
    print('the final weight vectors for w0 is ',ovr.intercept_)

    # predict the train label
    predict_train = ovr.predict(data_standard_train[:,:2])
    errorate_train = accuracy_score(label_train,predict_train)
    print('accuracy of train is %.5f' %errorate_train)

    # predict the test label
    predict_test = ovr.predict(data_standard_test[:,:2])
    errorate_test = accuracy_score(label_test,predict_test)
    print('accuracy of test is %.5f' %errorate_test)

    print('using all 13 features to train')
    ovr1 = Perceptron(tol=1e-3, random_state=index,class_weight=weight)
    # using standard train to train model
    ovr1.fit(data_standard_train, label_train)
    print('the final weight vectors for nonaugmented w is ', ovr1.coef_)
    print('the final weight vectors for w0 is ', ovr1.intercept_)

    # predict the train label
    predict_train1 = ovr1.predict(data_standard_train)
    errorate_train1 = accuracy_score(label_train, predict_train1)
    print('accuracy of train is %.5f' % errorate_train1)

    # predict the test label
    predict_test1 = ovr1.predict(data_standard_test)
    errorate_test1 = accuracy_score(label_test, predict_test1)
    print('accuracy of test is %.5f' % errorate_test1)

# problemD()

def perceptron2and13(count,weight,result2feature,result13feature):
    index=0

    ovr = Perceptron(tol=1e-3, random_state=index,class_weight=weight)
    # using standard train to train model
    ovr.fit(data_standard_train[:, :2], label_train)

    # predict the train label
    predict_train = ovr.predict(data_standard_train[:, :2])
    errorate_train = accuracy_score(label_train, predict_train)

    # predict the test label
    predict_test = ovr.predict(data_standard_test[:, :2])
    errorate_test = accuracy_score(label_test, predict_test)
    result2feature.append([count, errorate_train,errorate_test,ovr.coef_,ovr.intercept_])

    # 'using all 13 features to train'
    # ovr = OneVsRestClassifier(LinearSVC(random_state=0))
    ovr1 = Perceptron(tol=1e-3, random_state=index,class_weight=weight)
    # using standard train to train model
    ovr1.fit(data_standard_train, label_train)

    # predict the train label
    predict_train1 = ovr1.predict(data_standard_train)
    errorate_train1 = accuracy_score(label_train, predict_train1)

    # predict the test label
    predict_test1 = ovr1.predict(data_standard_test)
    errorate_test1 = accuracy_score(label_test, predict_test1)
    result13feature.append([count, errorate_train1,errorate_test1,ovr1.coef_,ovr1.intercept_])

def problemE():
    result2feature =[]
    result13feature=[]
    lower=-3
    upper=3
    for count in range(100):
        weight={1:random.randint(lower,upper),2:random.randint(lower,upper),3:random.randint(lower,upper)}
        perceptron2and13(count,weight,result2feature,result13feature)
    temp_2fea=result2feature[0]
    temp_13fea=result13feature[0]
    for i in range(100):
        if result2feature[i][1]>temp_2fea[1]:
            temp_2fea=result2feature[i]
        if result13feature[i][1]>temp_13fea[1]:
            temp_13fea=result13feature[i]

    print('after 100 random chooseing initial weight')
    print('in 2 feature the maximum accuracy for train set is \n',temp_2fea[1],'\nthe accuracy in test set is \n',temp_2fea[2])
    print('the final weight vectors for nonaugmented w is ', temp_2fea[3])
    print('the final weight vectors for w0 is ', temp_2fea[4])

    print('\nin 13 feature the maximum accuracy for train set is \n', temp_13fea[1], '\nthe accuracy in test set is \n',
          temp_13fea[2])
    print('the final weight vectors for nonaugmented w is ', temp_13fea[3])
    print('the final weight vectors for w0 is ', temp_13fea[4])


class MSE_binary(LinearRegression):
    def __init__(self):
        # print("calling  newly creatly MSE binary function")
        super(MSE_binary,self).__init__()
    def predict(self, X):
        thr = 0.5
        y = self._decision_function(X)
        newy = [int(i+thr) for i in y]
        return newy

def problemg(data_train,data_test,index=0):
    if (index):
        scaler = StandardScaler()
        scaler.fit(data_train)  # standard factor is from train data
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    binary_model13 = MSE_binary()
    mc_model = OneVsRestClassifier(binary_model13)
    mc_model.fit(data_train,label_train)
    predictNew = mc_model.predict(data_test)
    accuracy = accuracy_score(label_test, predictNew)
    print('the accuracy for 13 feature test data is', accuracy)

    binary_model2 = MSE_binary()
    mc_model1 = OneVsRestClassifier(binary_model2)
    mc_model1.fit(data_train[:,0:2], label_train)
    predictNew2 = mc_model1.predict(data_test[:,0:2])
    accuracy2 = accuracy_score(label_test, predictNew2)
    print('the accuracy for 2 feature test data is', accuracy2)



problemg(data_train,data_test)

problemg(data_train,data_test,1)

