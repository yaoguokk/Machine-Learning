# EE559 Project
# Written by Qi Zhao
# April 20, 2019

import csv
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# This method read the data set
def load_data_set(filename):

    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        data_set = list(lines)
        input_set = copy.deepcopy(data_set)

    return input_set
    pass


# This function perform SMOTE
def smote(input_set, label):
    smo = SMOTE(random_state=33)
    x_smo, y_smo = smo.fit_sample(input_set, label)
    return x_smo, y_smo
    pass


# This function performs Perceptron
def perceptron():

    # Load the data
    train_set_raw = load_data_set('change50k.csv')
    test_set_raw = load_data_set('test_50k.csv')

    # Handle data
    # Remove the first row of the data set
    list_no_unknown = []
    for i in range(len(train_set_raw)):
        list_no_unknown.append(train_set_raw[i])

    # for test
    list_no_unknown_test = []
    for i in range(len(test_set_raw)):
        list_no_unknown_test.append(test_set_raw[i])

    # Create a new list with features we want
    # Chosen features:f4,f7,f8,f10,f14
    new_list = []
    for j in range(len(list_no_unknown)):
        temp = [list_no_unknown[j][3], list_no_unknown[j][6], list_no_unknown[j][7], list_no_unknown[j][9],
                list_no_unknown[j][13], list_no_unknown[j][-1]]

        new_list.append(temp)

    print(new_list[0])

    # for test
    new_list_test = []
    for j in range(len(list_no_unknown_test)):
        temp2 = [list_no_unknown_test[j][3], list_no_unknown_test[j][6], list_no_unknown_test[j][7],
                 list_no_unknown_test[j][9], list_no_unknown_test[j][13], list_no_unknown_test[j][-1]]

        new_list_test.append(temp2)

    print(new_list_test[0])

    # Convert the label to number
    # <= 50K = -1, >50K = 1
    label_set = []
    for item in new_list:
        if item[-1] == '-1':
            item[-1] = -1
            label_set.append(item[-1])

        else:
            item[-1] = 1
            label_set.append(item[-1])

    # for test
    label_set_test = []
    for item in new_list_test:
        if item[-1] == '-1':
            item[-1] = -1
            label_set_test.append(item[-1])

        else:
            item[-1] = 1
            label_set_test.append(item[-1])

    education = []
    occupation = []
    relation = []
    sex = []
    country = []
    for item in new_list:
        education.append([item[0]])
        occupation.append([item[1]])
        relation.append([item[2]])
        sex.append([item[3]])
        country.append([item[4]])

    # For test
    education_test = []
    occupation_test = []
    relation_test = []
    sex_test = []
    country_test = []
    for item in new_list_test:
        education_test.append([item[0]])
        occupation_test.append([item[1]])
        relation_test.append([item[2]])
        sex_test.append([item[3]])
        country_test.append([item[4]])

    # One hot encode for education
    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(education)
    education_encode = enc.transform(education).toarray()
    education_test_encode = enc.transform(education_test).toarray()

    # One hot encode for occupation
    enc.fit(occupation)
    occupation_encode = enc.transform(occupation).toarray()
    occupation_test_encode = enc.transform(occupation_test).toarray()

    # One hot encode for relation
    enc.fit(relation)
    relation_encode = enc.transform(relation).toarray()
    relation_test_encode = enc.transform(relation_test).toarray()

    # One hot encode for sex
    enc.fit(sex)
    sex_encode = enc.transform(sex).toarray()
    sex_test_encode = enc.transform(sex_test).toarray()

    # One hot encode for country
    enc.fit(country)
    country_encode = enc.transform(country).toarray()
    country_test_encode = enc.transform(country_test).toarray()

    # Combine features
    train_set = np.concatenate((education_encode, occupation_encode, relation_encode, sex_encode, country_encode),
                               axis=1)

    test_set = np.concatenate((education_test_encode, occupation_test_encode, relation_test_encode, sex_test_encode,
                               country_test_encode), axis=1)

    print(train_set.shape)
    print(test_set.shape)

    # Perform training
    clf = Perceptron(random_state=0, tol=1e-3)
    clf.fit(train_set, label_set)

    predict = clf.predict(train_set)
    accuracy = accuracy_score(label_set, predict)

    print('Accuracy is ' + repr(accuracy))

    c_m = confusion_matrix(label_set, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(label_set, predict)
    print('F1 score is ' + repr(f1))

    # test
    predict_test = clf.predict(test_set)
    accuracy_test = accuracy_score(label_set_test, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(label_set_test, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(label_set_test, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    # With SMOTE
    # SMOTE on train data
    f_smo, l_smo = smote(train_set, label_set)
    clf.fit(f_smo, l_smo)

    predict = clf.predict(f_smo)
    accuracy = accuracy_score(l_smo, predict)
    print('After SMOTE')
    print('Accuracy is ' + repr(accuracy))

    c_m = confusion_matrix(l_smo, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(l_smo, predict)
    print('F1 score is ' + repr(f1))

    # test
    predict_test = clf.predict(test_set)
    accuracy_test = accuracy_score(label_set_test, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(label_set_test, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(label_set_test, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    pass


# This function performs cross-validation for LinearSVC
def cv_linearsvc():
    # Load the data
    train_set_raw = load_data_set('change50k.csv')

    # Handle data
    # Remove the first row of the data set
    list_no_unknown = []
    for i in range(len(train_set_raw)):
        list_no_unknown.append(train_set_raw[i])

    # Create a new list with features we want
    # Chosen features:f1,f2,f4,f6,f7,f8,f9,f10,f13
    new_list = []
    for j in range(len(list_no_unknown)):
        temp = [list_no_unknown[j][0], list_no_unknown[j][1], list_no_unknown[j][3], list_no_unknown[j][5],
                list_no_unknown[j][6], list_no_unknown[j][7], list_no_unknown[j][8], list_no_unknown[j][9],
                list_no_unknown[j][12], list_no_unknown[j][-1]]

        new_list.append(temp)

    print(new_list[0])

    # Convert the label to number
    # <= 50K = -1, >50K = 1
    label_set = []
    for item in new_list:
        if item[-1] == '-1':
            item[-1] = -1
            label_set.append(item[-1])

        else:
            item[-1] = 1
            label_set.append(item[-1])

    # Perform standardization on numerical values
    age = []
    work_hr = []
    for item in new_list:
        temp1 = float(item[0])
        temp2 = float(item[8])
        age.append([temp1])
        work_hr.append([temp2])

    scale = StandardScaler()
    scale.fit(age)
    age_std = scale.transform(age)

    scale.fit(work_hr)
    work_hr_std = scale.transform(work_hr)

    # Perform One Hot Encode for chosen features
    workclass = []
    education = []
    marital_status = []
    occupation = []
    relation = []
    race = []
    sex = []
    for item in new_list:
        workclass.append([item[1]])
        education.append([item[2]])
        marital_status.append([item[3]])
        occupation.append([item[4]])
        relation.append([item[5]])
        race.append([item[6]])
        sex.append([item[7]])

    # One hot encode for workclass
    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(workclass)
    workclass_encode = enc.transform(workclass).toarray()

    # One hot encode for education
    enc.fit(education)
    education_encode = enc.transform(education).toarray()

    # One hot encode for marital_status
    enc.fit(marital_status)
    marital_status_encode = enc.transform(marital_status).toarray()

    # One hot encode for occupation
    enc.fit(occupation)
    occupation_encode = enc.transform(occupation).toarray()

    # One hot encode for relationship
    enc.fit(relation)
    relation_encode = enc.transform(relation).toarray()

    # One hot encode for race
    enc.fit(race)
    race_encode = enc.transform(race).toarray()

    # One hot encode for sex
    enc.fit(sex)
    sex_encode = enc.transform(sex).toarray()

    # Combine features
    train_set = np.concatenate((age_std, workclass_encode, education_encode, marital_status_encode,
                                occupation_encode, relation_encode, race_encode, sex_encode, work_hr_std), axis=1)

    c = [1, 10, 100, 1000]
    for i in range(4):
        clf = LinearSVC(random_state=0, tol=1e-4, C=c[i])
        cv_result = cross_val_score(clf, train_set, label_set, cv=3)
        print(cv_result)
    pass


# This function performs LinearSVC
def linearsvc():

    # Load the data
    train_set_raw = load_data_set('change50k.csv')
    test_set_raw = load_data_set('test_50k.csv')

    # Handle data
    # Remove the first row of the data set
    list_no_unknown = []
    for i in range(len(train_set_raw)):
        list_no_unknown.append(train_set_raw[i])

    # for test
    list_no_unknown_test = []
    for i in range(len(test_set_raw)):
        list_no_unknown_test.append(test_set_raw[i])

    # Create a new list with features we want
    # Chosen features:f1,f2,f4,f6,f7,f8,f9,f10,f13
    new_list = []
    for j in range(len(list_no_unknown)):
        temp = [list_no_unknown[j][0], list_no_unknown[j][1], list_no_unknown[j][3], list_no_unknown[j][5],
                list_no_unknown[j][6], list_no_unknown[j][7], list_no_unknown[j][8], list_no_unknown[j][9],
                list_no_unknown[j][12], list_no_unknown[j][-1]]

        new_list.append(temp)

    print(new_list[0])

    # for test
    new_list_test = []
    for j in range(len(list_no_unknown_test)):
        temp = [list_no_unknown_test[j][0], list_no_unknown_test[j][1], list_no_unknown_test[j][3],
                list_no_unknown_test[j][5], list_no_unknown_test[j][6], list_no_unknown_test[j][7],
                list_no_unknown_test[j][8], list_no_unknown_test[j][9], list_no_unknown_test[j][12],
                list_no_unknown_test[j][-1]]

        new_list_test.append(temp)

    print(new_list_test[0])

    # Convert the label to number
    # <= 50K = -1, >50K = 1
    label_set = []
    for item in new_list:
        if item[-1] == '-1':
            item[-1] = -1
            label_set.append(item[-1])

        else:
            item[-1] = 1
            label_set.append(item[-1])

    # for test
    label_set_test = []
    for item in new_list_test:
        if item[-1] == '-1':
            item[-1] = -1
            label_set_test.append(item[-1])

        else:
            item[-1] = 1
            label_set_test.append(item[-1])

    # Perform standardization on numerical values
    age = []
    work_hr = []
    for item in new_list:
        temp1 = float(item[0])
        temp2 = float(item[8])
        age.append([temp1])
        work_hr.append([temp2])

    # for test
    age_test = []
    work_hr_test = []
    for item in new_list_test:
        temp1 = float(item[0])
        temp2 = float(item[8])
        age_test.append([temp1])
        work_hr_test.append([temp2])

    scale = StandardScaler()
    scale.fit(age)
    age_std = scale.transform(age)
    age_test_std = scale.transform(age_test)

    scale.fit(work_hr)
    work_hr_std = scale.transform(work_hr)
    work_hr_test_std = scale.transform(work_hr_test)

    # Perform One Hot Encode for chosen features
    workclass = []
    education = []
    marital_status = []
    occupation = []
    relation = []
    race = []
    sex = []
    for item in new_list:
        workclass.append([item[1]])
        education.append([item[2]])
        marital_status.append([item[3]])
        occupation.append([item[4]])
        relation.append([item[5]])
        race.append([item[6]])
        sex.append([item[7]])

    # For test
    workclass_test = []
    education_test = []
    marital_status_test = []
    occupation_test = []
    relation_test = []
    race_test = []
    sex_test = []
    for item in new_list_test:
        workclass_test.append([item[1]])
        education_test.append([item[2]])
        marital_status_test.append([item[3]])
        occupation_test.append([item[4]])
        relation_test.append([item[5]])
        race_test.append([item[6]])
        sex_test.append([item[7]])

    # One hot encode for workclass
    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(workclass)
    workclass_encode = enc.transform(workclass).toarray()
    workclass_test_encode = enc.transform(workclass_test).toarray()

    # One hot encode for education
    enc.fit(education)
    education_encode = enc.transform(education).toarray()
    education_test_encode = enc.transform(education_test).toarray()

    # One hot encode for marital_status
    enc.fit(marital_status)
    marital_status_encode = enc.transform(marital_status).toarray()
    marital_status_test_encode = enc.transform(marital_status_test).toarray()

    # One hot encode for occupation
    enc.fit(occupation)
    occupation_encode = enc.transform(occupation).toarray()
    occupation_test_encode = enc.transform(occupation_test).toarray()

    # One hot encode for relationship
    enc.fit(relation)
    relation_encode = enc.transform(relation).toarray()
    relation_test_encode = enc.transform(relation_test).toarray()

    # One hot encode for race
    enc.fit(race)
    race_encode = enc.transform(race).toarray()
    race_test_encode = enc.transform(race_test).toarray()

    # One hot encode for sex
    enc.fit(sex)
    sex_encode = enc.transform(sex).toarray()
    sex_test_encode = enc.transform(sex_test).toarray()

    # Combine features
    train_set = np.concatenate((age_std, workclass_encode, education_encode, marital_status_encode,
                                occupation_encode, relation_encode, race_encode, sex_encode, work_hr_std), axis=1)

    test_set = np.concatenate((age_test_std, workclass_test_encode, education_test_encode, marital_status_test_encode,
                               occupation_test_encode, relation_test_encode, race_test_encode, sex_test_encode,
                               work_hr_test_std), axis=1)

    print(train_set.shape)
    print(test_set.shape)

    # Perform training
    clf = LinearSVC(random_state=0, tol=1e-4, C=1)
    clf.fit(train_set, label_set)

    predict = clf.predict(train_set)
    accuracy = accuracy_score(label_set, predict)

    print('Accuracy is ' + repr(accuracy))

    c_m = confusion_matrix(label_set, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(label_set, predict)
    print('F1 score is ' + repr(f1))

    # test
    predict_test = clf.predict(test_set)
    accuracy_test = accuracy_score(label_set_test, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(label_set_test, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(label_set_test, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    # With SMOTE
    # SMOTE on train data
    f_smo, l_smo = smote(train_set, label_set)
    clf.fit(f_smo, l_smo)

    predict = clf.predict(f_smo)
    accuracy = accuracy_score(l_smo, predict)
    print('After SMOTE')
    print('Accuracy is ' + repr(accuracy))

    c_m = confusion_matrix(l_smo, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(l_smo, predict)
    print('F1 score is ' + repr(f1))

    # test
    predict_test = clf.predict(test_set)
    accuracy_test = accuracy_score(label_set_test, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(label_set_test, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(label_set_test, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    pass


# This function performs naive bayes
def naive_bayes():

    # Load the data
    train_set_raw = load_data_set('change50k.csv')
    test_set_raw = load_data_set('test_50k.csv')

    # Handle data
    # Remove the first row of the data set
    list_no_unknown = []
    for i in range(len(train_set_raw)):
        list_no_unknown.append(train_set_raw[i])

    # for test
    list_no_unknown_test = []
    for i in range(len(test_set_raw)):
        list_no_unknown_test.append(test_set_raw[i])

    # Create a new list with features we want
    # Chosen features:f1,f2,f4,f6,f7,f8,f9,f10,f13
    new_list = []
    for j in range(len(list_no_unknown)):
        temp = [list_no_unknown[j][0], list_no_unknown[j][4], list_no_unknown[j][9], list_no_unknown[j][12],
                list_no_unknown[j][-1]]

        new_list.append(temp)

    print(new_list[0])

    # for test
    new_list_test = []
    for j in range(len(list_no_unknown_test)):
        temp = [list_no_unknown_test[j][0], list_no_unknown_test[j][4], list_no_unknown_test[j][9],
                list_no_unknown_test[j][12], list_no_unknown_test[j][-1]]

        new_list_test.append(temp)

    print(new_list_test[0])

    # Convert the label to number
    # <= 50K = -1, >50K = 1
    label_set = []
    for item in new_list:
        if item[-1] == '-1':
            item[-1] = -1
            label_set.append(item[-1])

        else:
            item[-1] = 1
            label_set.append(item[-1])

    # for test
    label_set_test = []
    for item in new_list_test:
        if item[-1] == '-1':
            item[-1] = -1
            label_set_test.append(item[-1])

        else:
            item[-1] = 1
            label_set_test.append(item[-1])

    # Perform standardization on numerical values
    age = []
    education_num = []
    work_hr = []
    for item in new_list:
        temp1 = float(item[0])
        temp2 = float(item[1])
        temp3 = float(item[3])
        age.append([temp1])
        education_num.append([temp2])
        work_hr.append([temp3])

    # for test
    age_test = []
    education_num_test = []
    work_hr_test = []
    for item in new_list_test:
        temp1 = float(item[0])
        temp2 = float(item[1])
        temp3 = float(item[3])
        age_test.append([temp1])
        education_num_test.append([temp2])
        work_hr_test.append([temp3])

    scale = StandardScaler()
    scale.fit(age)
    age_std = scale.transform(age)
    age_test_std = scale.transform(age_test)

    scale.fit(education_num)
    education_num_std = scale.transform(education_num)
    education_num_test_std = scale.transform(education_num_test)

    scale.fit(work_hr)
    work_hr_std = scale.transform(work_hr)
    work_hr_test_std = scale.transform(work_hr_test)

    # Perform One Hot Encode for chosen features
    sex = []
    for item in new_list:
        sex.append([item[2]])

    # For test
    sex_test = []
    for item in new_list_test:
        sex_test.append([item[2]])

    # One hot encode for workclass
    enc = OneHotEncoder(handle_unknown='ignore')

    # One hot encode for sex
    enc.fit(sex)
    sex_encode = enc.transform(sex).toarray()
    sex_test_encode = enc.transform(sex_test).toarray()

    # Combine features
    train_set = np.concatenate((age_std, education_num_std, sex_encode, work_hr_std), axis=1)

    test_set = np.concatenate((age_test_std, education_num_test_std, sex_test_encode, work_hr_test_std), axis=1)

    print(train_set.shape)
    print(test_set.shape)

    # Perform training
    clf = GaussianNB()
    clf.fit(train_set, label_set)

    predict = clf.predict(train_set)
    accuracy = accuracy_score(label_set, predict)

    print('Accuracy is ' + repr(accuracy))

    c_m = confusion_matrix(label_set, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(label_set, predict)
    print('F1 score is ' + repr(f1))

    # test
    predict_test = clf.predict(test_set)
    accuracy_test = accuracy_score(label_set_test, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(label_set_test, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(label_set_test, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    # With SMOTE
    # SMOTE on train data
    f_smo, l_smo = smote(train_set, label_set)
    clf.fit(f_smo, l_smo)

    predict = clf.predict(f_smo)
    accuracy = accuracy_score(l_smo, predict)
    print('After SMOTE')
    print('Accuracy is ' + repr(accuracy))

    c_m = confusion_matrix(l_smo, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(l_smo, predict)
    print('F1 score is ' + repr(f1))

    # test
    predict_test = clf.predict(test_set)
    accuracy_test = accuracy_score(label_set_test, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(label_set_test, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(label_set_test, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    pass

perceptron()