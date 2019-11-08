import csv
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
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


# This function perform perceptron
def perceptron():
    # load the data
    train_set_raw = load_data_set('washed_data_train_52.csv')
    train_label_raw = load_data_set('washed_data_label_52.csv')

    test_set_raw = load_data_set('washed_data_test52.csv')
    test_label_raw = load_data_set('washed_data_label_test52.csv')

    # convert string to number for train
    train_label = []
    for item in train_label_raw:
        train_label.append(float(item[0]))

    train_set = []
    for x in range(len(train_set_raw)):
        for y in range(len(train_set_raw[0])):
            train_set_raw[x][y] = float(train_set_raw[x][y])

        train_set.append(train_set_raw[x])

    train_set = np.array(train_set)

    print(train_set.shape)
    print(len(train_set))
    print(len(train_label))

    # for test data
    test_label = []
    for item in test_label_raw:
        test_label.append(float(item[0]))

    test_set = []
    for x in range(len(test_set_raw)):
        for y in range(len(test_set_raw[0])):
            test_set_raw[x][y] = float(test_set_raw[x][y])

        test_set.append(test_set_raw[x])

    test_set = np.array(test_set)

    print(test_set.shape)
    print(len(test_set))
    print(len(test_label))

    # Feature chosen
    # [2   4   5   6   8  10  11  12  14  15  17  18  19  20  21  22  24  26
    # 27  29  30  31  32  34  35  37  38  39  40  44  46  48  49  50  51  52
    # 53  54  55  56  57  58  59  60  61  64  66  68  69  73  75  76  77  79
    # 80  81  84  85  87  88  89  90  91  92  93  94  95  97  98 101 103 106
    # 108 110 113 114 116 117]

    feature_selection = [0, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 29, 31, 32, 33, 34,
                         35, 36, 37, 42, 43, 44, 47, 48, 50, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69,
                         70, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 94, 95, 98, 99, 100, 102, 103,
                         106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117]

    f_selection = np.array(feature_selection)
    new_train_set = train_set[:, f_selection]

    new_test_set = test_set[:, f_selection]

    # Normalization
    scale = StandardScaler()
    scale.fit(new_train_set)
    train_set_std = scale.transform(new_train_set)
    test_set_std = scale.transform(new_test_set)

    clf = Perceptron(random_state=0, tol=1e-3)
    clf.fit(train_set_std, train_label)
    predict = clf.predict(train_set_std)
    accuracy = accuracy_score(train_label, predict)

    print('Accuracy is ' + repr(accuracy))
    c_m = confusion_matrix(train_label, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(train_label, predict)
    print('F1 score is ' + repr(f1))

    # For test data
    predict_test = clf.predict(test_set_std)
    accuracy_test = accuracy_score(test_label, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(test_label, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(test_label, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    # With SMOTE
    f_smo, l_smo = smote(train_set_std, train_label)
    clf.fit(f_smo, l_smo)
    predict_smo = clf.predict(f_smo)
    accuracy_smo = accuracy_score(l_smo, predict_smo)
    print('Accuracy(SMOTE) is ' + repr(accuracy_smo))
    c_m_smo = confusion_matrix(l_smo, predict_smo, labels=[1, -1])
    print('Confusion Matrix(SMOTE) is ' + repr(c_m_smo))

    f1_smo = f1_score(l_smo, predict_smo)
    print('F1 score(SMOTE) is ' + repr(f1_smo))

    # For test data
    predict_test_smo = clf.predict(test_set_std)
    accuracy_test_smo = accuracy_score(test_label, predict_test_smo)

    print('Accuracy(Test) is ' + repr(accuracy_test_smo))
    c_m_test_smo = confusion_matrix(test_label, predict_test_smo, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test_smo))
    f1_test_smo = f1_score(test_label, predict_test_smo)
    print('F1(Test) score is ' + repr(f1_test_smo))

    pass


# This function perform cross-validation of svc
def cv_linearsvc():
    # load the data
    train_set_raw = load_data_set('washed_data_train_52.csv')
    train_label_raw = load_data_set('washed_data_label_52.csv')

    # convert string to number for train
    train_label = []
    for item in train_label_raw:
        train_label.append(float(item[0]))

    train_set = []
    for x in range(len(train_set_raw)):
        for y in range(len(train_set_raw[0])):
            train_set_raw[x][y] = float(train_set_raw[x][y])

        train_set.append(train_set_raw[x])

    train_set = np.array(train_set)

    # Feature chosen
    # [  5  11  12  17  20  22  25  26  28  31  35  43  44  45  49  51  56  57
    #   58  62  66  67  69  70  76  78  80  85  86  92  97  99 101 103 107 108
    #  112 113]
    new_train_set = []
    for item in train_set:
        temp = [item[5], item[11], item[12], item[17], item[20], item[22], item[25], item[26], item[28], item[31],
                item[35], item[43], item[44], item[45], item[49], item[51], item[56], item[57], item[58], item[62],
                item[66], item[67], item[69], item[70], item[76], item[78], item[80], item[85], item[86], item[92],
                item[97], item[99], item[101], item[103], item[107], item[108], item[112], item[113]]

        new_train_set.append(temp)

    # Normalization
    scale = StandardScaler()
    scale.fit(new_train_set)
    train_set_std = scale.transform(new_train_set)

    # Cross-validation
    c = [1, 5, 10, 100, 1000]
    for i in range(5):
        clf = LinearSVC(random_state=0, tol=1e-4, C=c[i])
        cv_result = cross_val_score(clf, train_set_std, train_label, cv=3)
        print(cv_result)

    pass


# This function perform linearsvc
def linearsvc():
    # load the data
    train_set_raw = load_data_set('washed_data_train_52.csv')
    train_label_raw = load_data_set('washed_data_label_52.csv')

    test_set_raw = load_data_set('washed_data_test52.csv')
    test_label_raw = load_data_set('washed_data_label_test52.csv')

    # convert string to number for train
    train_label = []
    for item in train_label_raw:
        train_label.append(float(item[0]))

    train_set = []
    for x in range(len(train_set_raw)):
        for y in range(len(train_set_raw[0])):
            train_set_raw[x][y] = float(train_set_raw[x][y])

        train_set.append(train_set_raw[x])

    train_set = np.array(train_set)

    print(train_set.shape)
    print(len(train_set))
    print(len(train_label))

    # for test data
    test_label = []
    for item in test_label_raw:
        test_label.append(float(item[0]))

    test_set = []
    for x in range(len(test_set_raw)):
        for y in range(len(test_set_raw[0])):
            test_set_raw[x][y] = float(test_set_raw[x][y])

        test_set.append(test_set_raw[x])

    test_set = np.array(test_set)

    print(test_set.shape)
    print(len(test_set))
    print(len(test_label))

    # Feature chosen
    # [  5  11  12  17  20  22  25  26  28  31  35  43  44  45  49  51  56  57
    #   58  62  66  67  69  70  76  78  80  85  86  92  97  99 101 103 107 108
    #  112 113]
    new_train_set = []
    for item in train_set:
        temp = [item[5], item[11], item[12], item[17], item[20], item[22], item[25], item[26], item[28], item[31],
                item[35], item[43], item[44], item[45], item[49], item[51], item[56], item[57], item[58], item[62],
                item[66], item[67], item[69], item[70], item[76], item[78], item[80], item[85], item[86], item[92],
                item[97], item[99], item[101], item[103], item[107], item[108], item[112], item[113]]

        new_train_set.append(temp)

    new_test_set = []
    for item in test_set:
        temp = [item[5], item[11], item[12], item[17], item[20], item[22], item[25], item[26], item[28], item[31],
                item[35], item[43], item[44], item[45], item[49], item[51], item[56], item[57], item[58], item[62],
                item[66], item[67], item[69], item[70], item[76], item[78], item[80], item[85], item[86], item[92],
                item[97], item[99], item[101], item[103], item[107], item[108], item[112], item[113]]

        new_test_set.append(temp)

    # Normalization
    scale = StandardScaler()
    scale.fit(new_train_set)
    train_set_std = scale.transform(new_train_set)
    test_set_std = scale.transform(new_test_set)

    clf = LinearSVC(random_state=0, tol=1e-4, C=1)
    clf.fit(train_set_std, train_label)
    predict = clf.predict(train_set_std)
    accuracy = accuracy_score(train_label, predict)

    print('Accuracy is ' + repr(accuracy))
    c_m = confusion_matrix(train_label, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(train_label, predict)
    print('F1 score is ' + repr(f1))

    # For test data
    predict_test = clf.predict(test_set_std)
    accuracy_test = accuracy_score(test_label, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(test_label, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(test_label, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    # With SMOTE
    f_smo, l_smo = smote(train_set_std, train_label)
    clf.fit(f_smo, l_smo)
    predict_smo = clf.predict(f_smo)
    accuracy_smo = accuracy_score(l_smo, predict_smo)
    print('Accuracy(SMOTE) is ' + repr(accuracy_smo))
    c_m_smo = confusion_matrix(l_smo, predict_smo, labels=[1, -1])
    print('Confusion Matrix(SMOTE) is ' + repr(c_m_smo))

    f1_smo = f1_score(l_smo, predict_smo)
    print('F1 score(SMOTE) is ' + repr(f1_smo))

    # For test data
    predict_test_smo = clf.predict(test_set_std)
    accuracy_test_smo = accuracy_score(test_label, predict_test_smo)

    print('Accuracy(Test) is ' + repr(accuracy_test_smo))
    c_m_test_smo = confusion_matrix(test_label, predict_test_smo, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test_smo))
    f1_test_smo = f1_score(test_label, predict_test_smo)
    print('F1(Test) score is ' + repr(f1_test_smo))

    pass


# This function perform naive-bayes
def naive_bayes():
    # load the data
    train_set_raw = load_data_set('washed_data_train_52.csv')
    train_label_raw = load_data_set('washed_data_label_52.csv')

    test_set_raw = load_data_set('washed_data_test52.csv')
    test_label_raw = load_data_set('washed_data_label_test52.csv')

    # convert string to number for train
    train_label = []
    for item in train_label_raw:
        train_label.append(float(item[0]))

    train_set = []
    for x in range(len(train_set_raw)):
        for y in range(len(train_set_raw[0])):
            train_set_raw[x][y] = float(train_set_raw[x][y])

        train_set.append(train_set_raw[x])

    train_set = np.array(train_set)

    print(train_set.shape)
    print(len(train_set))
    print(len(train_label))

    # for test data
    test_label = []
    for item in test_label_raw:
        test_label.append(float(item[0]))

    test_set = []
    for x in range(len(test_set_raw)):
        for y in range(len(test_set_raw[0])):
            test_set_raw[x][y] = float(test_set_raw[x][y])

        test_set.append(test_set_raw[x])

    test_set = np.array(test_set)

    print(test_set.shape)
    print(len(test_set))
    print(len(test_label))

    # Feature chosen
    # [ 14 ]
    new_train_set = []
    for item in train_set:
        temp = [item[14]]

        new_train_set.append(temp)

    new_test_set = []
    for item in test_set:
        temp = [item[14]]

        new_test_set.append(temp)

    # Normalization
    scale = StandardScaler()
    scale.fit(new_train_set)
    train_set_std = scale.transform(new_train_set)
    test_set_std = scale.transform(new_test_set)

    clf = GaussianNB()
    clf.fit(train_set_std, train_label)
    predict = clf.predict(train_set_std)
    accuracy = accuracy_score(train_label, predict)

    print('Accuracy is ' + repr(accuracy))
    c_m = confusion_matrix(train_label, predict, labels=[1, -1])
    print('Confusion Matrix is ' + repr(c_m))
    f1 = f1_score(train_label, predict)
    print('F1 score is ' + repr(f1))

    # For test data
    predict_test = clf.predict(test_set_std)
    accuracy_test = accuracy_score(test_label, predict_test)

    print('Accuracy(Test) is ' + repr(accuracy_test))
    c_m_test = confusion_matrix(test_label, predict_test, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test))
    f1_test = f1_score(test_label, predict_test)
    print('F1(Test) score is ' + repr(f1_test))

    # With SMOTE
    f_smo, l_smo = smote(train_set_std, train_label)
    clf.fit(f_smo, l_smo)
    predict_smo = clf.predict(f_smo)
    accuracy_smo = accuracy_score(l_smo, predict_smo)
    print('Accuracy(SMOTE) is ' + repr(accuracy_smo))
    c_m_smo = confusion_matrix(l_smo, predict_smo, labels=[1, -1])
    print('Confusion Matrix(SMOTE) is ' + repr(c_m_smo))

    f1_smo = f1_score(l_smo, predict_smo)
    print('F1 score(SMOTE) is ' + repr(f1_smo))

    # For test data
    predict_test_smo = clf.predict(test_set_std)
    accuracy_test_smo = accuracy_score(test_label, predict_test_smo)

    print('Accuracy(Test) is ' + repr(accuracy_test_smo))
    c_m_test_smo = confusion_matrix(test_label, predict_test_smo, labels=[1, -1])
    print('Confusion Matrix(Test) is ' + repr(c_m_test_smo))
    f1_test_smo = f1_score(test_label, predict_test_smo)
    print('F1(Test) score is ' + repr(f1_test_smo))

    pass

cv_linearsvc()