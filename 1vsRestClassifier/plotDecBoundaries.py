################################################
## EE559 HW Wk2, Prof. Jenkins, Spring 2018
## Created by Arindam Jati, TA
## Tested in Python 3.6.3, OSX El Captain
################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plotDecBoundaries(training, label_train, sample_mean, lable=0):

    #Plot the decision boundaries and data points for minimum distance to
    #class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # sample_mean: mean vector for each class
    #
    # Total number of classes
    nclass =  len(np.unique(label_train))

    # Set the feature range for plotting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    #hstack  connect two arr into one from one tail to another head
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    # distance measure evaluations for each (x,y) pair. （xy-1922801x2; sample_mean-3x2)
    dist_mat = cdist(xy, sample_mean)#[1922801*3] one xy dist with 3 meanpointax
    pred_label = np.argmin(dist_mat, axis=1)# axis=1means output vertical direction the miniman number's index, [1922801*1]

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size,order='F')#reshape in terms of the diagram
    #order=’C’，是行优先读取（默认）
    #order=’F’，是列优先读取
    #order=’A’，是按照输入的array自动进行选择
    #reshape(size has to be tuple (2,3) [2,3]

    #show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap,  extent=[xrange[0], xrange[1], yrange[0], yrange[1]] , origin='lower')

    # plot the class training data.
    if lable !=0:
        plt.plot(training[label_train == lable, 0], training[label_train == lable, 1], 'rx')  # iterate with label_train
        plt.plot(training[label_train == 0, 0], training[label_train == 0, 1], 'go')
    else:
        plt.plot(training[label_train == 1, 0],training[label_train == 1, 1], 'rx')#iterate with label_train
        plt.plot(training[label_train == 2, 0],training[label_train == 2, 1], 'go')
        if nclass == 3:
            plt.plot(training[label_train == 3, 0],training[label_train == 3, 1], 'b*')

    # include legend for training data
    if nclass == 3:
        l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    else:
        l = plt.legend(('Class '+str(lable), 'Class other'), loc=2)
    plt.gca().add_artist(l)

    # plot the class mean vector.
    m1, = plt.plot(sample_mean[0,0], sample_mean[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(sample_mean[1,0], sample_mean[1,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    if nclass == 3:
        m3, = plt.plot(sample_mean[2,0], sample_mean[2,1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

    # include legend for class mean vector
    if nclass == 3:
        l1 = plt.legend([m1,m2,m3],['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
    else:
        l1 = plt.legend([m1,m2], ['Class '+str(lable), 'Class other'], loc=4)

    plt.gca().add_artist(l1)
    plt.show()


