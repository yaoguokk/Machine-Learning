from plotDecBoundaries import plotDecBoundaries
import numpy as np
import csv
import matplotlib.pyplot as plt


# point1 [x1 x2 x3 ...]
# point2 [y1 y2 y3 ...]
def euclidean_distance(point1, point2):
    dsquare = 0
    for i in range(len(point1)):
        dsquare = dsquare + (point1[i] - point2[i]) ** 2
    return dsquare ** 0.5


with open('wine_train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]  # list
    # print (rows)

# rows[i]
# len(list)

# calcualte mean
sum_x1 = 0
sum_y1 = 0
count1 = 0

sum_x2 = 0
sum_y2 = 0
count2 = 0

sum_x3 = 0
sum_y3 = 0
count3 = 0

'''
for row in rows:
    row
'''
for i in range(len(rows)):
    # 1 != '1'
    if rows[i][13] == '1':
        sum_x1 = sum_x1 + float(rows[i][0])
        sum_y1 = sum_y1 + float(rows[i][1])
        count1 = count1 + 1
    if rows[i][13] == '2':
        sum_x2 = sum_x2 + float(rows[i][0])
        sum_y2 = sum_y2 + float(rows[i][1])
        count2 = count2 + 1
    if rows[i][13] == '3':
        sum_x3 = sum_x3 + float(rows[i][0])
        sum_y3 = sum_y3 + float(rows[i][1])
        count3 = count3 + 1
# print(count1)
# print(count2)
# print(count3)
mean_x1 = sum_x1 / count1
mean_y1 = sum_y1 / count1
x1 = [mean_x1, mean_y1]

mean_x2 = sum_x2 / count2
mean_y2 = sum_y2 / count2
x2 = [mean_x2, mean_y2]

mean_x3 = sum_x3 / count3
mean_y3 = sum_y3 / count3
x3 = [mean_x3, mean_y3]

print(x1)
print(x2)
print(x3)

sum_x23 = 0
sum_y23 = 0
count23 = 0
sum_x13 = 0
sum_y13 = 0
count13 = 0
sum_x12 = 0
sum_y12 = 0
count12 = 0

for i in range(len(rows)):
    if rows[i][13] != '1':
        sum_x23 = sum_x23 + float(rows[i][0])
        sum_y23 = sum_y23 + float(rows[i][1])
        count23 = count23 + 1

for i in range(len(rows)):
    if rows[i][13] != '2':
        sum_x13 = sum_x13 + float(rows[i][0])
        sum_y13 = sum_y13 + float(rows[i][1])
        count13 = count13 + 1

for i in range(len(rows)):
    if rows[i][13] != '3':
        sum_x12 = sum_x12 + float(rows[i][0])
        sum_y12 = sum_y12 + float(rows[i][1])
        count12 = count12 + 1

mean_x23 = sum_x23 / count23
mean_y23 = sum_y23 / count23
x23 = [mean_x23, mean_y23]

mean_x13 = sum_x13 / count13
mean_y13 = sum_y13 / count13
x13 = [mean_x13, mean_y13]

mean_x12 = sum_x12 / count12
mean_y12 = sum_y12 / count12
x12 = [mean_x12, mean_y12]

print(x23)
print(x13)
print(x12)

# error rate
error = 0
for i in range(len(rows)):
    # ['1','2','1'] -> map object (1,2,1) -> [1,2,1]
    point = list(map(eval, rows[i][0:2]))

    dist1 = euclidean_distance(x1, point)
    dist2 = euclidean_distance(x2, point)
    dist3 = euclidean_distance(x3, point)
    dist12 = euclidean_distance(x12, point)
    dist13 = euclidean_distance(x13, point)
    dist23 = euclidean_distance(x23, point)

    result = 0
    if dist1 < dist23 and dist2 >= dist13 and dist3 >= dist12:
        result = 1

    if dist2 < dist13 and dist1 >= dist23 and dist3 >= dist12:
        result = 2
    if dist3 < dist12 and dist1 >= dist23 and dist2 >= dist13:
        result = 3

    if result != int(rows[i][13]):
        error = error + 1

accuracy = 1 - error / len(rows)
print(accuracy)

# rows = [[1,2,3],[2,3,4],[], ...] -> [[1,2],[3,4]] / [1,1,1,2,2,...]
# plot the training set
points = [list(map(eval, point[0:2])) for point in rows]
# labels = [list(map(eval, point[2])) for point in rows]

labels = [int(point[13]) for point in rows]
'''
labels=[]
for i in rows:
    print(type(i[13]))
    if i[13] != "1":
        labels.append(2)
    else:
        labels.append(1)
    print(i[13])


print(labels)
'''
sample_means = []
sample_means.append(x1);
sample_means.append(x23)
sample_means.append(x23)

# list -> numpy.array
points = np.array(points)
labels = np.array(labels)
sample_means = np.array(sample_means)

# plot
plotDecBoundaries(points, labels, sample_means)

sample_means = []
sample_means.append(x2);
sample_means.append(x13)
sample_means.append(x13)

points = np.array(points)
labels = np.array(labels)
sample_means = np.array(sample_means)

plotDecBoundaries(points, labels, sample_means)

sample_means = []
sample_means.append(x3);
sample_means.append(x12)
sample_means.append(x12)

points = np.array(points)
labels = np.array(labels)
sample_means = np.array(sample_means)

plotDecBoundaries(points, labels, sample_means)

plt.show()
# classify the test Set
with open('wine_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]  # list
rows = [list(map(eval, row)) for row in rows]
'''
for i in range(len(rows)):
    rows[i]
'''
error_test = 0
for row in rows:
    point = row[0:2]
    label = row[13]
    dist1 = euclidean_distance(point, x1)
    dist2 = euclidean_distance(point, x2)
    dist3 = euclidean_distance(point, x3)
    dist12 = euclidean_distance(x12, point)
    dist13 = euclidean_distance(x13, point)
    dist23 = euclidean_distance(x23, point)

    result = 0
    if dist1 < dist23 and dist2 >= dist13 and dist3 >= dist12:
        result = 1

    if dist2 < dist13 and dist1 >= dist23 and dist3 >= dist12:
        result = 2
    if dist3 < dist12 and dist1 >= dist23 and dist2 >= dist13:
        result = 3

    if result != int(row[13]):
        error_test = error_test + 1

# print(error_test)
accuracy_test_rate = 1 - error_test / len(rows)
print(accuracy_test_rate)

rows = np.array(rows)
points_test = rows[:, 0:2]
labels_test = rows[:, 13]
sample_means_test = sample_means

# plotDecBoundaries(points_test, labels_test, sample_means_test)
