import csv
import numpy as np
import re


#if one row have question mark. drop it.
def deleteQuestionMark_other(name1,writename,writename1):
    f1=[]
    f2=[]
    with open(name1,"r") as csvfile:
        data = csv.reader(csvfile)
        newone = list(data)
        for row in range(len(newone)):
            flag = 1
            for column in range(len(newone[row])):
                k = re.sub('\s','',newone[row][column])
                newone[row][column] = k
                a = [0,1,3,6,7,8,9,12,13]
                #delet the qusetion mark in non-numerical feature
                if column in [1,3,5,6,7,8,9,13]:
                    if "?" not in newone[row][column]:
                        flag = flag* 1
                    else:
                        flag = flag* 0
            if flag:
                f1.append(newone[row])
            else:
                f2.append(newone[row])


    with open(writename,"w") as csvfile:
        csvwrite= csv.writer(csvfile)
        for i in range(len(f1)):
            csvwrite.writerow(f1[i])

    with open(writename1,"w") as csvfile:
        csvwrite= csv.writer(csvfile)
        for i in range(len(f2)):
            csvwrite.writerow(f2[i])

    print("the size of data without question mark ",len(f1)-1)
    print("the size of data with question mark ",len(f2))


#read file and save the data label
def readfile(filename1,dataL50k,dataS50k):
    with open(filename1,"r") as csvfile:
        raw = csv.reader(csvfile)
        raw = list(raw)
        row = []
        for row in raw:
            if (">50K" in row) or (">50K." in row):
                row[-1]=1
                dataL50k.append(row)
            elif ("<=50K" in row) or ("<=50K." in row):
                row[-1]=-1
                dataS50k.append(row)
    
#output the data to csv file 
def change50K_label(filenameout,dataL50k,dataS50k):
    
    print("above 50k total ",len(dataL50k))
    print("below 50k total ",len(dataS50k))
    with open(filenameout,"w") as csvfile:
        csvwrite= csv.writer(csvfile)
        for i in range(len(dataL50k)):
            csvwrite.writerow(dataL50k[i])
        for j in range(len(dataS50k)):
            csvwrite.writerow(dataS50k[j])





def main_test():

    name2 = "adult.test_SMALLER.csv"
    nameout = "adult.test_noQ.csv"
    nameout1= "adult.test_haveQ.csv"
    writename2 = "test_50k.csv"

    deleteQuestionMark_other(name2,nameout,nameout1)

    dataL50k = []
    dataS50k = []

    readfile(nameout,dataL50k,dataS50k)
    change50K_label(writename2,dataL50k,dataS50k)

# main_test()

def main_train():
    name1 = "adult.train_SMALLER.csv"
    writename = "noQuestionmark_adult.train_SMALLER.csv"
    writename1 = "hasQuestionmark_adult.train_SMALLER.csv"

    deleteQuestionMark_other(name1,writename,writename1)

    filenameout = "change50k.csv"

    dataL50k = []
    dataS50k = []

    readfile(writename,dataL50k,dataS50k)
    change50K_label(filenameout,dataL50k,dataS50k)


main_train()
#read the csv file and change the label(last column) to number 0/1

