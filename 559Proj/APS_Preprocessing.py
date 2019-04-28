import csv
import re
import numpy as np
from collections import Counter
import operator #sort dic
from pandas import read_csv
import pandas as pd



#"-1" label neg  "+1" label pos
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







def readfilesInrow(filename):
    out = []
    with open(filename) as csvfile:
        data = csv.reader(csvfile)
        data = list(data)
        #if read the untouch csv, change the read range(1,len)
        for i in range(0,len(data)):
            out.append(data[i])
    return out

def getOneCol(data,num):
    col = []
    for i in data:
        col.append(i[num])
    return col

def count_label(data,col_num):
    onecol = []

    for row in range(1,len(data)):
        onecol.append(data[row][col_num])

    print(Counter(onecol))

def howManyNA(onecol,pattern):
    count = 0
    length = len(onecol)
    for i in onecol:
        if i == pattern:
            count = count+1
    return (count/length)


#index 1 output dic, 0 output list
def writeTofile(data,filename,index):
    with open(filename,"w") as csvfile:
        csvwrite= csv.writer(csvfile)
        if index:
            for i in range(len(data)):
                csvwrite.writerow(data[i])
        else:
            for key in data.keys():
                csvfile.write("%d,%2.3f\n"%(key,data[key]))

def getAllNA(data,sort,pattern):
    Na_data={}
    for i in range(len(data[0])):
        onecol = getOneCol(data,i)
        temp = howManyNA(onecol,pattern)
        Na_data.update({i:temp})

    
    if(sort):
        temp = sorted(Na_data.items(), key = operator.itemgetter(1))
        return temp
    return Na_data


#to count how many na and 0 inside the index and output the result into csv file 
def output_howmany_NA_per():
    rawfile = "aps_failure_training_set_SMALLER.csv"
    filesort_0 = "sorted_0_collection.csv"
    filesort_NA = "sorted_na_collection.csv"

    data= readfilesInrow(rawfile)
    outputList = getAllNA(data,0,"na")
    writeTofile(outputList,filesort_NA,0)
    outputList1 = getAllNA(data,0,"0")
    writeTofile(outputList1,filesort_0,0)



def transfer_label_delete_na0_feature(filename,rawfile,filesort_NA,filesort_0):
    # filesort_0 = "sorted_0_collection.csv"
    # filesort_NA = "sorted_na_collection.csv"
    # rawfile = "aps_failure_training_set_SMALLER.csv"

    data_Na = readfilesInrow(filesort_NA)
    datapos = []
    dataneg = []


    threshold_NA = 0.3
    col_na=[]
    for i in data_Na:
        if(float(i[1])>=threshold_NA):
            col_na.append(int(i[0]))
    

    # # handle drop 0 above 70% threshold 
    data_0 = readfilesInrow(filesort_0)
    threshold_0 = 0.7
    col_0 = []
    for i in data_0:
        if(float(i[1])>=threshold_0):
            col_0.append(int(i[0]))

    col3 = np.unique(col_0+col_na)
    # print(len(col3))
    rawdata = readfile_InTwoSet(rawfile,datapos,dataneg,col3)

    writeTofile(rawdata,filename,1)

   



def fill_the_na_data_export(filename1,filename2,filename3):
    dataset = pd.read_csv(filename1, na_values=['na'], header=None).fillna(np.nan)
    label = dataset.iloc[:, 0:1]
    train = dataset.iloc[:, 1:]
    train.fillna(train.mean(), inplace=True)
    label.to_csv(filename3,index=False,float_format='%.2f',header=False)
    train.to_csv(filename2,index=False ,float_format='%.2f',header=False)

def main():
    filesort_0 = "sorted_0_collection.csv"
    filesort_NA = "sorted_na_collection.csv"
    rawfile = "aps_failure_training_set_SMALLER.csv"
    feature52 = "delete_NA_delete_52features.csv"
    # output = "delete_NA.csv"
    #read file and delete NA data output file 
    transfer_label_delete_na0_feature(feature52,rawfile,filesort_NA,filesort_0)
    # read the output nadelete file, change the na to mean and output label and the train data
    fill_the_na_data_export(feature52,'washed_data_train_52.csv','washed_data_label_52.csv')


def main_test():
    filesort_0 = "sorted_0_collection.csv"
    filesort_NA = "sorted_na_collection.csv"
    rawfile = "aps_failure_test_set.csv"
    washedfile = "delete_NA_test_52.csv"
    #read file and delete NA data output file 
    transfer_label_delete_na0_feature(washedfile,rawfile,filesort_NA,filesort_0)
    # read the output nadelete file, change the na to mean and output label and the train data
    fill_the_na_data_export(washedfile,'washed_data_test52.csv','washed_data_label_test52.csv')

   
import time


start = time. time()
main_test() 
end = time. time()
print(end - start)
