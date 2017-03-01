import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def knn(training, test, training_labels, test_labels, k):
    tr = len(training)
    ts = len(test)
    label = np.zeros(shape = (ts), dtype= int)
    for i in range(0,ts):
        weight0=0
        weight1=0
        distance = np.zeros(shape = (tr,2))
        for j in range(0,tr):
            x = test[i]
            y = training[j]

            distance[j][0] = j
            distance[j][1] = np.linalg.norm(x-y)

        #print distance
        distance = distance[np.argsort(distance[:,1])]
        #print distance
        #print stop
    
        knearest = distance[:k,]

        for l in range(0,k):
            if int(training_labels[distance[l][0]]) == 0:
                weight0 = weight0 + 1/(distance[l][1]**2)
                #weight0 = weight0 + 1
            else:
                weight1 = weight1 + 1/(distance[l][1]**2)
                #weight1 = weight1 + 1
        #print str(weight0)+"         "+str(weight1)
        if weight0>weight1:
            label[i] = 0
        else:
            label[i] = 1

        #print label[i]

    #correct = 0
    a = 0
    b = 0
    c = 0
    d = 0
    for x in range(0,ts):
        #print str(label[x])+" "+str(test_labels[x])
        if label[x] == int(test_labels[x]):
            #correct=correct+1
            if label[x] == 0:
                d = d+1
            else:
                a = a+1
        else:
            if label[x] == 0:
                b = b+1
            else:
                c = c+1
    
    accuracy = ((a+d)/float(a+b+c+d))
    precision = a/float(a+c)
    recall = a/float(a+b)
    fmeasure = (2*a)/float((2*a)+b+c)

    print "accuracy: " + str(accuracy)
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "fmeasure: " + str(fmeasure)
    print ""
    return accuracy, precision, recall, fmeasure

def todis(strin):
    if strin=="Present":
        return 1
    else:
        return 0
 

k = 25
fold = 10

part="dem"

filename="project3_dataset1"
#filename="project3_dataset2"


if part=="demo":
    data_train = np.genfromtxt('project3_dataset3_train.txt',dtype='f',delimiter='\t')
    data_test = np.genfromtxt('project3_dataset3_test.txt',dtype='f',delimiter='\t')
    
    col_train = data_train.shape[1]
    col_test = data_test.shape[1]
    
    train_labels = data_train[:,col_train-1:col_train]
    test_labels = data_test[:,col_test-1:col_test]
    
    data_train = data_train[:,:col_train-1]
    data_test = data_test[:,:col_test-1]
    
    data_scale = preprocessing.MinMaxScaler(feature_range=(0, 100))
    data_train = data_scale.fit_transform(data_train)
    data_scale = preprocessing.MinMaxScaler(feature_range=(0, 100))
    data_test = data_scale.fit_transform(data_test)

    a,p,r,f = knn(data_train, data_test, train_labels, test_labels, k)

    

else:
    file_in=filename+str(".txt")

    lines = [line.rstrip('\n') for line in open(''+str(file_in))]


    lines
    nodes_cnt=len(lines)
    each_line=lines[0].split()
    col_each_line=len(each_line)
    #print each_line
    #print col_each_line




    list_container=[]
    for line in lines:
        each_line=line.split()
        #print each_line
        list_container.append(each_line)
    #print list_container

    print "done"
    df_data = pd.DataFrame(list_container)

    if filename=="project3_dataset2":
        df_data[4]=df_data[4].apply(lambda x: todis(x))

    data = df_data.as_matrix()

    #print data

    data = data.astype(np.float)
    np.random.shuffle(data)

    #data = np.genfromtxt('project3_dataset1.txt',dtype='f',delimiter='\t')
    #print data[:,4]
    row =   data.shape[0]
    col = data.shape[1]
    print row
    print col
    real_labels = data[:,col-1:col]
    #print real_labels
    data = data[:,:col-1]
    #print stop
    data_scale = preprocessing.MinMaxScaler(feature_range=(0, 100))
    data = data_scale.fit_transform(data)
    #print data
    segment = int(row/fold)
    print segment

    accuracy = 0
    precision = 0
    recall = 0
    fmeasure = 0

    for i in range(0,fold):
        if i==fold-1:
            training = data[:segment*i,:]
            test = data[segment*i:row,:]
            training_labels = real_labels[:segment*i,:]
            test_labels = real_labels[segment*i:row,:]

        else:
            a = data[:segment*i,:]
            b = data[segment*(i+1):row,:]
            training = np.concatenate((a, b), axis=0)
            test = data[segment*i:segment*(i+1),:]
            x = real_labels[:segment*i,:]
            y = real_labels[segment*(i+1):row,:]
            training_labels = np.concatenate((x, y), axis=0)
            test_labels = real_labels[segment*i:segment*(i+1),:]

        acc, p, r, f = knn(training, test, training_labels, test_labels, k)

        accuracy = accuracy + acc
        precision = precision + p
        recall = recall + r
        fmeasure = fmeasure + f

    accuracy = accuracy/fold
    precision = precision/fold
    recall = recall/fold
    fmeasure = fmeasure/fold

    print "final accuracy: " + str(accuracy)
    print "final precision: " + str(precision)
    print "final recall: " + str(recall)
    print "final fmeasure: " + str(fmeasure)

