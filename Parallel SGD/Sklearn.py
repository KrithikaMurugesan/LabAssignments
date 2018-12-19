from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from mpi4py import MPI

import pandas as pd
import numpy as np
import time
import scipy
import os
import re

#Initialize communicators
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



def virus():
    path = "/home/kritz/Downloads/dataset/"
    label = []
    data = np.array([])
    #Listing all the files in directory
    for rootdir, subdirs, files in os.walk(path):
        if files:
            for file in sorted(files):
                #Reading the data into sparse matrix from libsvm
                x,y = load_svmlight_file(rootdir+'/'+file)
                #Adding empty rows to compensate for missing columns in these four files
                if file in ["2011-09.txt","2013-07.txt","2013-09.txt","2014-01.txt"]:
                    update = np.zeros((x.shape[0],1))
                    x = scipy.sparse.hstack((x,np.array((update))))
                label.extend(y)
                #Adding all files to same csr matrix
                data = scipy.sparse.vstack((data,x))
    print(len(y),data.shape)
    data = data.tocsr()
    return(data[1:,].tocsr(),label)

#Pre-processing Kdd data
def Kdd():
    path = "/home/kritz/Downloads/cup98LRN.txt"
    df = pd.read_csv(path,low_memory=False)

    #Getting the numerical and categorical columns
    cols = df.columns
    numCol = df._get_numeric_data().columns
    catCol = list(set(cols) - set(numCol))
    #Type casting for float and string
    for each in numCol:
        df[each] = df[each].astype('float')
    for each in catCol:
         df[each] = df[each].astype('category')
            
    #Filling mean values to Nans 
    newDf = pd.get_dummies(df,drop_first = True)  
    newDf = newDf.fillna(newDf.mean())

    #Reducing the dimension of predictor space based on variable importance
    X, y = newDf.loc[:,newDf.columns != 'TARGET_D'], newDf.loc[:,df.columns == 'TARGET_D']
    XNew = SelectKBest(chi2, k=200).fit_transform(X, y)

    return(scipy.sparse.csr_matrix(XNew),y)

if rank == 0:
    tic = time.time()
    data,y = Kdd() 
    x = data[:,:6000]
    y = np.array(y[:6000]).ravel()

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.30, random_state=15)

    clf = linear_model.SGDRegressor()
    clf.fit(xTrain,yTrain)
    print(time.time()-tic)