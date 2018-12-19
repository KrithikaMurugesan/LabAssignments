#Import library

from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
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

#Pre-processing virus data
def virus():
    path = "/home/kritz/Downloads/dataset/"
    label = []
    data = np.array([])
    #Listing all the files in directory
    for rootdir, subdirs, files in os.walk(path):
        if files:
            for file in sorted(files):
                x,y = load_svmlight_file(rootdir+'/'+file)
                if file in ["2011-09.txt","2013-07.txt","2013-09.txt","2014-01.txt"]:
                    update = np.zeros((x.shape[0],1))
                    x = scipy.sparse.hstack((x,np.array((update))))
                label.extend(y)
                data = scipy.sparse.vstack((data,x))
    print(len(y),data.shape)
    data = data.tocsr()
    return(data[1:,].tocsr(),label)

def Kdd():
    path = "/home/kritz/Downloads/cup98LRN.txt"
    df = pd.read_csv(path,low_memory=False)

    df = df.iloc[1:10000,:]

    cols = df.columns
    numCol = df._get_numeric_data().columns
    catCol = list(set(cols) - set(numCol))
    for each in numCol:
        df[each] = df[each].astype('float')
    for each in catCol:
         df[each] = df[each].astype('category')

    newDf = pd.get_dummies(df,drop_first = True)  
    newDf = newDf.fillna(newDf.mean())

    X, y = newDf.loc[:,newDf.columns != 'TARGET_D'], newDf.loc[:,df.columns == 'TARGET_D']
    XNew = SelectKBest(chi2, k=200).fit_transform(X, y)

    return(scipy.sparse.csr_matrix(XNew),y)

def dataDistribute(data):
    '''All the send functions in root process'''
    rows = data.shape[0]
    for i in range(1,size):
        startIndex = int((i-1)*(rows/(size-1)))
        endIndex = int(((rows/(size-1))*i))
        ##print("Index",startIndex,endIndex)
        #May cause index out of range due to index strting from 0, but we are counting from 1
        if endIndex > rows:
            endIndex = int(rows)
        comm.send(data[startIndex:endIndex,:],dest = i,tag =1)
        
def trainTestSplit(data,y):
    #3. Split the data into test and train
    #Add bias
    print("Data raw ", data.shape)
    bias = np.ones((data.shape[0],1))
    dataBias = scipy.sparse.hstack([data,np.array(bias)])
    dataBias = dataBias.tocsr()
    print("After bias ",dataBias.shape,len(y))
    #Adding target
    y = np.reshape(y,(len(y),1))
    dataXY = scipy.sparse.hstack((dataBias,y)).tocsr()
    print("After y ",dataXY.shape)
    #Generate a random list of true and false and assign train and test based on those values
    split = np.random.rand(dataXY.shape[0]) < 0.7
    #Assign train to true
    train= dataXY[split]
    #Assign test to false
    test = dataXY[~split]
    return(train,test)

def leastSquareLoss(train,beta):
    yPredicted = train[:,:-1].dot(beta)
    leastSquareLoss = (np.square(train[:,-1]-yPredicted)).sum()
    return(leastSquareLoss)

#Function for derivative
def derivative(train,beta):
    x,y = train[:,:-1],train[:,-1]
    predicted = y - (x.dot(beta))
    #print((-2 * x.T.dot(predicted)).shape)
    return (-2 * x.T.dot(predicted))

def SGD(train,beta,alpha):
    #Shuffling csr
    index = np.arange(np.shape(train)[0])
    np.random.shuffle(index)
    train = train[index,:]
    for each in train:
        #Computing next beta
        betaNext = beta - (alpha*derivative(each,beta))
        beta = betaNext  
    return(beta)

betaPart,updatedBeta,flag = None,None,False
iteration,trainRmse,testRmse,trainloss,testloss,times = 0,[],[],[],[],[]

if rank == 0:
    updatedBeta,betaPart =  None,None
    data,y = virus() 
    train,test = trainTestSplit(data[:6000,:],y[:6000])
    beta = np.zeros(((train.shape[1])-1,1))
    print("Shape of train at root ",train.shape)
    dataDistribute(train)

else:
    partFile = comm.recv(source = 0,tag=1)
    beta = None
    test = None
    print("At rank ",rank, " size of data received is ",partFile.shape)

comm.Barrier()  
    

while flag == False :  
    tic = time.time()
    beta = comm.bcast(beta,root = 0) 
    test = comm.bcast(test,root = 0) 

    comm.Barrier()

    if rank != 0:   
        betaPart = SGD(partFile,beta,0.00000000001)

    comm.Barrier()    
    updatedBeta = comm.gather(betaPart,0)
    comm.Barrier()

    if rank == 0:
        lossOld = leastSquareLoss(train,beta)
        updatedBeta = np.sum(np.array(updatedBeta)[1:,])/(size-1)
        lossNew = leastSquareLoss(train,updatedBeta)
        if lossOld - lossNew <= 0.000000000000000001:
            print("Algorithm converged")
            flag = True
        if iteration > 100:
            print("Algorithm did not converge in 100 iterations")
            flag = True
        iteration = iteration+1
        temp = leastSquareLoss(test,beta)
        testRmse.append(np.sqrt(temp/test.shape[0]))
        trainRmse.append(np.sqrt(lossOld/train.shape[0]))
        trainloss.append(lossOld)
        testloss.append(temp)
        times.append(time.time()-tic)
        beta = updatedBeta

    comm.Barrier()
    root = comm.bcast(flag,root = 0)
    comm.Barrier()
    print("Iteration ",iteration)
    if flag==True:
        loss = pd.DataFrame({'trainloss': trainloss,'testloss': testloss,'trainRmse':trainRmse,'testRmse': testRmse,'time':times})
        loss.to_csv('ResidualsVirus1.csv', sep='\t',index = False)
        comm.Abort()