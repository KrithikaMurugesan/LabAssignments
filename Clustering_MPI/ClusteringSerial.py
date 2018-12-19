from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from scipy.sparse import vstack,csr_matrix
from nltk.stem import PorterStemmer
from collections import Counter
from mpi4py import MPI
import pandas as pd
import numpy as np
import scipy
import time


def readData():
    '''Read the Tfid data from 20 newsgroup dataset'''
    #cats = ['alt.atheism', 'sci.space','comp.graphics']
    newsgroup_train = fetch_20newsgroups(subset = 'train',remove=('headers', 'footers', 'quotes'))#,categories=cats)
    text = pd.DataFrame([str(i) for i in newsgroup_train.data])
    ps = PorterStemmer()
    #Stemming
    text['stemmed'] = text.apply(lambda x: [ps.stem(y) for y in x])
    #Tfid
    vectorizer = TfidfVectorizer(stop_words = 'english')
    vector = vectorizer.fit_transform(text.stemmed.dropna())
    return(vector)

def initialCentroid(x,k):
    '''Setting the initial value for centroids'''
    centroidIndex = list(np.random.randint(x.shape[0],size = k))
    centroid = x[centroidIndex[0],:].todense()
    #print("first",first)
    for i in centroidIndex[1:]:
        temp = np.array(x[i,:].todense())
        #print("temp",temp)
        centroid = np.concatenate((centroid,temp))
    return(centroid)

def distance(a,b):
    '''Calculate the euclidean distance and return the closest cluster center'''
    dist = np.sqrt(a.dot(b.T))
    cluster = np.argmin(dist, axis=1)
    return(np.array(cluster))

def concatenate(partFile,dist,k):
    '''Group the documents based on the clustering done for the first time'''
    newSum = np.array([])
    clusters = np.unique(dist)
    partFile.tocsr()
    for each in range(0,k):
        if each in clusters:
            points = scipy.sparse.vstack([partFile[j] for j in range(partFile.shape[0]) if dist[j] == each])
        else:
            points = partFile[each,:]
        newSum = scipy.sparse.vstack((newSum,np.array(points.sum(axis = 0))))
    newSum = (newSum.tocsr())[1:,]
    return(newSum)

def globalMean(partFile,dist,k):
    '''Each worker returns the local sum and number of elements in each cluster, merging and computing global mean'''
    newSum = np.array([])
    actualCluster = np.unique(dist)
    partFile.tocsr()
    #print("Actual ",actualCluster)
    #print("Update centroid ", partFile.shape)
    for each in range(0,k):
        if each in actualCluster:
            for j in range(partFile.shape[0]):
                if dist[j] == each:
                    points = scipy.sparse.vstack(partFile[j])
            dnr = dist.count(each)
        else:
            dnr =1
            points = partFile[each]
        newSum = scipy.sparse.vstack((newSum,np.array(points.sum(axis = 0))))/dnr
            
    newSum = (newSum.tocsr())[1:,]
    return(newSum)

def kMeans(data,centroid,k):
    '''The function called recursively to compute KNN'''
    comm.Barrier()
    if rank == 0:
        '''Root process'''
        for i in range(1,size):
            '''All the send functions in root process'''
            rows = data.shape[0]
            startIndex = int((i-1)*(rows/(size-1)))
            endIndex = int(((rows/(size-1))*i))
            ##print("Index",startIndex,endIndex)
            #May cause index out of range due to index strting from 0, but we are counting from 1
            if endIndex > rows:
                endIndex = int(rows)
            comm.send(data[startIndex:endIndex,:],dest = i,tag =1)
            comm.send(centroid, dest = i,tag =2)
            ##print("BRACE FOR IMPACT")
            '''All the receive functions in root process'''
            updatedCentroid,updatedCluster,meanCluster = np.array([]),[],[]
            #print("data received from ",i)
            updatedCentroid = scipy.sparse.vstack((updatedCentroid,comm.recv(source = i, tag = 3)))
            updatedCluster.extend(comm.recv(source = i, tag = 4))
            meanCluster.extend(comm.recv(source =i, tag =5))
            updatedCentroid = (updatedCentroid.tocsr())[1:,]
            finalSum = globalMean(updatedCentroid,updatedCluster,k)
            #print("At source centroid ",updatedCentroid.shape," partFile ", finalSum.shape, "recived after update")

    else :
        '''Workers'''
        updatedCluster,updatedCentroid = None,None
        partFile = comm.recv(source = 0, tag =1)
        centroid = comm.recv(source = 0, tag =2)
        #print("At rank ", rank, "shape is ",partFile.shape)
        #print("At rank ", rank, "centroid ", centroid.shape[0])
        dist = distance(partFile,centroid)
        #print("At rank ", rank, "dist ", dist.shape)
        clusters = np.unique(dist)
        #print("Clusters ",np.unique(dist))
        newSum = concatenate(partFile,dist,k)
        comm.send(newSum, dest = 0, tag =3)
        comm.send(dist,dest = 0, tag = 4)
        comm.send(clusters, dest = 0, tag = 5)
        print("Data send from worker ", rank)
    
    comm.Barrier()
    #Broadcasting the values to be returned or else they end up null in main function :(
    updatedCluster = comm.bcast(updatedCluster, root = 0)
    updatedCentroid = comm.bcast(updatedCentroid, root = 0)
    comm.Barrier()
    return(np.concatenate(updatedCluster,axis = 0),updatedCentroid)
    comm.Barrier()
    

#Setting the communicators
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main():    
    data = readData()
    flag = True
    cnt = 1
    k = 20
    #Random array to check compare the clusters after every iteration
    default = np.array([k+1]*data.shape[0])
    comm.Barrier()
    #Getting initial centroids
    centroid = initialCentroid(data,k)
    #Start of parallel processing
    comm.Barrier()
    tic = time.time()
    #Calling Kmeans function for first time to get intial clusters
    updatedCluster,updatedCentroid = kMeans(data,centroid,k)
    print("Centroid at first ",np.unique(updatedCluster))
    comm.Barrier()   
    #Looping until membership remains the same
    while flag == True:
        print("Recomputing Centroids, iteration ",cnt,(updatedCluster == default).all())
        #Checking for membership
        if cnt == 500:
            print("Algorithm did not converge in 1000 iterations")
            print("Time taken : ", time.time()-tic)
            return()
            break
        if (updatedCluster==default).all():
            print("Algorithm Converged")
            flag = False
        else:
            comm.Barrier()
            #When condition fails recursively calling the kMeans function
            updatedClusterNew,updatedCentroidNew = kMeans(data,updatedCentroid,k)
            comm.Barrier()
            print("Time taken : ", time.time()-tic)
            cnt = cnt+1
            print("Centroid shape in loop ", cnt ,np.unique(updatedCluster))
        #Updating cluster memberships and centroids
        default = updatedCluster
        updatedCluster = updatedClusterNew
        updatedCentroid = updatedCentroidNew

        
if __name__ == "__main__":
    
    '''Main function'''
    
    main()