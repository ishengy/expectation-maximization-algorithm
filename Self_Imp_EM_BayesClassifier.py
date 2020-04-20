# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:37:42 2020

@author: ivan.sheng

Description: Self-Implemented Expectation-Maximization algorithm and Bayes Classifier
                performed on the iris dataset,
"""

import numpy as np
import copy 
from sklearn import datasets 
import pandas as pd

# load the iris dataset 
iris = datasets.load_iris().data 
df_iris = pd.DataFrame(iris)

"""
Problem #2
"""

k = 3 #subpops
n,d = iris.shape
iterations = 1

irisMean = np.mean(iris,axis = 0).reshape(-1,1)
irisStd = np.std(iris,axis=0,ddof=1).reshape(-1,1)
onesMtx = np.ones((1,k))
mLeft = np.dot(irisMean,onesMtx)
    
r = np.random.uniform(-1,1,size=3).reshape(1,-1)
mRight = np.dot(irisStd,r)
    
mMtx = mLeft + mRight
avgStd = np.mean(irisStd)*onesMtx
threshold = avgStd[0][0]*1.0e-6
pk = (onesMtx/k)[0]

while True:
    pPrev = copy.copy(pk)
    mPrev = copy.copy(mMtx)
    stdPrev = copy.copy(avgStd)
    pg = np.zeros([d,n])
    
    #E step
    for i in range(k):
        coeff = (np.sqrt(2*np.pi)*avgStd[0][i])**d
        diff = (iris-mMtx[:,i]).T
        expon = -.5*(np.sum(diff**2, axis=0)/(avgStd[0][i]**2))
        e = np.exp(expon)
        g = e/coeff
        pg[i] += (g*pk[i])
    probMtx = pg/np.sum(pg, axis = 0)
    
    #M step
    pk = (np.sum(probMtx, axis = 1)/n)[0:k]
    
    for i in range(k):
        mMtx[:,i] = np.sum(probMtx[i] * iris.T,axis=1)/np.sum(probMtx[i])
        avgStd[0][i] = np.sqrt(sum(probMtx[i] * np.sum((iris.T - mMtx[:,i].reshape(-1,1))**2, axis=0))/(d*sum(probMtx[i])))

    meanDelta = max(np.sqrt(sum((mMtx - mPrev)**2)))
    sMean = np.mean(np.sqrt(sum(mMtx**2)))
    convMean = (meanDelta <= sMean*threshold)
    
    sigmaDelta = max(np.sqrt(np.sum((avgStd - stdPrev)**2,axis=1)))
    sSigma = np.mean(np.sqrt(np.sum(avgStd**2,axis=1)))
    convSigma = (sigmaDelta <= sSigma*threshold)
    
    pDelta = np.sqrt(sum((pk - pPrev)**2))
    sProbabillity = np.mean(np.sqrt(sum(pk**2)))
    convProbability = (pDelta <= sProbabillity*threshold)

    if convMean & convSigma & convProbability:
        break
    
    iterations += 1

print('Mean:')
print(mMtx)
print()

print('Sample Standard Deviation:')
print(avgStd)
print()

print('Prabability:')
print(pk)
print()

"""
Problem #3
"""
    
classification = np.argmax(pg.T, axis = 1)
irisClass = np.column_stack((iris,classification))
df_irisClass = pd.DataFrame(irisClass)

sentosa = df_irisClass[df_irisClass[4]==0].iloc[:,0:4]
versi = df_irisClass[df_irisClass[4]==1].iloc[:,0:4]
virg = df_irisClass[df_irisClass[4]==2].iloc[:,0:4]

meanSentosa = sentosa.mean(axis=0).values.reshape(1,-1)
meanVersi = versi.mean(axis=0).values.reshape(1,-1)
meanVirg = virg.mean(axis=0).values.reshape(1,-1)
mean = [meanSentosa, meanVersi, meanVirg]

covSentosa = np.cov(sentosa.T)
covVersi = np.cov(versi.T)
covVirg = np.cov(virg.T)
cov = [covSentosa, covVersi, covVirg]

invSentosa = np.linalg.inv(covSentosa)
invVersi = np.linalg.inv(covVersi)
invVirg = np.linalg.inv(covVirg)
invCov = [invSentosa, invVersi, invVirg]

detSentosa = np.linalg.det(covSentosa)
detVersi = np.linalg.det(covVersi)
detVirg = np.linalg.det(covVirg)
det = [detSentosa, detVersi, detVirg]

prob_sum = 0
probC = pd.DataFrame()
for c in range(3):
    coeff = 1/(np.sqrt(((2*np.pi)**d)*det[c]))
    diff = iris-mean[c]
    mah_mtx = (-np.dot(np.dot(diff, invCov[c]),diff.T)/2).diagonal()
    e = np.exp(mah_mtx)
    probC[c] = coeff*e*pk[c]
    prob_sum += probC[c]

prob_sum = np.array([prob_sum,]*3).T
bayesClassifier = probC/prob_sum
find_max = bayesClassifier.idxmax(axis=1)
class_prob = find_max.value_counts()/150

print('Problem 3')
print('Iterations:', iterations)
print('Mean:')
print(mMtx)
print()

print('Sample Standard Deviation:')
print(avgStd)
print()

print('Prabability (Bayes):')
print(class_prob.sort_index())
print()

print('True Sentosa Stats:')
print(sentosa.describe().T)
print()

print('True Versicolor Stats:')
print(versi.describe().T)
print()

print('True Virginica Stats:')
print(virg.describe().T)
print()