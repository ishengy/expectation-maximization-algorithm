# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:37:42 2020

@author: ivan.sheng

Description: Self-Implemented Expectation-Maximization algorithm for a generic
            number of clusters (minimum of 3).
"""

import numpy as np
import copy 

#initialize matrix and k
x = np.array([[1,2],[4,2],[1,3],[4,3]])
k = 2

n,d = x.shape
iterations = 1

xMean = np.mean(x,axis = 0).reshape(-1,1)
xstd = np.std(x,axis=0,ddof=1).reshape(-1,1)
onesMtx = np.ones((1,k))
mLeft = np.dot(xMean,onesMtx)
    
r = np.random.uniform(-1,1,size=2).reshape(1,-1)
mRight = np.dot(xstd,r)
    
mMtx = mLeft + mRight
avgStd = np.mean(xstd)*onesMtx
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
        diff = (x-mMtx[:,i]).T
        expon = -.5*(np.sum(diff**2, axis=0)/(avgStd[0][i]**2))
        e = np.exp(expon)
        g = e/coeff
        pg[i] += (g*pk[i])
    probMtx = pg/np.sum(pg, axis = 0)
    
    #M step
    pk = (np.sum(probMtx, axis = 1)/n)[0:k]
    
    for i in range(k):
        mMtx[:,i] = np.sum(probMtx[i] * x.T,axis=1)/np.sum(probMtx[i])
        avgStd[0][i] = np.sqrt(sum(probMtx[i] * np.sum((x.T - mMtx[:,i].reshape(-1,1))**2, axis=0))/(d*sum(probMtx[i])))

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
    
    print('Iterations:', iterations)
    iterations += 1

    print('Conditional Probability:')
    print(pg.T)
    print()
    
    print('Mean:')
    print(mMtx)
    print()
    
    print('Sample Standard Deviation:')
    print(avgStd)
    print()
    
    print('Prabability:')
    print(pk)
    print()