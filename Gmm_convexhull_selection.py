#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:06:41 2020

@author: zhangjr
"""

from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull

import numpy as np 

def initial_seed_selection(z,coming_cycle):
    #perform GMM
    probability_cutoff=0.1
    BIC_cutoff=0.3
    
    n_components=np.arange(1,21)
    models=[GaussianMixture(n,covariance_type='full',random_state=0).fit(z) for n in n_components]
    bic=[m.bic(z) for m in models]
    
    #print(bic)
    
    #determine the number of components of GMM using BIC_cutoff 0.3
    slope=(bic-min(bic))/(max(bic)-min(bic))<BIC_cutoff
    model_index=np.where(slope==True)[0][0]
    components=model_index+1
    
    print(components)
    
    #seperate points using probability 0.1 and then us convex hull at each set 
    gmm2=models[model_index]
    prob=gmm2.fit(z).predict_proba(z).round(3)
    
    
    #index of each components, index of vertices, index of not vertices
    index=[]
    hull_index=[]
    index_not_hull=[]
    for i in range(components):
        index.append(np.argwhere((prob[:,i]>probability_cutoff)==True)[:,0])
        hull=ConvexHull(z[index[i]])
        hull_index_vertices=index[i][hull.vertices]
        hull_index.append(hull_index_vertices)
        index_not_hull.append(set(index[i]).difference(set(hull_index[i])))
        
    #get the unique index of seeds after eliminating the interaction points
    vertix_index=[]
    for i in range(components):
        #hull=ConvexHull(z[index[i]])
        #hull_index_vertices=index[i][hull.vertices]
        
        hull_index_vertices=hull_index[i]
        
        for j in hull_index_vertices:
            for k in range(components):
                mark=True 
                
                if i==k:
                    continue
                else:
                    if j in index_not_hull[k]:
                        mark=False 
                        break
            
            if mark==True:
                vertix_index.append(j)
    
    vertix_index=np.unique(vertix_index)
    np.savetxt('./'+str(coming_cycle)+'/vertix_index.txt',vertix_index,fmt='%d')
    
    #return vertix_index
    
    
    
