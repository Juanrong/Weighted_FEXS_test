#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:27:05 2020

@author: zhangjr
"""
import Get_coord
import AE_architecture
import Gmm_convexhull_selection
import Coarse_CV
import For_simulation

import torch.optim as optim
import simtk.openmm.app as omma

import os
import numpy as np


gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




#read prmtop and inpcrd file of system
prmtop=omma.AmberPrmtopFile('./closed_sys.prmtop')
initial_inpcrd=omma.AmberInpcrdFile('./closed_sys.inpcrd')

#build system
simulation=For_simulation.build_system(prmtop)

#run 10 ns eq simulation, output eq.dcd
For_simulation.run_short_eq(simulation,initial_inpcrd,5000000)

number_of_short_simulations=50 
number_of_cycle=5

#repeat
for coming_cycle in range(number_of_cycle):
    
    #all dcd up to now

    filenames=['./eq.dcd']+['./'+str(cycle)+'/'+str(index+1)+'.dcd' 
                            for cycle in range(coming_cycle) 
                            for index in range(number_of_short_simulations)]
    
    os.system('mkdir '+str(coming_cycle))
    
    #get coordinate and save in coming_cycle dir, named coord_CA.txt
    dcd=Get_coord.get_coord_after_alignment(filenames,coming_cycle)
    
    #load coordinate
    data=np.loadtxt('./'+str(coming_cycle)+'/coord_CA.txt')
    
    #reduce the dimensionality by AE and save in coming_cycle dir, named z_all.txt
    vae=VAE_architecture.VAE(x_dim=642,h_dim1=321,h_dim2=160,z_dim=2).to(gpu)
    optimizer=optim.Adam(vae.parameters(),lr=0.0001)  
    VAE_architecture.train_and_get(vae,optimizer,data,30,coming_cycle)
    
    #select seed by GMM and convex hull and save, named vertix_index.txt
    z_all=np.loadtxt('./'+str(coming_cycle)+'/z_all.txt')
    Gmm_convexhull_selection.initial_seed_selection(z_all,coming_cycle)
    vertix_index=np.loadtxt('./'+str(coming_cycle)+'/vertix_index.txt',dtype=int)
    
    #get coarse CV of primary seed based on vertix_index and save, named domain_distance.txt
    Coarse_CV.distance_two_domains(filenames,coming_cycle)
    domain_distance=np.loadtxt('./'+str(coming_cycle)+'/domain_distance.txt')
    
    #assign computational resources according to the distance of two domains
    
    #1. compute the weight of each initial seed
    domain_distance_vertix=domain_distance[vertix_index]
    distance_vertix_square=domain_distance_vertix**2
    distance_vertix_square_sum=np.sum(distance_vertix_square)
    vertix_weight=[element/distance_vertix_square_sum for element in distance_vertix_square]
    
    #2. compute the number of compuational resources of each initial seed
    num=np.array([round(weight*number_of_short_simulations) for weight in vertix_weight],dtype=int)
    
    #add the left to the index with the biggest weight 
    check=np.sum(num)
    left=number_of_short_simulations-check
    max_weight_index=np.where(vertix_weight==max(vertix_weight))[0][0]
    num[max_weight_index]=num[max_weight_index]+left
    
    final_index=[]
    for i in range(len(vertix_index)):
        final_index=final_index+[vertix_index[i]]*num[i]
    
    np.savetxt('./'+str(coming_cycle)+'/final_index.txt',final_index,fmt='%d')
    
    #save the final seed
    dcd_save=dcd[final_index]
    dcd_save.save_amberrst7('./'+str(coming_cycle)+'/rst')
    
    #run multiple short simulations
    for j in range(number_of_short_simulations):
        simulation_seed=For_simulation.build_system(prmtop)
        seed_inpcrd=omma.AmberInpcrdFile('./'+str(coming_cycle)+'/rst.'+
                                         ("{0:0"+str(len(str(len(final_index))))+"d}").format(j+1))
        
        For_simulation.run_seed_simulation(simulation_seed,seed_inpcrd,50000,j+1,coming_cycle)
    
    
