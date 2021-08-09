#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:50:30 2020

@author: zhangjr
"""

import mdtraj as md 
import numpy as np

#get distance of two domains: NMP and LID
#NMP: 30 to 59
#LID: 122 to 159

def distance_two_domains(filenames,coming_cycle):
    #load system
    ensemble=md.load_pdb('./closed_sys.pdb')
    topology=ensemble.topology
    
    #NMP domain
    dcd_NMP_CA=md.load(filenames,top='./closed_sys.pdb',atom_indices=topology.select('name CA and resid 30 to 59'))
    NMP_CA_center=md.compute_center_of_mass(dcd_NMP_CA)
    
    #LID domain
    dcd_LID_CA=md.load(filenames,top='./closed_sys.pdb',atom_indices=topology.select('name CA and resid 122 to 159'))
    LID_CA_center=md.compute_center_of_mass(dcd_LID_CA)
    
    
    domain_distance=np.sqrt(np.sum(np.square(NMP_CA_center-LID_CA_center),axis=1))
    
    np.savetxt('./'+str(coming_cycle)+'/domain_distance.txt',domain_distance)
    
