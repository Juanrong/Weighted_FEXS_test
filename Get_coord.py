#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:13:33 2020

@author: zhangjr
"""

import mdtraj as md
import numpy as np

def get_coord_after_alignment(filenames,coming_cycle):
    
    #list to save coordinate
    coord=[]
    
    #load system
    ensemble=md.load_pdb('./closed_sys.pdb')
    topology=ensemble.topology
    
    #load reference
    ref=md.load_pdb('./closed_sys.pdb',atom_indices=topology.select('name CA'))
    
    #load all trajectories
    dcd_CA=md.load(filenames,top='./closed_sys.pdb',atom_indices=topology.select('name CA'))
    dcd=md.load(filenames,top='./closed_sys.pdb')
    
    #superpose and get xyz
    dcd_CA=dcd_CA.superpose(reference=ref)
    coord.extend(dcd_CA.xyz)
    coord=np.array(coord)
    
    #reshape
    d_1,d_2,d_3=coord.shape
    coord=coord.reshape(d_1,d_2*d_3)
    
    np.savetxt('./'+str(coming_cycle)+'/coord_CA.txt',coord)
    
    return dcd 