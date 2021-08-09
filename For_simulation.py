#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:37:03 2020

@author: zhangjr
"""

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

#build the system and it can be used by initial short simulation and seed simulation
def build_system(prmtop):
    system = prmtop.createSystem(nonbondedMethod = omma.PME,
                                 nonbondedCutoff = 1*unit.nanometer,
                                 constraints = omma.HBonds)
    integrator = omm.LangevinIntegrator(310*unit.kelvin, 1/unit.picoseconds,
                                        0.002*unit.picoseconds)
    platform = omm.Platform.getPlatformByName('CUDA')
    #properties={'DeviceIndex':'2'}
    simulation = omma.Simulation(prmtop.topology, system, integrator, platform)
    #simulation=omma.Simulation(prmtop.topology,system,integrator)
    
    return simulation


#run pre-eq simulation
def run_short_pre(simulation,inpcrd,steps):
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    simulation.minimizeEnergy()

    simulation.reporters.append(omma.DCDReporter('pre.dcd',5000))
    simulation.reporters.append(omma.StateDataReporter('pre_log.txt',5000,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))
    #checkpoint
    simulation.reporters.append(omma.CheckpointReporter('checkpnt.chk', 5000))


    simulation.step(steps)




#run short simulation
def run_short_eq(simulation,inpcrd,steps):
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocities(inpcrd.velocities)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    
    #simulation.minimizeEnergy()
    
    simulation.reporters.append(omma.DCDReporter('eq.dcd',5000))
    simulation.reporters.append(omma.StateDataReporter('eq_log.txt',5000,
                                                       step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))
    
    simulation.step(steps)


#run seed simulation by reading the coordinates of seed and intialize velocity based on tempoerture
def run_seed_simulation(simulation,inpcrd,steps,seed_index,coming_cycle):
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(310)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    
    simulation.reporters.append(omma.DCDReporter('./'+str(coming_cycle)+'/'+
                                                  str(seed_index)+'.dcd',5000))
    
    simulation.reporters.append(omma.StateDataReporter('./'+str(coming_cycle)+'/'
                                                       +str(seed_index)+'.txt',
                                                       5000,step=True,
                                                       potentialEnergy=True,
                                                       temperature=True))
    
    simulation.step(steps)
    
