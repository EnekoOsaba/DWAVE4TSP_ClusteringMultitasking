# DWAVE4TSPLIB

This repository contains a modification of the Quantum TSP for using TSPLIB instances in DWAVE platform.

The reference code used can be found in: https://github.com/BOHRTECHNOLOGY/quantum_tsp

## Modifications

Several modification have been done over the code above mentioned:

1. The Rigetti Computing code has been erased, in order to consider only DWAVE platform.
2. Some functionalities have been added in TSP_utilities.py in order to consider the reading and solving of TSPLib instances.
3. Two reading procedures have been added to deem 2D and Geographical coordinates. The user should select the reading procedure in the code.
4. A folder has been added with the instances considered: burma14, ulysses16, ulysses22, wi28, dj39.
5. The solving procedure has been modified in order to use BQSolve partitioning solver. In this version of the code, the local Tabu Search is used for solving the problem, for properly using the DWAVE platform, please, descomment the above lines.

For properly using BQSolve, the reference library should be installed. The other libraried needed are the ones detailed in: https://github.com/BOHRTECHNOLOGY/quantum_tsp. 


