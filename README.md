look at this readme

Author: Darian Lagman (7917268)
Class: ASTRO 3180
Date: December 6, 2024

Description

This program utilizes the Barnes-Hut algorithm to run an N-body simulation

Required Packages

The following packages are requried to run this simulation

- numpy
- scipy
- matplotlib
- Axes3D
- click

All of which can be installed using

pip install "Package"

Usage
The program accepts the following command-line arguments:

--n An integer value for the number of bodies in the simulation. Default is 50 bodies
--R0 Radius to Initially generate bodies. Default is 3.0856776e10m
--timespan A integer value to run the simulation for. Default is 1e6 seconds
--theta Opening angle parameter for the Barnes-Hut algorithm. Default: 0.5
--bound-condition Limit at which stars are no longer a part of the cluster, this is a distance multiplied R0. Default: 10

These parameters can be accessed by editing the RUNME

Example

python Lagman_Darian_nBody.py --n 35
