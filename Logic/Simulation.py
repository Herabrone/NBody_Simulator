'''
Darian Lagman

This project will simulate the movements of n bodies with realtion to each other
utilizaing the Barnes-Hut algorithm to calculate the forces applied on each body.

'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import click
