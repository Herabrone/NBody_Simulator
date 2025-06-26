'''
Darian Lagman

This project will simulate the movements of n bodies with realtion to each other
utilizaing the Barnes-Hut algorithm to calculate the forces applied on each body.

'''

import numpy as np
import sys
import time
from scipy.integrate import solve_ivp
# Updated import for ForceCalculator
from Logic.forces import ForceCalculator
# Existing imports for Octree, Body, and insert_body remain
from Logic.BarnesHut import OctreeNode, insert_body, Body
from UI.Visualization import print_results

#Imports for Tree Flattener
from Logic.TreeFlattener import flatten_tree
from Logic.TreeFlattener import flattened_nodes_to_numpy


dll_path = r"C:\Users\darja\Documents\ASTRO 3180\Project REPO\NBody_Simulator\build\Logic\GPU\Release"
sys.path.append(dll_path)
# The pybind11 module with both force and verlet kernels, used for GPU acceleration
import force_gpu

def equations_of_motion(t, y, masses, theta):
    '''
    Computes the equations of motion for integrater

    Parameters:
    - t (float): Current time of the simulation.
    - y (numpy array): Flattened array containing positions and velocities of all bodies.
    - masses (numpy array): Array of masses for all bodies.
    - theta (float): Opening angle parameter for the Barnes-Hut algorithm (default: 0.5).

    Returns:
    -  dydt (numpy array): Flattened array containing velocities and accelerations of all bodies
    '''
    n = len(masses)
    positions = y[:3*n].reshape((n, 3))
    velocities = y[3*n:].reshape((n, 3))

    # Create body objects with masses, positions, and velocities
    bodies = [Body(masses[i], positions[i], velocities[i]) for i in range(n)]
    
    # Create the root of the octree
    root_center = np.mean(positions, axis=0)
    root_size = np.max(np.linalg.norm(positions - root_center, axis=1)) * 2
    root = OctreeNode(root_center, root_size)
    
    # Insert bodies into the octree
    for body in bodies:
        insert_body(root, body)
    
    # Create an instance of ForceCalculator
    force_calculator = ForceCalculator()
    
    # Calculate accelerations using the octree and the new force calculator
    for body in bodies:
        force_calculator.compute_forces(root, body, theta)
    
    # Extract accelerations from the body objects
    accelerations = np.array([body.acceleration for body in bodies])
    
    # Concatenate velocities and accelerations into a single array
    dydt = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return dydt

# Function to initialize bodies
def initialize_bodies(n, R0, m):
    '''
    Initialize the masses, positions, and velocities of n bodies
    
    Parameters:
    - n (int): Number of bodies.
    - R0 (float): Initial radius of the cluster.
    - m (float): Mass of each body.
    Returns:
     - masses (numpy array): Array of masses.
     - positions (numpy array): Array of positions for all bodies.
     - velocities (numpy array): Array of velocities for all bodies.
    '''

    masses = np.full(n, m)
    positions = np.random.randn(n, 3)
    positions /= np.linalg.norm(positions, axis=1)[:, np.newaxis]
    positions *= R0 * np.random.rand(n)[:, np.newaxis] ** (1/3)

    G = 6.67430e-11

    velocities = np.random.randn(n, 3)  # Random 3D velocities
    velocities *= np.sqrt(G * m / R0)  # Scale velocities to match orbital velocity

    KE = 0.5 * np.sum(m * np.linalg.norm(velocities, axis=1)**2)
    #print(f"Initial total KE {KE}")

    return masses, positions, velocities

def run_simulation(n, R0, timespan, theta, bound_condition):


    '''
    Main function for running the n-body simulation
     Parameters:
     - n (int): Number of bodies in the simulation (default: 20).
     - r0 (float): Initial radius of the cluster (default: 3.0856776e10).
     - timespan (float): Duration of the simulation in seconds (default: 1e6).
     - theta (float): Opening angle parameter for the Barnes-Hut algorithm (default: 0.5).
    '''

    m = 1.9891e30  # Mass of the bodies (kg)
    

    masses, positions, velocities = initialize_bodies(n, R0, m)

    # Initial state vector
    y0 = np.concatenate([positions.flatten(), velocities.flatten()])

    # Adjusting timesteps with number of bodies for better resolution
    if n <= 40:
        time_step = 1
    elif n > 40 and n <=100:
        time_step = 0.1
    elif n >100 and n <200:
        time_step = 0.001
    else:
        time_step = 0.0001

    # Time span for the integration
    t_span = (0, timespan)  # Integrate from t=0 to timespan seconds
    t_eval = np.linspace(*t_span, 3000, time_step)

    solution = solve_ivp(equations_of_motion, t_span, y0, t_eval=t_eval, args=(masses, theta), method='RK45')

    print_results(n, solution, m, bound_condition)


def run_simulation_gpu(n, R0, timespan, theta, dt, bound_condition):
    m = 1.9891e30

    # 1) initialize masses, positions, velocities

    masses, pos, vel = initialize_bodies(n, R0, m)

    # cast to float32
    masses = masses.astype(np.float32)
    pos    = pos.astype(np.float32)
    vel    = vel.astype(np.float32)
    acc    = np.zeros_like(pos, dtype=np.float32)


    # 2) Build and flatten the tree once
    bodies = [Body(masses[i], pos[i], vel[i]) for i in range(n)]
    root_center = np.mean(pos, axis=0)
    root_size   = np.max(np.linalg.norm(pos - root_center, axis=1))*2
    root = OctreeNode(root_center, root_size)
    for b in bodies: 
        insert_body(root, b)

    flat = flatten_tree(root)
    tree_arrays = flattened_nodes_to_numpy(flat)

    # 3) Copy static tree → GPU
    force_gpu.upload_tree(
        tree_arrays["center_of_mass"],
        tree_arrays["total_mass"],
        tree_arrays["size"],
        tree_arrays["center"],
        tree_arrays["is_leaf"],
        tree_arrays["children"],
    )

    steps = int(timespan / dt)
    t0 = time.time() # for tracking how long the simulation takes

    positions_record  = []  #These will record the position and velocity of the objects throughpout the simulation
    velocities_record = []

    for step in range(steps):
        # A) Position update
        force_gpu.verlet_step_cuda(pos, vel, acc, np.float32(dt))

        # B) Compute new accelerations at updated pos
        acc_new = force_gpu.compute_gpu_acceleration(
            pos, masses,
            tree_arrays["center_of_mass"],
            tree_arrays["total_mass"],
            tree_arrays["size"],
            tree_arrays["center"],
            tree_arrays["is_leaf"],
            tree_arrays["children"],
            np.float32(theta),
            np.float32(1e-9),   # softening
            np.float32(6.67430e-11),
        )

        # C) Velocity update
        force_gpu.verlet_velocity_update_cuda(vel, acc, acc_new, np.float32(dt))

        # D) Swap old/new accelerations
        acc[:] = acc_new

        #Record the positions and velocities
        positions_record.append(pos.copy())
        velocities_record.append(vel.copy())

    print(f"Finished {steps} steps in {time.time()-t0:.2f}s")

    positions = np.array(positions_record)  # shape: (steps, n, 3)
    velocities = np.array(velocities_record)  # shape: (steps, n, 3)
    times = np.linspace(0, timespan, steps)

    # Flatten positions and velocities for final format (like solve_ivp)
    y = np.concatenate([positions.transpose(1, 0, 2).reshape(n, -1),
                        velocities.transpose(1, 0, 2).reshape(n, -1)], axis=1)
    y = y.reshape(-1)

    # Create a fake solution object similar to solve_ivp output
    class Solution:
        def __init__(self, t, y):
            self.t = t
            self.y = y

    solution = Solution(t=times, y=y)

    print_results(n, solution, m, bound_condition)