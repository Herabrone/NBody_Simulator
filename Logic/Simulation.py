'''
Darian Lagman

This project will simulate the movements of n bodies with realtion to each other
utilizaing the Barnes-Hut algorithm to calculate the forces applied on each body.

'''

import numpy as np
from scipy.integrate import solve_ivp
import click
# Updated import for ForceCalculator
from logic.forces import ForceCalculator
# Existing imports for Octree, Body, and insert_body remain
from Logic.BarnesHut import OctreeNode, insert_body, Body
from Visualization import print_results

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


@click.command()
@click.option('--n', default= 10, help ='An integer value for the number of bodies in the simulation. Default is 20 bodies')
@click.option('--R0', default= 3.0856776e10 )
@click.option('--timespan', default= 1e6, help = 'A integer value (in seconds) to run the simulation for. Default is 1e6 seconds')
@click.option('--theta', default= 0.5, help = 'Opening angle parameter for the Barnes-Hut algorithm. Default: 0.5')
@click.option('--bound-condition', default= 10, help = 'Limit at which stars are no longer a part of the cluster. Default: 10')



def main(n, R0, timespan, theta, bound_condition):
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

if __name__ == '__main__':
     main()