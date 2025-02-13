'''
Darian Lagman
7917268
ASTR 3180 Final Project

This project will simulate the movements of n bodies with realtion to each other
utilizaing the Barnes-Hut algorithm to calculate the forces applied on each body.

'''
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import click


# Implementation imports
from BarnesHut import OctreeNode, insert_body, Body

from Visualization import print_results

# Function to calculate forces using the octree
def calculate_forces(node, body, theta):
    '''
     Calculate the gravitational forces acting on a body using the Barnes-Hut approximation.

    Parameters:
    - node (OctreeNode): The current node in the octree.
    - body (Body): The body for which forces are being calculated.
    - theta (float): Threshold parameter for the Barnes-Hut approximation (default: 0.5).
    '''
    if node.is_leaf():
        for other_body in node.bodies:
            if other_body is not body:
                apply_gravitational_force(body, other_body)
    else:
        distance = np.linalg.norm(node.center_of_mass - body.position)
        if node.size / distance < theta: # If s/d < theta
            # Treat this node as a single body
            apply_gravitational_force(body, node)
        else:
            # Recursively calculate forces from child nodes
            for child in node.children:
                if child is not None:
                    calculate_forces(child, body, theta)

def apply_gravitational_force(body, other):
    '''
    Apply gravitational force to a body from another body or octree node.

    Parameters:
    - body (Body): The body experiencing the force.
    - other (Body or OctreeNode): The source of the gravitational force.
    '''
    G = 6.67430e-11  # Gravitational constant

    if isinstance(other, OctreeNode):
        r = other.center_of_mass - body.position
        mass = other.total_mass
    else:
        r = other.position - body.position
        mass = other.mass
    distance = np.linalg.norm(r)
    force = G * body.mass * mass / distance**2
    body.acceleration += force / body.mass * r / distance

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
    
    # Calculate accelerations using the octree
    for body in bodies:
        calculate_forces(root, body, theta)
    
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



def main(n, r0, timespan, theta, bound_condition):
    '''
    Main function for running the n-body simulation
     Parameters:
     - n (int): Number of bodies in the simulation (default: 20).
     - r0 (float): Initial radius of the cluster (default: 3.0856776e10).
     - timespan (float): Duration of the simulation in seconds (default: 1e6).
     - theta (float): Opening angle parameter for the Barnes-Hut algorithm (default: 0.5).
    '''

    m = 1.9891e30  # Mass of the bodies (kg)
    

    masses, positions, velocities = initialize_bodies(n, r0, m)

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