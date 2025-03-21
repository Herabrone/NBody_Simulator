import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def print_results(n, solution, m, cluster_size):
    '''
    Print the results of the N-body simulation, including the 3d graph, Energies, 
    and radial density.
    
    Parameters:
    - n (int): Number of bodies in the simulation.
    - solution (np.array): Solution array containing the positions and velocities of all bodies at the final time step.
    - m: The mass of the objects
    - cluster_size (int): The size (radius) of the cluster
    '''
    # Extract the results and plot trajectories in 3D
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    for i in range(n):
        x = solution.y[3*i]
        y = solution.y[3*i+1]
        z = solution.y[3*i+2]
        ax1.plot(x, y, z)

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_zlabel('Distance (m)')
    ax1.set_title(f'Trajectories of {n} Bodies')

    # Calculate radial density profiles
    positions = solution.y[:3*n].reshape((n, 3, -1))
    final_positions = positions[:, :, -1]

    radii = np.linalg.norm(final_positions, axis=1)
    max_radius = np.max(radii)
    num_shells = 20
    shell_edges = np.linspace(0, max_radius, num_shells + 1)
    shell_volumes = (4/3) * np.pi * (shell_edges[1:]**3 - shell_edges[:-1]**3)

    shell_masses = np.zeros(num_shells)

    for i, (r_min, r_max) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        in_shell = (radii >= r_min) & (radii < r_max)
        shell_masses[i] = np.sum(in_shell) * m

    densities = shell_masses[1:] / shell_volumes[1:]  # Exclude the smallest shell

    # Plot radial density profile
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    shell_centers = (shell_edges[1:-1] + shell_edges[2:]) / 2  # Exclude the smallest shell
    ax2.plot(shell_centers, densities, marker='o')
    ax2.set_xlabel('Radius (m)')
    ax2.set_ylabel('Density (Star/m^3)')
    ax2.set_title('Radial Density Profile')

    # Calculate KE and U
    R = cluster_size * 3.0856776e10
    G = 6.67430e-11

    KE = compute_KE(solution, n, m, R)
    U = compute_U(solution, n, m, G, R)

    num_stars_within_R = calculate_num_stars_within_R(solution, n, R, m)

    # Plot KE, U, and # of stars in cluster
    fig, axs = plt.subplots(nrows=3, figsize=(8, 12))  # Create three subplots vertically
    # Plot KE on the first subplot
    axs[0].plot(solution.t, KE, label='Kinetic Energy (KE)', color='blue')
    axs[0].set_xlabel('Time (s)', labelpad=-10)  
    axs[0].set_ylabel('Kinetic Energy (J)')
    axs[0].set_title(f'Total Kinetic Energy Within Cluster Over Time')
    axs[0].legend()

    # Plot U on the second subplot
    axs[1].plot(solution.t, U, label='Potential Energy (U)', color='red')
    axs[1].set_xlabel('Time (s)', labelpad=-10)  
    axs[1].set_ylabel('Potential Energy (J)')
    axs[1].set_title('Total Potential Energy Within Cluster Over Time')
    axs[1].legend()

    
    # Plot # of stars in cluster
    axs[2].plot(solution.t, num_stars_within_R, label='# of stars within cluster', color='green')
    axs[2].set_xlabel('Time (s)', labelpad=-10)
    axs[2].set_ylabel('Number of Stars')
    axs[2].set_title('Number of Stars Within Cluster')
    axs[2].legend()
    

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()


def compute_KE(solution, n, m, R):
    '''
    Compute the total kinetic energy (KE) of the system within some radius.
    
    Parameters:
    - solution (np.array): An array containing the positions and velocities of all bodies in the system.
      It is assumed that the first 3 columns correspond to the positions (x, y, z), and the next 3 columns
      correspond to the velocities (vx, vy, vz).
    - n (int): The number of bodies in the simulation.
    - m (int): The mass of the objects
    - R (float): The radius from the COM to consider stars to.
    
    Returns:
    - KE (float): The total kinetic energy of the system in Joules.
    '''
    # Calculate the center of mass at the first time step
    com = get_centre_mass(solution, m)

    # Initialize kinetic energy array
    KE = np.zeros(len(solution.t))  # Array to store KE for each time step

    # Loop over each time step
    for t_idx in range(len(solution.t)):
        KE_t = 0.0  # Initialize total kinetic energy for this time step

        # Loop over all bodies to calculate the kinetic energy for those within radius R from COM
        for i in range(n):
            xi = solution.y[3 * i, t_idx]     # x position of body i at time t_idx
            yi = solution.y[3 * i + 1, t_idx] # y position of body i at time t_idx
            zi = solution.y[3 * i + 2, t_idx] # z position of body i at time t_idx

            # Calculate the distance from body i to the center of mass (COM)
            dist = np.sqrt((xi - com[0])**2 + (yi - com[1])**2 + (zi - com[2])**2)

            # If the body is within radius R from the COM, calculate its kinetic energy
            if dist < R:
                vi = solution.y[3 * n + 3 * i, t_idx]     # x velocity of body i
                vi_y = solution.y[3 * n + 3 * i + 1, t_idx] # y velocity of body i
                vi_z = solution.y[3 * n + 3 * i + 2, t_idx] # z velocity of body i

                v2 = vi**2 + vi_y**2 + vi_z**2  # Velocity squared

                KE_t += 0.5 * m * v2  # Add to total kinetic energy

        KE[t_idx] = KE_t  # Store the total kinetic energy for this time step

    return KE  # Return the KE array, which has the same length as solution.t

def compute_U(solution, n, m, G, R, epsilon=1e2):
    '''
     Compute the total Gravitational Potential Energy (U) of the system within some radius.
    
    Parameters:
    - solution (np.array): An array containing the positions and velocities of all bodies in the system.
      It is assumed that the first 3 columns correspond to the positions (x, y, z), and the next 3 columns
      correspond to the velocities (vx, vy, vz).
    - n (int): The number of bodies in the simulation.
    - m (int): The mass of the objects
    - G (const): Gravitational constant
    - R (float): The radius from the COM to consider stars to.
    - epsilon (int): Softening radius (min sitance that two bodies can get within each other)
    
    Returns:
    - U (float): The total kinetic energy of the system in Joules.
    
    '''
    # Calculate the center of mass once at the first time step
    com = get_centre_mass(solution, m)

    # Initialize potential energy array
    U = np.zeros(len(solution.t))

    # Loop over each time step
    for t_idx in range(len(solution.t)):
        # List of bodies within radius R at time step t_idx
        bodies_within_R = []

        # Loop over all bodies to check if they are within radius R from the COM
        for i in range(n):
            xi = solution.y[3*i, t_idx]   # x position of body i at time t_idx
            yi = solution.y[3*i+1, t_idx] # y position of body i at time t_idx
            zi = solution.y[3*i+2, t_idx] # z position of body i at time t_idx

            # Calculate the distance from body i to the center of mass (COM)
            dist = np.sqrt((xi - com[0])**2 + (yi - com[1])**2 + (zi - com[2])**2)

            # If within radius R, add the body to the list
            if dist < R:
                bodies_within_R.append(i)
              


        # Now calculate the potential energy between all pairs of bodies within radius R
        for i in range(len(bodies_within_R)):
            for j in range(i + 1, len(bodies_within_R)):
                body_i = bodies_within_R[i]
                body_j = bodies_within_R[j]

                # Get positions of body i and body j at time step t_idx
                xi = solution.y[3*body_i, t_idx]
                yi = solution.y[3*body_i+1, t_idx]
                zi = solution.y[3*body_i+2, t_idx]

                xj = solution.y[3*body_j, t_idx]
                yj = solution.y[3*body_j+1, t_idx]
                zj = solution.y[3*body_j+2, t_idx]

                # Calculate the softened distance between body i and body j
                dist_ij = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2) + epsilon

                

                # Add potential energy for the pair (i, j)
                U[t_idx] -= G * m * m / dist_ij  # Add potential energy for the pair

    return U  # Return the total potential energy



def calculate_num_stars_within_R(solution, n, R, m):
    ''' 
    Compute the number of stars within some radius.
    
    Parameters:
    - solution (np.array): An array containing the positions and velocities of all bodies in the system.
      It is assumed that the first 3 columns correspond to the positions (x, y, z), and the next 3 columns
      correspond to the velocities (vx, vy, vz).
    - n (int): The number of bodies in the simulation.
    - m (int): The mass of the objects
    - R (float): The radius from the COM to consider stars to.
    
    Returns:
    - num_stars_within_R (int)
    '''
    
    # Calculate the center of mass at the first time step
    com = get_centre_mass(solution, m)

    # Initialize array to store the number of stars within radius R for each time step
    num_stars_within_R = np.zeros(len(solution.t))

    # Loop over each time step
    for t_idx in range(len(solution.t)):
        count = 0  # Initialize count of stars within radius R for this time step

        # Loop over all bodies to check if they are within radius R from the COM
        for i in range(n):
            xi = solution.y[3*i, t_idx]   # x position of body i at time t_idx
            yi = solution.y[3*i+1, t_idx] # y position of body i at time t_idx
            zi = solution.y[3*i+2, t_idx] # z position of body i at time t_idx

            # Calculate the distance from body i to the center of mass (COM)
            dist = np.sqrt((xi - com[0])**2 + (yi - com[1])**2 + (zi - com[2])**2)

            # If within radius R, increment the count
            if dist < R:
                count += 1

        num_stars_within_R[t_idx] = count  # Store the count for this time step

    return num_stars_within_R  # Return the array of counts

def get_centre_mass(solution, m):
    ''' 
    Compute the Centre of mass of all the stars at the first step
    
    Parameters:
    - solution (np.array): An array containing the positions and velocities of all bodies in the system.
      It is assumed that the first 3 columns correspond to the positions (x, y, z), and the next 3 columns
      correspond to the velocities (vx, vy, vz).
    - m (int): The mass of the objects
    
    Returns:
    - com (np.array): A 1D array containing the x, y, and z coordinates of the center of mass at t=0.
    '''
    x_com = np.sum(m * solution.y[0::3, 0]) / np.sum(m)  # x-component of COM
    y_com = np.sum(m * solution.y[1::3, 0]) / np.sum(m)  # y-component of COM
    z_com = np.sum(m * solution.y[2::3, 0]) / np.sum(m)  # z-component of COM
    com = np.array([x_com, y_com, z_com])  # Center of mass at t=0
    return com
