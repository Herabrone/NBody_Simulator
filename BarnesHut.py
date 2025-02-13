import numpy as np



class Body:
    '''
    Represents a celestial body in the simulation.

    Attributes:
    - mass (float): The mass of the body.
    - position (numpy array): The position of the body in 3D space (x, y, z).
    - velocity (numpy array): The velocity vector of the body in 3D space (vx, vy, vz).
    - acceleration (numpy array): The acceleration vector of the body in 3D space, initialized to zero.
    
    Parameters:
    - mass (float): Initial mass of the body.
    - position (numpy array): Initial position as a 3D vector.
    - velocity (numpy array): Initial velocity as a 3D vector.

    '''

    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = np.zeros_like(position)

# Define the OctreeNode class
class OctreeNode:
    '''
    Represents a node in the octree structure used for the Barnes-Hut algorithm.

    Attributes:
    - center (numpy array): The geometric center of the node in 3D space.
    - size (float): The side length of the cubic node.
    - bodies (list): List of Body objects contained within the node.
    - children (list): List of 8 child nodes, corresponding to octants of the cube. Initialized to None.
    - center_of_mass (numpy array): The combined center of mass of all bodies in the node.
    - total_mass (float): The total mass of all bodies in the node.

    Parameters:
    - center (numpy array): The geometric center of the node in 3D space.
    - size (float): The side length of the cubic node.

    '''
    def __init__(self, center, size):
        self.center = center  
        self.size = size  
        self.bodies = []  
        self.children = [None] * 8  
        self.center_of_mass = np.zeros(3)
        self.total_mass = 0.0

    def is_leaf(self):
        '''
        Check if the node is a leaf node (i.e., it has no child nodes).

        Returns:
        - bool: True if the node has no child nodes, otherwise False.
        '''
        return all(child is None for child in self.children)

    def update_center_of_mass(self):
        '''
        Calculate and update the center of mass and total mass for the node based on the bodies it contains.

        If the node contains bodies, the center of mass is the mass-weighted average of the positions of the
        bodies, and the total mass is the sum of their masses. If the node is empty, these values are reset
        to zero.
        '''
        if self.bodies:
            total_mass = sum(body.mass for body in self.bodies)
            center_of_mass = sum(body.mass * body.position for body in self.bodies) / total_mass
            self.center_of_mass = center_of_mass
            self.total_mass = total_mass
        else:
            self.center_of_mass = np.zeros(3)
            self.total_mass = 0.0

# Function to insert a body into the octree
def insert_body(node, body, max_bodies_per_node=1):
    '''
    Insert a body into the octree, redistributing bodies if necessary.

    Parameters:
    - node (OctreeNode): The current node in the octree.
    - body (Body): The body to insert.
    - max_bodies_per_node (int): Maximum number of bodies allowed in a leaf node before subdivision.

    '''
    if node.is_leaf():
        if len(node.bodies) < max_bodies_per_node:
            node.bodies.append(body)
            node.update_center_of_mass()
        else:
            # Subdivide the node and redistribute bodies
            subdivide_node(node)
            insert_body(node, body)
    else:
        # Determine which child node the body belongs to and insert it
        index = get_child_index(node, body)
        insert_body(node.children[index], body)
    node.update_center_of_mass()

# Function to subdivide a node and redistribute bodies
def subdivide_node(node):
    '''
    Subdivide a node into 8 child nodes and redistribute its bodies among the children.

    Parameters:
    - node (OctreeNode): The node to subdivide.
    '''
    half_size = node.size / 2
    offsets = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
               (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    
    # Create child nodes
    for i, offset in enumerate(offsets):
        center = node.center + half_size * np.array(offset)
        node.children[i] = OctreeNode(center, half_size)
    
    # Redistribute bodies into the appropriate child nodes
    for body in node.bodies:
        index = get_child_index(node, body)
        insert_body(node.children[index], body)
    
    # Clear the bodies list in the current node as they are now in child nodes
    node.bodies = []

# Function to determine the index of the child node where the body should be inserted
def get_child_index(node, body):
    '''
    Determine the index of the child node where a body should be placed.

    Parameters:
    - node (OctreeNode): The current node.
    - body (Body): The body to locate.

    Returns:
    - int: Index of the child node (0 to 7).
    '''
    index = 0
    if body.position[0] > node.center[0]:
        index |= 4
    if body.position[1] > node.center[1]:
        index |= 2
    if body.position[2] > node.center[2]:
        index |= 1
    return index