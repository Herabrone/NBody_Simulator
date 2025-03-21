import numpy as np
from Logic.BarnesHut import OctreeNode

class ForceCalculator:
    def __init__(self, G=6.67430e-11):
        self.G = G

    def compute_forces(self, node, body, theta):
        """
        Public method to compute gravitational forces on a given body using Barnes-Hut.
        """

        if node.is_leaf():
            for other_body in node.bodies:
                if other_body is not body:
                    self._apply_gravitational_force(body, other_body)
        else:
            distance = np.linalg.norm(node.center_of_mass - body.position)
            if node.size / distance < theta:
                self._apply_gravitational_force(body, node)
            else:
                for child in node.children:
                    if child is not None:
                        self.compute_forces(child, body, theta)

    def _apply_gravitational_force(self, body, other):
        """
        Private helper to apply the gravitational force from 'other' onto 'body'.
        'other' can be a single body or an OctreeNode.
        """
        if isinstance(other, OctreeNode):
            r = other.center_of_mass - body.position
            mass = other.total_mass
        else:
            r = other.position - body.position
            mass = other.mass
        distance = np.linalg.norm(r)
        force = self.G * body.mass * mass / distance**2
        # Update the body's acceleration (assuming acceleration is already initialized)
        body.acceleration += (force / body.mass) * (r / distance)