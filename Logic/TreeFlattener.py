import numpy as np

class FlattenedNode:
    def __init__(self, center_of_mass, total_mass, size, center, is_leaf, child_indices):
        self.center_of_mass = center_of_mass
        self.total_mass = total_mass
        self.size = size
        self.center = center
        self.is_leaf = is_leaf
        self.child_indices = child_indices

def flatten_tree(node):
    flat_nodes = []

    def recurse(current_node):
        idx = len(flat_nodes)
        child_indices = [-1] * 8
        flat_nodes.append(None)  # placeholder

        for i, child in enumerate(current_node.children):
            if child is not None:
                child_indices[i] = recurse(child)

        flat_nodes[idx] = FlattenedNode(
            center_of_mass=current_node.center_of_mass,
            total_mass=current_node.total_mass,
            size=current_node.size,
            center=current_node.center,
            is_leaf=current_node.is_leaf(),
            child_indices=child_indices
        )
        return idx

    recurse(node)
    return flat_nodes