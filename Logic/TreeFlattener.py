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

def flattened_nodes_to_numpy(flat_nodes):
    # This method will convert the flattened tree into an SoA that will be passed to the 
    # CUDA program for the force calculations
    n = len(flat_nodes)

    coms     = np.zeros((n, 3), dtype=np.float32)
    masses   = np.zeros(n, dtype=np.float32)
    sizes    = np.zeros(n, dtype=np.float32)
    centers  = np.zeros((n, 3), dtype=np.float32)
    is_leaf  = np.zeros(n, dtype=np.int32)
    children = np.full((n, 8), -1, dtype=np.int32)

    for i, node in enumerate(flat_nodes):
        coms[i]     = node.center_of_mass
        masses[i]   = node.total_mass
        sizes[i]    = node.size
        centers[i]  = node.center
        is_leaf[i]  = int(node.is_leaf)
        children[i] = node.child_indices

    return {
        "center_of_mass": coms,
        "total_mass": masses,
        "size": sizes,
        "center": centers,
        "is_leaf": is_leaf,
        "children": children
    }