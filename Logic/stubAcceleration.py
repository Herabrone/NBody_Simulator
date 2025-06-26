import sys
import os
import numpy as np

dll_path = r"C:\Users\darja\Documents\ASTRO 3180\Project REPO\NBody_Simulator\build\Logic\GPU\Release"
sys.path.append(dll_path)

import force_gpu

n = 100000000  # number of bodies
#Instant for up to 100 million bodies
m = 1  # node

# Dummy data :p
body_pos = np.random.rand(n, 3).astype(np.float32)
body_mass = np.ones(n, dtype=np.float32)
node_com = np.random.rand(m, 3).astype(np.float32)
node_mass = np.ones(m, dtype=np.float32)
node_size = np.ones(m, dtype=np.float32)
node_center = np.random.rand(m, 3).astype(np.float32)
node_is_leaf = np.ones(m, dtype=np.int32)
node_children = -np.ones((m, 8), dtype=np.int32)

# Parameters
theta = 0.5
softening = 1e-2
G = 1.0

# Call GPU function
acc = force_gpu.compute_gpu_acceleration(
    body_pos, body_mass, node_com, node_mass,
    node_size, node_center, node_is_leaf, node_children,
    theta, softening, G
)

print("Accelerations:\n", acc)
