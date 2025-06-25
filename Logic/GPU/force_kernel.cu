// force_kernel.cu
#include <math.h>

__global__ void compute_forces(
    const float3* body_pos,
    const float* body_mass,
    int num_bodies,

    const float3* node_com,
    const float* node_mass,
    const float* node_size,
    const float3* node_center,
    const int* node_is_leaf,
    const int* node_children,  // flattened: 8 * num_nodes
    int num_nodes,

    float3* out_acc,
    float theta,
    float softening,
    float G
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    float3 pos_i = body_pos[i];
    float3 acc = {0.0f, 0.0f, 0.0f};

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int node_idx = stack[--sp];

        float3 com = node_com[node_idx];
        float dx = com.x - pos_i.x;
        float dy = com.y - pos_i.y;
        float dz = com.z - pos_i.z;
        float dist2 = dx*dx + dy*dy + dz*dz + softening*softening;
        float dist = sqrtf(dist2);

        float size = node_size[node_idx];
        bool leaf = node_is_leaf[node_idx];

        if ((size / dist) < theta || leaf) {
            if (dist > 1e-5f) {
                float m = node_mass[node_idx];
                float f = G * m / dist2;
                acc.x += f * dx / dist;
                acc.y += f * dy / dist;
                acc.z += f * dz / dist;
            }
        } else {
            for (int j = 0; j < 8; j++) {
                int child = node_children[node_idx * 8 + j];
                if (child != -1 && sp < 64)
                    stack[sp++] = child;
            }
        }
    }

    out_acc[i] = acc;
}
