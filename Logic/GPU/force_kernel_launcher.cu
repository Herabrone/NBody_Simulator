#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// External force computation kernel
extern __global__ void compute_forces(
    const float3* body_pos, const float* body_mass, int num_bodies,
    const float3* node_com, const float* node_mass, const float* node_size, const float3* node_center,
    const int* node_is_leaf, const int* node_children, int num_nodes,
    float3* out_acc, float theta, float softening, float G
);

// Global device pointers for Barnes-Hut tree arrays
static float3* d_node_com       = nullptr;
static float*  d_node_mass      = nullptr;
static float*  d_node_size      = nullptr;
static float3* d_node_center    = nullptr;
static int*    d_node_is_leaf   = nullptr;
static int*    d_node_children  = nullptr;
static int     num_nodes_global = 0;

// Upload flattened tree data once to the GPU
extern "C" void upload_tree(
    float* node_com_flat, float* node_mass_flat, float* node_size_flat,
    float* node_center_flat, int* node_is_leaf_flat, int* node_children_flat,
    int num_nodes
) {
    // Free previous allocations if they exist
    if (d_node_com) {
        cudaFree(d_node_com);
        cudaFree(d_node_mass);
        cudaFree(d_node_size);
        cudaFree(d_node_center);
        cudaFree(d_node_is_leaf);
        cudaFree(d_node_children);
    }

    num_nodes_global = num_nodes;

    size_t f3_M = num_nodes * sizeof(float3);
    size_t f1_M = num_nodes * sizeof(float);
    size_t i1_M = num_nodes * sizeof(int);
    size_t i8_M = num_nodes * 8 * sizeof(int);

    cudaMalloc(&d_node_com, f3_M);
    cudaMalloc(&d_node_mass, f1_M);
    cudaMalloc(&d_node_size, f1_M);
    cudaMalloc(&d_node_center, f3_M);
    cudaMalloc(&d_node_is_leaf, i1_M);
    cudaMalloc(&d_node_children, i8_M);

    cudaMemcpy(d_node_com, node_com_flat, f3_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_mass, node_mass_flat, f1_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_size, node_size_flat, f1_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_center, node_center_flat, f3_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_is_leaf, node_is_leaf_flat, i1_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_children, node_children_flat, i8_M, cudaMemcpyHostToDevice);
}

// Launch the force kernel using pre-uploaded tree data
extern "C" void launch_cuda_force_kernel(
    float* body_pos_flat, float* body_mass_flat, int num_bodies,
    float* out_acc_flat,
    float theta, float softening, float G
) {
    float3* d_body_pos = nullptr;
    float*  d_body_mass = nullptr;
    float3* d_out_acc = nullptr;

    size_t f3_N = num_bodies * sizeof(float3);
    size_t f1_N = num_bodies * sizeof(float);

    cudaMalloc(&d_body_pos, f3_N);
    cudaMalloc(&d_body_mass, f1_N);
    cudaMalloc(&d_out_acc, f3_N);

    cudaMemcpy(d_body_pos, body_pos_flat, f3_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_body_mass, body_mass_flat, f1_N, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (num_bodies + threadsPerBlock - 1) / threadsPerBlock;

    compute_forces<<<blocksPerGrid, threadsPerBlock>>>(
        d_body_pos, d_body_mass, num_bodies,
        d_node_com, d_node_mass, d_node_size, d_node_center,
        d_node_is_leaf, d_node_children, num_nodes_global,
        d_out_acc, theta, softening, G
    );

    cudaMemcpy(out_acc_flat, d_out_acc, f3_N, cudaMemcpyDeviceToHost);

    cudaFree(d_body_pos);
    cudaFree(d_body_mass);
    cudaFree(d_out_acc);
}
