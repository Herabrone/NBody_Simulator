#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

extern __global__ void compute_forces(
    const float3*, const float*, int,
    const float3*, const float*, const float*, const float3*,
    const int*, const int*, int,
    float3*, float, float, float
);

extern "C" void launch_cuda_force_kernel(
    float* body_pos_flat, float* body_mass, int num_bodies,
    float* node_com_flat, float* node_mass, float* node_size,
    float* node_center_flat, int* node_is_leaf, int* node_children, int num_nodes,
    float* out_acc_flat,
    float theta, float softening, float G
) {
    float3* d_body_pos;
    float* d_body_mass;
    float3* d_node_com;
    float* d_node_mass;
    float* d_node_size;
    float3* d_node_center;
    int* d_node_is_leaf;
    int* d_node_children;
    float3* d_out_acc;

    size_t f3_N = num_bodies * sizeof(float3);
    size_t f3_M = num_nodes * sizeof(float3);
    size_t f1_N = num_bodies * sizeof(float);
    size_t f1_M = num_nodes * sizeof(float);
    size_t i1_M = num_nodes * sizeof(int);
    size_t i8_M = num_nodes * 8 * sizeof(int);

    cudaMalloc(&d_body_pos, f3_N);
    cudaMalloc(&d_body_mass, f1_N);
    cudaMalloc(&d_node_com, f3_M);
    cudaMalloc(&d_node_mass, f1_M);
    cudaMalloc(&d_node_size, f1_M);
    cudaMalloc(&d_node_center, f3_M);
    cudaMalloc(&d_node_is_leaf, i1_M);
    cudaMalloc(&d_node_children, i8_M);
    cudaMalloc(&d_out_acc, f3_N);

    cudaMemcpy(d_body_pos, body_pos_flat, f3_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_body_mass, body_mass, f1_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_com, node_com_flat, f3_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_mass, node_mass, f1_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_size, node_size, f1_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_center, node_center_flat, f3_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_is_leaf, node_is_leaf, i1_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_children, node_children, i8_M, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (num_bodies + threadsPerBlock - 1) / threadsPerBlock;

    compute_forces<<<blocksPerGrid, threadsPerBlock>>>(
        d_body_pos, d_body_mass, num_bodies,
        d_node_com, d_node_mass, d_node_size, d_node_center,
        d_node_is_leaf, d_node_children, num_nodes,
        d_out_acc, theta, softening, G
    );

    cudaMemcpy(out_acc_flat, d_out_acc, f3_N, cudaMemcpyDeviceToHost);

    cudaFree(d_body_pos);
    cudaFree(d_body_mass);
    cudaFree(d_node_com);
    cudaFree(d_node_mass);
    cudaFree(d_node_size);
    cudaFree(d_node_center);
    cudaFree(d_node_is_leaf);
    cudaFree(d_node_children);
    cudaFree(d_out_acc);
}
