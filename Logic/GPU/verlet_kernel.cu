#include <cuda_runtime.h>

__global__ void verlet_step(
    float3* positions,
    float3* velocities,
    float3* accelerations,
    float dt,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 pos = positions[i];
    float3 vel = velocities[i];
    float3 acc = accelerations[i];

    // Update position
    pos.x += vel.x * dt + 0.5f * acc.x * dt * dt;
    pos.y += vel.y * dt + 0.5f * acc.y * dt * dt;
    pos.z += vel.z * dt + 0.5f * acc.z * dt * dt;

    // Store old acceleration
    float3 old_acc = acc;

    // Synchronize to allow new accelerations to be computed externally here
    positions[i] = pos;
    velocities[i] = vel;  // Temporarily unchanged
    accelerations[i] = old_acc;
}

__global__ void verlet_velocity_update( // FOr updating the integrator
    float3* velocities,
    float3* acc_old,
    float3* acc_new,
    float dt,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    velocities[i].x += 0.5f * (acc_old[i].x + acc_new[i].x) * dt;
    velocities[i].y += 0.5f * (acc_old[i].y + acc_new[i].y) * dt;
    velocities[i].z += 0.5f * (acc_old[i].z + acc_new[i].z) * dt;
}