#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

extern "C" void launch_cuda_force_kernel(
    float* body_pos, float* body_mass, int num_bodies,
    float* node_com, float* node_mass, float* node_size,
    float* node_center, int* node_is_leaf, int* node_children, int num_nodes,
    float* out_acc,
    float theta, float softening, float G
);

extern "C" void upload_tree(
    float* node_com, float* node_mass, float* node_size,
    float* node_center, int* node_is_leaf, int* node_children,
    int num_nodes
);

// This is the declaration of the kernel from the .cu file
void compute_forces(
    const float3*, const float*, int,
    const float3*, const float*, const float*, const float3*,
    const int*, const int*, int,
    float3*, float, float, float
);

namespace py = pybind11;

py::array_t<float> compute_gpu_acceleration(
    py::array_t<float> body_positions,   // shape (N, 3)
    py::array_t<float> body_masses,      // shape (N,)
    py::array_t<float> node_com,         // shape (M, 3)
    py::array_t<float> node_masses,      // shape (M,)
    py::array_t<float> node_sizes,       // shape (M,)
    py::array_t<float> node_centers,     // shape (M, 3)
    py::array_t<int> node_is_leaf,       // shape (M,)
    py::array_t<int> node_children,      // shape (M, 8)
    float theta,
    float softening,
    float G
) {
    int N = body_positions.shape(0);
    int M = node_com.shape(0);

    py::array_t<float> accels({N, 3});

    launch_cuda_force_kernel(
        body_positions.mutable_data(), body_masses.mutable_data(), N,
        node_com.mutable_data(), node_masses.mutable_data(),
        node_sizes.mutable_data(), node_centers.mutable_data(),
        node_is_leaf.mutable_data(), node_children.mutable_data(), M,
        accels.mutable_data(),
        theta, softening, G
    );

    return accels;
}

void upload_tree_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> node_com,
    py::array_t<float, py::array::c_style | py::array::forcecast> node_mass,
    py::array_t<float, py::array::c_style | py::array::forcecast> node_size,
    py::array_t<float, py::array::c_style | py::array::forcecast> node_center,
    py::array_t<int, py::array::c_style | py::array::forcecast> node_is_leaf,
    py::array_t<int, py::array::c_style | py::array::forcecast> node_children
) {
    int num_nodes = node_com.shape(0);
    upload_tree(
        (float*)node_com.data(), (float*)node_mass.data(), (float*)node_size.data(),
        (float*)node_center.data(), (int*)node_is_leaf.data(), (int*)node_children.data(),
        num_nodes
    );
}

PYBIND11_MODULE(force_gpu, m) {
    m.def("compute_gpu_acceleration", &compute_gpu_acceleration, "Compute gravitational accelerations with CUDA");
    m.def("upload_tree", &upload_tree_py, "Upload flattened tree data to GPU");
}
