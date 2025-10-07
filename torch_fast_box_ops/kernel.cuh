#pragma once

#include <cuda/cmath>
#include <cuda_occupancy.h>

#ifdef __CUDACC__
#define TFBO_HOST_DEVICE __host__ __device__
#else
#define TFBO_HOST_DEVICE
#endif

/**
 * @brief Launches a kernel that applies an elementwise operation over num_element.
 *
 * @tparam F The kernel function type.
 * @param f The kernel function to launch.
 * @param num_element The total number of elements to process.
 */
template<class F> __global__ void kernel(F f, std::size_t num_element)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_element) { f(idx); }
}


/***
 * @brief Launches an elementwise kernel with optimal block size.
 * @param lambda The kernel function to launch, the input to this lambda will be the element index.
 * @param num_elements The total number of elements to process.
 * @param stream The CUDA stream to launch the kernel on.
 */
template<typename KernelFunc>
static void launch_elementwise_kernel(KernelFunc lambda, size_t num_elements, cudaStream_t stream)
{
    if (num_elements == 0) { return; }// No work
    int min_grid_size = 0;
    int block_size = 0;

    // Let CUDA calculate the optimal block size
    const auto err = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel<KernelFunc>);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "Failed to calculate optimal block size for kernel: " + std::string(cudaGetErrorString(err)));
    }
    if (block_size == 0) { throw std::runtime_error("Failed to calculate optimal block size for kernel."); }

    kernel<<<cuda::ceil_div(num_elements, static_cast<size_t>(block_size)), block_size, 0, stream>>>(
        lambda, num_elements);
}
