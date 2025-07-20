#pragma once

#include <cooperative_groups.h>
#include <cuda/cmath>
#include <cuda_occupancy.h>

namespace cg = cooperative_groups;

/**
 * @brief Launches a kernel with cooperative groups.
 *
 * @tparam F The kernel function type.
 * @param f The kernel function to launch.
 * @param num_element The total number of elements to process.
 */
template<class F> __global__ void kernel(F f, std::size_t num_element)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    for (auto idx = grid.thread_rank(); idx < num_element; idx += grid.size()) { f(idx); }
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
    int min_grid_size = 0;
    int block_size = 0;

    // Let CUDA calculate the optimal block size
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel<KernelFunc>);

    kernel<<<cuda::ceil_div(num_elements, block_size), block_size, 0, stream>>>(lambda, num_elements);
}
