#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "iou_common.cuh"

auto box_area(const torch::Tensor &boxes) -> torch::Tensor
{
    TORCH_CHECK(boxes.size(-1) == 4, "Input tensor must have shape (..., 4) for boxes");
    auto boxes_c = boxes.contiguous();

    // Output shape is the same shape as input except the last dimension
    // for compatibility with any batch or unbatched input.
    auto output_shape = boxes.sizes().vec();
    output_shape.back() = 1;// Area is a single value per box

    // Promote output type to float32 for float16 to prevent overflow
    auto output_dtype = boxes.scalar_type();
    if (output_dtype == torch::kFloat16) { output_dtype = torch::kFloat32; }
    auto output = torch::empty(output_shape, boxes.options().dtype(output_dtype));

    TFBO_DISPATCH_BOX_TYPES(boxes.scalar_type(), "box_area", [&] {
        const auto boxes_ptr = static_cast<const XYXY<scalar_t> *>(boxes_c.const_data_ptr());
        const int64_t num_boxes = boxes.numel() / 4;
        auto areas_ptr = output.mutable_data_ptr<area_t<scalar_t>>();

        if (boxes.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) { areas_ptr[idx] = box_area_op(boxes_ptr[idx]); };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            std::transform(boxes_ptr, boxes_ptr + num_boxes, areas_ptr, box_area_op<scalar_t>);
        }
    });

    return output;
}

template<typename T> auto TFBO_HOST_DEVICE box_area_backward_(T grad, XYXY<T> box) -> XYXY<T>
{
    XYXY<T> grad_box = box_area_grad(box);
    grad_box.x1 = grad * grad_box.x1;
    grad_box.y1 = grad * grad_box.y1;
    grad_box.x2 = grad * grad_box.x2;
    grad_box.y2 = grad * grad_box.y2;
    return grad_box;
}

auto box_area_backward(const torch::Tensor &grad, const torch::Tensor &boxes) -> torch::Tensor
{
    TORCH_CHECK(boxes.size(-1) == 4, "Boxes tensor must have shape (..., 4)");
    const auto common_dtype = c10::promoteTypes(grad.scalar_type(), boxes.scalar_type());
    auto boxes_c = boxes.contiguous().to(common_dtype);
    auto grad_c = grad.contiguous().to(common_dtype);

    auto input_grad = torch::empty_like(boxes_c);
    TFBO_DISPATCH_BOX_TYPES(common_dtype, "box_area_backward", [&] {
        auto grad_ptr = grad_c.const_data_ptr<scalar_t>();
        auto boxes_ptr = static_cast<const XYXY<scalar_t> *>(boxes_c.const_data_ptr());
        auto input_grad_ptr = static_cast<XYXY<scalar_t> *>(input_grad.mutable_data_ptr());

        if (boxes.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                input_grad_ptr[idx] = box_area_backward_(grad_ptr[idx], boxes_ptr[idx]);
            };
            launch_elementwise_kernel(kernel, grad.numel(), at::cuda::getCurrentCUDAStream());
        } else {
            std::transform(grad_ptr, grad_ptr + grad.numel(), boxes_ptr, input_grad_ptr, box_area_backward_<scalar_t>);
        }
    });

    return input_grad;
}

template<typename IouType, typename In, typename Out>
auto TFBO_HOST_DEVICE
    box_iou_fn(const XYXY<In> &box1, const area_t<In> area1, const XYXY<In> &box2, const area_t<In> area2) -> Out
{
    // Use float32 for intermediate calculations to prevent overflow with float16
    // Union area and enclosing area can easily exceed float16 max (65504)
    // Only promote c10::Half, not BFloat16 which has same exponent range as float32
    using ComputeT = std::conditional_t<std::is_same_v<Out, c10::Half>, float, Out>;

    const area_t<In> intersection = box_intersection_area(box1, box2);
    const ComputeT union_area =
        static_cast<ComputeT>(area1) + static_cast<ComputeT>(area2) - static_cast<ComputeT>(intersection);
    const ComputeT iou = static_cast<ComputeT>(intersection) / union_area;
    if constexpr (std::is_same_v<IouType, iou_tag>) {
        return static_cast<Out>(iou);
    } else if constexpr (std::is_same_v<IouType, giou_tag>) {
        XYXY<In> enclosing_box = min_enclosing_box(box1, box2);
        ComputeT enclosing_area = std::max(static_cast<ComputeT>(box_area_op(enclosing_box)), static_cast<ComputeT>(0));
        return static_cast<Out>(iou - (enclosing_area - union_area) / enclosing_area);
    } else if constexpr (std::is_same_v<IouType, diou_tag> || std::is_same_v<IouType, ciou_tag>) {
        // Distance squared can easily exceed float16 max (65504) with typical box coordinates
        XYXY<In> enclosing_box = min_enclosing_box(box1, box2);
        const auto diag_dist_sq = dist_sq(static_cast<ComputeT>(enclosing_box.x2 - enclosing_box.x1),
            static_cast<ComputeT>(enclosing_box.y2 - enclosing_box.y1));
        const CXCY<ComputeT> box1c(box1);
        const CXCY<ComputeT> box2c(box2);
        const auto cent_dist_sq = dist_sq(box1c.cx - box2c.cx, box1c.cy - box2c.cy);
        const ComputeT eps = 1e-7f;
        const ComputeT diou = iou - cent_dist_sq / (diag_dist_sq + eps);
        if constexpr (std::is_same_v<IouType, diou_tag>) {
            return static_cast<Out>(diou);
        } else {
            const ComputeT aspect =
                std::atan(static_cast<ComputeT>(box1.width()) / (static_cast<ComputeT>(box1.height()) + eps))
                - std::atan(static_cast<ComputeT>(box2.width()) / (static_cast<ComputeT>(box2.height()) + eps));
            const ComputeT v = (4.f / (M_PIf * M_PIf)) * aspect * aspect;
            const ComputeT alpha = v / (1 - iou + v + eps);
            return static_cast<Out>(diou - alpha * v);
        }
    }
}

template<typename IouType>
void box_iou_cpu_impl(const torch::Tensor &boxes1, const torch::Tensor &boxes2, torch::Tensor &output)
{
    TFBO_DISPATCH_BOX_TYPES(boxes1.scalar_type(), "box_iou", [&] {
        const auto B = boxes1.size(0);
        const auto N = boxes1.size(1);
        const auto M = boxes2.size(1);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());

        using output_t = std::conditional_t<std::is_integral_v<scalar_t>, float, scalar_t>;
        auto output_ptr = static_cast<output_t *>(output.mutable_data_ptr());

        std::vector<area_t<scalar_t>> areas2(M);

        for (int b = 0; b < B; ++b) {
            std::transform(boxes2_ptr + b * M, boxes2_ptr + (b + 1) * M, areas2.begin(), box_area_op<scalar_t>);
            for (int n = 0; n < N; ++n) {
                const auto &box1 = boxes1_ptr[b * N + n];
                const area_t<scalar_t> area1 = box_area_op(box1);
                for (int m = 0; m < M; ++m) {
                    const auto &box2 = boxes2_ptr[b * M + m];
                    output_ptr[b * N * M + n * M + m] =
                        box_iou_fn<IouType, scalar_t, output_t>(box1, area1, box2, areas2[m]);
                }
            }
        }
    });
}


constexpr uint box_iou_block_size_x = 32;
constexpr uint box_iou_block_size_y = 16;

template<typename T, typename IouType, typename U = std::conditional_t<std::is_integral_v<T>, float, T>>
__global__ void box_iou_kernel(const XYXY<T> *__restrict__ boxes1,
    const XYXY<T> *__restrict__ boxes2,
    U *output,
    unsigned int N,
    unsigned int M)
{
    __shared__ XYXY<T> shared_boxes1[box_iou_block_size_y];
    __shared__ area_t<T> shared_boxes1_area[box_iou_block_size_y];
    __shared__ XYXY<T> shared_boxes2[box_iou_block_size_x];
    __shared__ area_t<T> shared_boxes2_area[box_iou_block_size_x];

    unsigned int b = blockIdx.x;
    unsigned int n = blockIdx.z * blockDim.y + threadIdx.y;
    unsigned int m = blockIdx.y * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && m < M) {
        shared_boxes2[threadIdx.x] = boxes2[b * M + m];
        shared_boxes2_area[threadIdx.x] = box_area_op(shared_boxes2[threadIdx.x]);
    }
    unsigned int n_ = blockIdx.z * blockDim.y + threadIdx.x;
    if (threadIdx.y == 1 && threadIdx.x < blockDim.y && n_ < N) {
        shared_boxes1[threadIdx.x] = boxes1[b * N + n_];
        shared_boxes1_area[threadIdx.x] = box_area_op(shared_boxes1[threadIdx.x]);
    }
    __syncthreads();
    if (n >= N || m >= M) return;// Prevent out-of-bounds access

    output[b * N * M + n * M + m] = box_iou_fn<IouType, T, U>(shared_boxes1[threadIdx.y],
        shared_boxes1_area[threadIdx.y],
        shared_boxes2[threadIdx.x],
        shared_boxes2_area[threadIdx.x]);
}

// Flattened version for large batch sizes
template<typename T, typename IouType, typename U = std::conditional_t<std::is_integral_v<T>, float, T>>
__global__ void box_iou_flat_kernel(const XYXY<T> *__restrict__ boxes1,
    const XYXY<T> *__restrict__ boxes2,
    U *output,
    unsigned int B,
    unsigned int N,
    unsigned int M)
{
    const unsigned int global_tid = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int total_elements = B * N * M;
    if (global_tid >= total_elements) { return; }

    // Decompose global thread ID into b, n, m indices
    const unsigned int nm = N * M;
    const unsigned int b = global_tid / nm;
    const unsigned int local_tid = global_tid % nm;
    const unsigned int n = local_tid / M;
    const unsigned int m = local_tid % M;

    const auto box1 = boxes1[b * N + n];
    const auto box2 = boxes2[b * M + m];
    output[global_tid] = box_iou_fn<IouType, T, U>(box1, box_area_op(box1), box2, box_area_op(box2));
}

template<typename IouType>
void box_iou_gpu_impl(const torch::Tensor &boxes1, const torch::Tensor &boxes2, torch::Tensor &output)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    TFBO_DISPATCH_BOX_TYPES(boxes1.scalar_type(), "box_iou_gpu", [&] {
        const uint B = boxes1.size(0);
        const uint N = boxes1.size(1);
        const uint M = boxes2.size(1);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto output_ptr =
            static_cast<std::conditional_t<std::is_integral_v<scalar_t>, float, scalar_t> *>(output.mutable_data_ptr());

        // Use flattened kernel when N and M are small (inefficient for 2D block strategy)
        if (N < box_iou_block_size_x || M < box_iou_block_size_y) {
            int min_grid_size = 0;
            int block_size = 0;
            const auto err =
                cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, box_iou_flat_kernel<scalar_t, IouType>);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "Failed to calculate optimal block size for kernel: " + std::string(cudaGetErrorString(err)));
            }
            if (block_size == 0) { throw std::runtime_error("Failed to calculate optimal block size for kernel."); }

            const auto grid_size = cuda::ceil_div(B * N * M, static_cast<unsigned int>(block_size));
            box_iou_flat_kernel<scalar_t, IouType>
                <<<grid_size, block_size, 0, stream>>>(boxes1_ptr, boxes2_ptr, output_ptr, B, N, M);
        } else {
            // Use optimized 2D block kernel for larger N and M with reasonable batch sizes
            auto block_dim = dim3(box_iou_block_size_x, box_iou_block_size_y);
            auto grid_dim = dim3(B, cuda::ceil_div(M, block_dim.x), cuda::ceil_div(N, block_dim.y));
            box_iou_kernel<scalar_t, IouType>
                <<<grid_dim, block_dim, 0, stream>>>(boxes1_ptr, boxes2_ptr, output_ptr, N, M);
        }
    });
}

auto regularize_shape_for_iou(const torch::Tensor &boxes1, const torch::Tensor &boxes2, const torch::Tensor &output)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
    torch::Tensor boxes1_flat, boxes2_flat, output_flat;
    if (boxes1.ndimension() == 2) {
        boxes1_flat = boxes1.unsqueeze(0);
        boxes2_flat = boxes2.unsqueeze(0);
        output_flat = output.unsqueeze(0);
    } else if (boxes1.ndimension() > 3) {
        boxes1_flat = boxes1.flatten(0, -3);
        boxes2_flat = boxes2.flatten(0, -3);
        output_flat = output.flatten(0, -3);
    } else {
        boxes1_flat = boxes1;
        boxes2_flat = boxes2;
        output_flat = output;
    }
    return { boxes1_flat, boxes2_flat, output_flat };
}

auto create_iou_output_tensor(const torch::Tensor &boxes1, const torch::Tensor &boxes2) -> torch::Tensor
{
    auto output_shape = boxes1.sizes().vec();
    output_shape.back() = boxes2.size(-2);// Replace '4' with the number of boxes in boxes2
    auto opts = boxes1.options();
    opts = opts.dtype(c10::promoteTypes(boxes1.scalar_type(), boxes2.scalar_type()));
    if (c10::isIntegralType(opts.dtype().toScalarType(), true)) { opts = opts.dtype(torch::kFloat32); }
    return torch::empty(output_shape, opts);
}

template<typename IouType> auto box_iou(const torch::Tensor &boxes1, const torch::Tensor &boxes2) -> torch::Tensor
{
    TORCH_CHECK(boxes1.size(-1) == 4 && boxes2.size(-1) == 4, "Input tensors must have shape (..., 4) for boxes");
    TORCH_CHECK(boxes1.ndimension() == boxes2.ndimension(),
        "Input tensors boxes1 and boxes2 must have the same number of dimensions");
    TORCH_CHECK(boxes1.ndimension() >= 2, "Input tensors boxes1 and boxes2 must have at least 2 dimensions");

    auto output = create_iou_output_tensor(boxes1, boxes2);
    if (output.numel() == 0) { return output; }

    // Regularize the shape to Batch x Nboxes x 4
    torch::Tensor boxes1_flat, boxes2_flat, output_flat;
    std::tie(boxes1_flat, boxes2_flat, output_flat) = regularize_shape_for_iou(boxes1, boxes2, output);

    const auto common_dtype = c10::promoteTypes(boxes1_flat.scalar_type(), boxes2_flat.scalar_type());
    boxes1_flat = boxes1_flat.contiguous().to(common_dtype);
    boxes2_flat = boxes2_flat.contiguous().to(common_dtype);

    if (boxes1_flat.is_cuda()) {
        box_iou_gpu_impl<IouType>(boxes1_flat, boxes2_flat, output_flat);
    } else {
        box_iou_cpu_impl<IouType>(boxes1_flat, boxes2_flat, output_flat);
    }

    return output;
}


TORCH_LIBRARY_IMPL(box_ops, CPU, m)
{
    m.impl("box_area", &box_area);
    m.impl("box_iou", &box_iou<iou_tag>);
    m.impl("generalized_box_iou", &box_iou<giou_tag>);
    m.impl("distance_box_iou", &box_iou<diou_tag>);
    m.impl("complete_box_iou", &box_iou<ciou_tag>);

    m.impl("box_area_backward", &box_area_backward);
}

TORCH_LIBRARY_IMPL(box_ops, CUDA, m)
{
    m.impl("box_area", &box_area);
    m.impl("box_iou", &box_iou<iou_tag>);
    m.impl("generalized_box_iou", &box_iou<giou_tag>);
    m.impl("distance_box_iou", &box_iou<diou_tag>);
    m.impl("complete_box_iou", &box_iou<ciou_tag>);

    m.impl("box_area_backward", &box_area_backward);
}
