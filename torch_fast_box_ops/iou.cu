#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "boxes.cuh"
#include "kernel.cuh"


template<typename T> TFBO_HOST_DEVICE auto box_area_op(const XYXY<T> &box) -> T
{
    return (box.x2 - box.x1) * (box.y2 - box.y1);
}

auto box_area(const torch::Tensor &boxes) -> torch::Tensor
{
    TORCH_CHECK(boxes.stride(-1) == 1, "Input tensor must be contiguous in last dimension");
    TORCH_CHECK(boxes.size(-1) == 4, "Input tensor must have shape (..., 4) for boxes");

    // Output shape is the same shape as input except the last dimension
    // for compatibility with any batch or unbatched input.
    auto output_shape = boxes.sizes().vec();
    output_shape.back() = 1;// Area is a single value per box
    auto output = torch::empty(output_shape, boxes.options());

    TFBO_DISPATCH_BOX_TYPES(boxes.scalar_type(), "box_area", [&] {
        const auto boxes_ptr = static_cast<const XYXY<scalar_t> *>(boxes.const_data_ptr());
        const int64_t num_boxes = boxes.numel() / 4;
        auto areas_ptr = output.mutable_data_ptr<scalar_t>();

        if (boxes.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) { areas_ptr[idx] = box_area_op(boxes_ptr[idx]); };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            std::transform(boxes_ptr, boxes_ptr + num_boxes, areas_ptr, box_area_op<scalar_t>);
        }
    });

    return output;
}

template<typename T> auto TFBO_HOST_DEVICE box_area_grad(XYXY<T> box) -> XYXY<T>
{
    XYXY<T> grad_box;
    grad_box.x1 = box.y1 - box.y2;
    grad_box.y1 = box.x1 - box.x2;
    grad_box.x2 = box.y2 - box.y1;
    grad_box.y2 = box.x2 - box.x1;
    return grad_box;
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
    TORCH_CHECK(grad.stride(-1) == 1, "Gradient tensor must be contiguous in last dimension");
    TORCH_CHECK(boxes.stride(-1) == 1, "Boxes tensor must be contiguous in last dimension");
    TORCH_CHECK(boxes.size(-1) == 4, "Boxes tensor must have shape (..., 4)");

    auto input_grad = torch::empty_like(boxes);
    TFBO_DISPATCH_BOX_TYPES(boxes.scalar_type(), "box_area_backward", [&] {
        auto grad_ptr = grad.const_data_ptr<scalar_t>();
        auto boxes_ptr = static_cast<const XYXY<scalar_t> *>(boxes.const_data_ptr());
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

template<typename T> TFBO_HOST_DEVICE auto box_intersection(const XYXY<T> &box1, const XYXY<T> &box2) -> XYXY<T>
{
    XYXY<T> inter_box;
    inter_box.x1 = std::max(box1.x1, box2.x1);
    inter_box.y1 = std::max(box1.y1, box2.y1);
    inter_box.x2 = std::min(box1.x2, box2.x2);
    inter_box.y2 = std::min(box1.y2, box2.y2);
    return inter_box;
}


template<typename T> TFBO_HOST_DEVICE auto box_intersection_area(const XYXY<T> &box1, const XYXY<T> &box2) -> T
{
    auto inter_box = box_intersection(box1, box2);
    T inter_area = std::max(inter_box.x2 - inter_box.x1, static_cast<T>(0))
                   * std::max(inter_box.y2 - inter_box.y1, static_cast<T>(0));
    return inter_area;
}

/**
 * @brief Compute the minimum enclosing box of two boxes.
 *
 * @tparam T
 * @param box1
 * @param box2
 * @return XYXY<T> minumum enclosing box that contains both box1 and box2.
 */
template<typename T> TFBO_HOST_DEVICE auto min_enclosing_box(const XYXY<T> &box1, const XYXY<T> &box2) -> XYXY<T>
{
    XYXY<T> enclosing_box;
    enclosing_box.x1 = std::min(box1.x1, box2.x1);
    enclosing_box.y1 = std::min(box1.y1, box2.y1);
    enclosing_box.x2 = std::max(box1.x2, box2.x2);
    enclosing_box.y2 = std::max(box1.y2, box2.y2);
    return enclosing_box;
}

struct iou_type_tag
{
};
struct iou_tag : iou_type_tag
{
};
struct giou_tag : iou_type_tag
{
};
struct diou_tag : iou_type_tag
{
};
struct ciou_tag : iou_type_tag
{
};

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

        std::vector<scalar_t> areas2(M);

        for (int b = 0; b < B; ++b) {
            std::transform(boxes2_ptr + b * M, boxes2_ptr + (b + 1) * M, areas2.begin(), box_area_op<scalar_t>);
            for (int n = 0; n < N; ++n) {
                const scalar_t area1 = box_area_op(boxes1_ptr[b * N + n]);
                for (int m = 0; m < M; ++m) {
                    const scalar_t intersection = box_intersection_area(boxes1_ptr[b * N + n], boxes2_ptr[b * M + m]);
                    const output_t union_area = area1 + areas2[m] - intersection;
                    if constexpr (std::is_same_v<IouType, iou_tag>) {
                        output_ptr[b * N * M + n * M + m] = intersection / union_area;
                    } else if constexpr (std::is_same_v<IouType, giou_tag>) {
                        XYXY<scalar_t> enclosing_box = min_enclosing_box(boxes1_ptr[b * N + n], boxes2_ptr[b * M + m]);
                        output_t enclosing_area = std::max(box_area_op(enclosing_box), static_cast<scalar_t>(0));
                        output_ptr[b * N * M + n * M + m] =
                            intersection / union_area - (enclosing_area - union_area) / enclosing_area;
                    }
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
    __shared__ T shared_boxes1_area[box_iou_block_size_y];
    __shared__ XYXY<T> shared_boxes2[box_iou_block_size_x];
    __shared__ T shared_boxes2_area[box_iou_block_size_x];

    unsigned int b = blockIdx.z;
    unsigned int n = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && m < M) {
        shared_boxes2[threadIdx.x] = boxes2[b * M + m];
        shared_boxes2_area[threadIdx.x] = box_area_op(shared_boxes2[threadIdx.x]);
    }
    unsigned int n_ = blockIdx.y * blockDim.y + threadIdx.x;
    if (threadIdx.y == 1 && threadIdx.x < blockDim.y && n_ < N) {
        shared_boxes1[threadIdx.x] = boxes1[b * N + n_];
        shared_boxes1_area[threadIdx.x] = box_area_op(shared_boxes1[threadIdx.x]);
    }
    __syncthreads();
    if (n >= N || m >= M) return;// Prevent out-of-bounds access

    const auto box2 = shared_boxes2[threadIdx.x];
    const auto box1 = shared_boxes1[threadIdx.y];
    const T intersection = box_intersection_area(box1, box2);
    const U union_area = shared_boxes1_area[threadIdx.y] + shared_boxes2_area[threadIdx.x] - intersection;
    if constexpr (std::is_same_v<IouType, iou_tag>) {
        output[b * N * M + n * M + m] = intersection / union_area;
    } else if constexpr (std::is_same_v<IouType, giou_tag>) {
        XYXY<T> enclosing_box = min_enclosing_box(box1, box2);
        U enclosing_area = std::max(box_area_op(enclosing_box), static_cast<T>(0));
        output[b * N * M + n * M + m] = intersection / union_area - (enclosing_area - union_area) / enclosing_area;
    }
}

template<typename T, typename IouType, typename U = std::conditional_t<std::is_integral_v<T>, float, T>>
__global__ void box_iou_simple_kernel(const XYXY<T> *__restrict__ boxes1,
    const XYXY<T> *__restrict__ boxes2,
    U *output,
    unsigned int N,
    unsigned int M)
{
    const unsigned int b = blockIdx.z;
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // Why is there no __divmod intrinsic????
    const unsigned int n = tid / M;
    const unsigned int m = tid % M;
    if (n >= N || m >= M) { return; }// Prevent out-of-bounds access

    const auto box1 = boxes1[b * N + n];
    const auto box2 = boxes2[b * M + m];
    const T intersection = box_intersection_area(box1, box2);
    const U union_area = box_area_op(box1) + box_area_op(box2) - intersection;
    if constexpr (std::is_same_v<IouType, iou_tag>) {
        output[b * N * M + n * M + m] = intersection / union_area;
    } else if constexpr (std::is_same_v<IouType, giou_tag>) {
        XYXY<T> enclosing_box = min_enclosing_box(box1, box2);
        U enclosing_area = std::max(box_area_op(enclosing_box), static_cast<T>(0));
        output[b * N * M + n * M + m] = intersection / union_area - (enclosing_area - union_area) / enclosing_area;
    }
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

        if (N < box_iou_block_size_x || M < box_iou_block_size_y) {
            int min_grid_size = 0;
            int block_size = 0;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, box_iou_simple_kernel<scalar_t, IouType>);
            block_size = std::min(static_cast<int>(M * N), block_size);
            const auto grid_dim = dim3(cuda::ceil_div(M * N, static_cast<unsigned int>(block_size)), 1, B);
            box_iou_simple_kernel<scalar_t, IouType>
                <<<grid_dim, block_size, 0, stream>>>(boxes1_ptr, boxes2_ptr, output_ptr, N, M);
        } else {
            auto block_dim = dim3(box_iou_block_size_x, box_iou_block_size_y);
            auto grid_dim = dim3(cuda::ceil_div(M, block_dim.x), cuda::ceil_div(N, block_dim.y), B);
            box_iou_kernel<scalar_t, IouType>
                <<<grid_dim, block_dim, 0, stream>>>(boxes1_ptr, boxes2_ptr, output_ptr, N, M);
        }
    });
}

auto regularize_for_iou(const torch::Tensor &boxes1, const torch::Tensor &boxes2, const torch::Tensor &output)
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
    if (opts.dtype() == torch::kInt32) {
        opts = opts.dtype(torch::kFloat32);// Ensure output is float for IoU
    }
    return torch::empty(output_shape, opts);
}

template<typename IouType> auto box_iou(const torch::Tensor &boxes1, const torch::Tensor &boxes2) -> torch::Tensor
{
    TORCH_CHECK(boxes1.is_contiguous() && boxes2.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(boxes1.size(-1) == 4 && boxes2.size(-1) == 4, "Input tensors must have shape (..., 4) for boxes");
    TORCH_CHECK(boxes1.ndimension() == boxes2.ndimension(),
        "Input tensors boxes1 and boxes2 must have the same number of dimensions");
    TORCH_CHECK(boxes1.ndimension() >= 2, "Input tensors boxes1 and boxes2 must have at least 2 dimensions");

    auto output = create_iou_output_tensor(boxes1, boxes2);

    // Regularize the shape to Batch x Nboxes x 4
    torch::Tensor boxes1_flat, boxes2_flat, output_flat;
    std::tie(boxes1_flat, boxes2_flat, output_flat) = regularize_for_iou(boxes1, boxes2, output);

    if (boxes1_flat.is_cuda()) {
        box_iou_gpu_impl<IouType>(boxes1_flat, boxes2_flat, output_flat);
    } else {
        box_iou_cpu_impl<IouType>(boxes1_flat, boxes2_flat, output_flat);
    }

    return output;
}

auto loss_inter_union(const torch::Tensor &boxes1, const torch::Tensor &boxes2)
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    TORCH_CHECK(boxes1.is_contiguous() && boxes2.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");

    torch::Tensor intersection = boxes1.new_empty({ boxes1.size(0) });
    torch::Tensor union_area = boxes1.new_empty({ boxes1.size(0) });

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes1.scalar_type(), "_loss_inter_union", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto intersection_ptr = static_cast<scalar_t *>(intersection.mutable_data_ptr());
        auto union_area_ptr = static_cast<scalar_t *>(union_area.mutable_data_ptr());

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                intersection_ptr[idx] = box_intersection_area(boxes1_ptr[idx], boxes2_ptr[idx]);
                union_area_ptr[idx] =
                    box_area_op(boxes1_ptr[idx]) + box_area_op(boxes2_ptr[idx]) - intersection_ptr[idx];
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                intersection_ptr[i] = box_intersection_area(boxes1_ptr[i], boxes2_ptr[i]);
                union_area_ptr[i] = box_area_op(boxes1_ptr[i]) + box_area_op(boxes2_ptr[i]) - intersection_ptr[i];
            }
        }
    });

    return { intersection, union_area };
}

template<typename T>
TFBO_HOST_DEVICE auto intersection_grad(const XYXY<T> &box1, const XYXY<T> &box2, const XYXY<T> &inter_box)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    XYXY<T> grad_box1, grad_box2;

    T inter_width = inter_box.x2 - inter_box.x1;
    T inter_height = inter_box.y2 - inter_box.y1;
    bool okay = inter_width > 0 && inter_height > 0;
    inter_width *= okay ? 1 : 0;
    inter_height *= okay ? 1 : 0;
    const auto subgrad = static_cast<T>(0.5);

    bool x1_gt = box1.x1 > box2.x1;
    bool x1_eq = box1.x1 == box2.x1;
    bool x1_lt = box1.x1 < box2.x1;
    grad_box1.x1 = -(x1_gt + subgrad * x1_eq) * inter_height;
    grad_box2.x1 = -(x1_lt + subgrad * x1_eq) * inter_height;

    bool y1_gt = box1.y1 > box2.y1;
    bool y1_eq = box1.y1 == box2.y1;
    bool y1_lt = box1.y1 < box2.y1;
    grad_box1.y1 = -(y1_gt + subgrad * y1_eq) * inter_width;
    grad_box2.y1 = -(y1_lt + subgrad * y1_eq) * inter_width;

    bool x2_gt = box1.x2 > box2.x2;
    bool x2_eq = box1.x2 == box2.x2;
    bool x2_lt = box1.x2 < box2.x2;
    grad_box1.x2 = (x2_lt + subgrad * x2_eq) * inter_height;
    grad_box2.x2 = (x2_gt + subgrad * x2_eq) * inter_height;

    bool y2_gt = box1.y2 > box2.y2;
    bool y2_eq = box1.y2 == box2.y2;
    bool y2_lt = box1.y2 < box2.y2;
    grad_box1.y2 = (y2_lt + subgrad * y2_eq) * inter_width;
    grad_box2.y2 = (y2_gt + subgrad * y2_eq) * inter_width;

    return { grad_box1, grad_box2 };
}


template<typename T>
TFBO_HOST_DEVICE auto inter_union_grad(T grad_inter, T grad_union, const XYXY<T> &box1, const XYXY<T> &box2)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    XYXY<T> inter_box = box_intersection(box1, box2);
    T inter_area = std::max(box_area_op(inter_box), static_cast<T>(0));
    T union_area = box_area_op(box1) + box_area_op(box2) - inter_area;

    auto [inter_grad_box1, inter_grad_box2] = intersection_grad(box1, box2, inter_box);
    auto area_grad_box1 = box_area_grad(box1);
    auto area_grad_box2 = box_area_grad(box2);

    // dUnion = dArea1 + dArea2 - dIntersection
    // grad = dUnion * gradUnion + dIntersection * gradInter
    // grad = (dArea - dIntersection) * gradUnion + dIntersection * gradInter
    // grad = dArea * gradUnion + (gradInter - gradUnion) * dIntersection
    T grad_inter_ = grad_inter - grad_union;

    XYXY<T> grad_box1;
    grad_box1.x1 = grad_inter_ * inter_grad_box1.x1 + grad_union * area_grad_box1.x1;
    grad_box1.y1 = grad_inter_ * inter_grad_box1.y1 + grad_union * area_grad_box1.y1;
    grad_box1.x2 = grad_inter_ * inter_grad_box1.x2 + grad_union * area_grad_box1.x2;
    grad_box1.y2 = grad_inter_ * inter_grad_box1.y2 + grad_union * area_grad_box1.y2;

    XYXY<T> grad_box2;
    grad_box2.x1 = grad_inter_ * inter_grad_box2.x1 + grad_union * area_grad_box2.x1;
    grad_box2.y1 = grad_inter_ * inter_grad_box2.y1 + grad_union * area_grad_box2.y1;
    grad_box2.x2 = grad_inter_ * inter_grad_box2.x2 + grad_union * area_grad_box2.x2;
    grad_box2.y2 = grad_inter_ * inter_grad_box2.y2 + grad_union * area_grad_box2.y2;

    return { grad_box1, grad_box2 };
}

auto loss_inter_union_backward(const torch::Tensor &grad_inter,
    const torch::Tensor &grad_union,
    const torch::Tensor &boxes1,
    const torch::Tensor &boxes2) -> std::tuple<torch::Tensor, torch::Tensor>
{
    TORCH_CHECK(
        grad_inter.is_contiguous() && grad_union.is_contiguous() && boxes1.is_contiguous() && boxes2.is_contiguous(),
        "Input tensors must be contiguous");
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");

    auto grad_boxes1 = torch::empty_like(boxes1);
    auto grad_boxes2 = torch::empty_like(boxes2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes1.scalar_type(), "_loss_inter_union_backward", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto grad_boxes1_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes1.mutable_data_ptr());
        auto grad_boxes2_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes2.mutable_data_ptr());
        const auto grad_inter_ptr = grad_inter.const_data_ptr<scalar_t>();
        const auto grad_union_ptr = grad_union.const_data_ptr<scalar_t>();

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                auto [grad_boxes1, grad_boxes2] =
                    inter_union_grad(grad_inter_ptr[idx], grad_union_ptr[idx], boxes1_ptr[idx], boxes2_ptr[idx]);
                grad_boxes1_ptr[idx] = grad_boxes1;
                grad_boxes2_ptr[idx] = grad_boxes2;
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                std::tie(grad_boxes1_ptr[i], grad_boxes2_ptr[i]) =
                    inter_union_grad(grad_inter_ptr[i], grad_union_ptr[i], boxes1_ptr[i], boxes2_ptr[i]);
            }
        }
    });

    return { grad_boxes1, grad_boxes2 };
}

template<typename T> TFBO_HOST_DEVICE auto giou_loss_fn(const XYXY<T> &box1, const XYXY<T> &box2, T eps) -> T
{
    auto intersection = box_intersection_area(box1, box2);
    auto union_area = box_area_op(box1) + box_area_op(box2) - intersection;
    XYXY<T> enclosing_box = min_enclosing_box(box1, box2);
    T enclosing_area = std::max(box_area_op(enclosing_box), static_cast<T>(0));
    T giou = intersection / union_area - (enclosing_area - union_area) / (enclosing_area + eps);
    return 1 - giou;
}

auto generalized_box_iou_loss(const torch::Tensor &boxes1, const torch::Tensor &boxes2, double eps) -> torch::Tensor
{
    TORCH_CHECK(boxes1.is_contiguous() && boxes2.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");

    auto giou_loss = boxes1.new_empty({ boxes1.size(0) });

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes1.scalar_type(), "generalized_box_iou_loss", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto giou_loss_ptr = giou_loss.mutable_data_ptr<scalar_t>();
        const auto eps_t = static_cast<scalar_t>(eps);

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                giou_loss_ptr[idx] = giou_loss_fn(boxes1_ptr[idx], boxes2_ptr[idx], eps_t);
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                giou_loss_ptr[i] = giou_loss_fn(boxes1_ptr[i], boxes2_ptr[i], eps_t);
            }
        }
    });

    return giou_loss;
}

template<typename T>
TFBO_HOST_DEVICE auto min_enclosing_box_grad(const XYXY<T> &box1, const XYXY<T> &box2, const XYXY<T> &enc_box)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    XYXY<T> box1_grad, box2_grad;
    T enc_w = enc_box.x2 - enc_box.x1;
    T enc_h = enc_box.y2 - enc_box.y1;
    const T subgrad = static_cast<T>(0.5);

    bool x1_lt = box1.x1 < box2.x1;
    bool x1_eq = box1.x1 == box2.x1;
    bool x1_gt = box1.x1 > box2.x1;
    box1_grad.x1 = -(x1_lt + subgrad * x1_eq) * enc_h;
    box2_grad.x1 = -(x1_gt + subgrad * x1_eq) * enc_h;

    bool y1_lt = box1.y1 < box2.y1;
    bool y1_eq = box1.y1 == box2.y1;
    bool y1_gt = box1.y1 > box2.y1;
    box1_grad.y1 = -(y1_lt + subgrad * y1_eq) * enc_w;
    box2_grad.y1 = -(y1_gt + subgrad * y1_eq) * enc_w;

    bool x2_lt = box1.x2 < box2.x2;
    bool x2_eq = box1.x2 == box2.x2;
    bool x2_gt = box1.x2 > box2.x2;
    box1_grad.x2 = (x2_gt + subgrad * x2_eq) * enc_h;
    box2_grad.x2 = (x2_lt + subgrad * x2_eq) * enc_h;

    bool y2_lt = box1.y2 < box2.y2;
    bool y2_eq = box1.y2 == box2.y2;
    bool y2_gt = box1.y2 > box2.y2;
    box1_grad.y2 = (y2_gt + subgrad * y2_eq) * enc_w;
    box2_grad.y2 = (y2_lt + subgrad * y2_eq) * enc_w;

    return { box1_grad, box2_grad };
}

template<typename T>
TFBO_HOST_DEVICE auto giou_grad(T grad_loss, const XYXY<T> &box1, const XYXY<T> &box2, T eps)
    -> std::tuple<XYXY<T>, XYXY<T>>
{
    T inter_area = box_intersection_area(box1, box2);
    T union_area = box_area_op(box1) + box_area_op(box2) - inter_area;
    XYXY<T> enclosing_box = min_enclosing_box(box1, box2);
    T enc_area = std::max(box_area_op(enclosing_box), static_cast<T>(0));

    T enc_area_eps = enc_area + eps;
    T union_area_eps = union_area + eps;

    T grad_enc_area = grad_loss * union_area / (enc_area_eps * enc_area_eps);
    T grad_inter = -grad_loss / union_area_eps;
    T grad_union = grad_loss * (inter_area / (union_area_eps * union_area_eps) - 1 / enc_area_eps);

    auto [grad_box1_enc, grad_box2_enc] = min_enclosing_box_grad(box1, box2, enclosing_box);
    auto [grad_box1, grad_box2] = inter_union_grad(grad_inter, grad_union, box1, box2);

    // Combine gradients with FMA
    grad_box1.x1 = fma(grad_box1_enc.x1, grad_enc_area, grad_box1.x1);
    grad_box1.y1 = fma(grad_box1_enc.y1, grad_enc_area, grad_box1.y1);
    grad_box1.x2 = fma(grad_box1_enc.x2, grad_enc_area, grad_box1.x2);
    grad_box1.y2 = fma(grad_box1_enc.y2, grad_enc_area, grad_box1.y2);

    grad_box2.x1 = fma(grad_box2_enc.x1, grad_enc_area, grad_box2.x1);
    grad_box2.y1 = fma(grad_box2_enc.y1, grad_enc_area, grad_box2.y1);
    grad_box2.x2 = fma(grad_box2_enc.x2, grad_enc_area, grad_box2.x2);
    grad_box2.y2 = fma(grad_box2_enc.y2, grad_enc_area, grad_box2.y2);

    return { grad_box1, grad_box2 };
}

auto generalized_box_iou_loss_backward(const torch::Tensor &grad,
    const torch::Tensor &boxes1,
    const torch::Tensor &boxes2,
    double eps) -> std::tuple<torch::Tensor, torch::Tensor>
{
    TORCH_CHECK(
        grad.is_contiguous() && boxes1.is_contiguous() && boxes2.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");

    auto grad_boxes1 = torch::empty_like(boxes1);
    auto grad_boxes2 = torch::empty_like(boxes2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes1.scalar_type(), "generalized_box_iou_loss_backward", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto grad_boxes1_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes1.mutable_data_ptr());
        auto grad_boxes2_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes2.mutable_data_ptr());
        const auto grad_ptr = grad.const_data_ptr<scalar_t>();
        auto eps_t = static_cast<scalar_t>(eps);
        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(unsigned int idx) {
                auto [grad_box1, grad_box2] = giou_grad(grad_ptr[idx], boxes1_ptr[idx], boxes2_ptr[idx], eps_t);
                grad_boxes1_ptr[idx] = grad_box1;
                grad_boxes2_ptr[idx] = grad_box2;
            };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            for (std::size_t i = 0; i < num_boxes; ++i) {
                std::tie(grad_boxes1_ptr[i], grad_boxes2_ptr[i]) =
                    giou_grad(grad_ptr[i], boxes1_ptr[i], boxes2_ptr[i], eps_t);
            }
        }
    });

    return { grad_boxes1, grad_boxes2 };
}

TORCH_LIBRARY_IMPL(box_ops, CPU, m)
{
    m.impl("box_area", &box_area);
    m.impl("box_iou", &box_iou<iou_tag>);
    m.impl("generalized_box_iou", &box_iou<giou_tag>);
    m.impl("box_area_backward", &box_area_backward);
    m.impl("_loss_inter_union", &loss_inter_union);
    m.impl("_loss_inter_union_backward", &loss_inter_union_backward);
    m.impl("generalized_box_iou_loss", &generalized_box_iou_loss);
    m.impl("generalized_box_iou_loss_backward", &generalized_box_iou_loss_backward);
}

TORCH_LIBRARY_IMPL(box_ops, CUDA, m)
{
    m.impl("box_area", &box_area);
    m.impl("box_iou", &box_iou<iou_tag>);
    m.impl("generalized_box_iou", &box_iou<giou_tag>);
    m.impl("box_area_backward", &box_area_backward);
    m.impl("_loss_inter_union", &loss_inter_union);
    m.impl("_loss_inter_union_backward", &loss_inter_union_backward);
    m.impl("generalized_box_iou_loss", &generalized_box_iou_loss);
    m.impl("generalized_box_iou_loss_backward", &generalized_box_iou_loss_backward);
}
