#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "boxes.cuh"

#ifdef __CUDACC__
#define FN_QUAL __host__ __device__
#else
#define FN_QUAL
#endif

auto iou_cpu(const torch::Tensor &boxes1, const torch::Tensor &boxes2) -> torch::Tensor
{
    // This is a placeholder implementation; replace with actual logic
    return torch::empty({ boxes1.size(0), boxes2.size(0) }, boxes1.options());
}

template<typename T> __global__ void box_area_kernel(const XYXY<T> *boxes, T *output, int64_t num_boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_boxes) {
        const auto &box = boxes[idx];
        output[idx] = (box.x2 - box.x1) * (box.y2 - box.y1);
    }
}

auto box_area(const torch::Tensor &boxes) -> torch::Tensor
{
    TORCH_CHECK(boxes.is_contiguous(), "Input tensor must be contiguous");
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
            auto stream = at::cuda::getCurrentCUDAStream();
            const auto num_threads = 256;
            box_area_kernel<scalar_t>
                <<<cuda::ceil_div(num_boxes, num_threads), num_threads, 0, stream>>>(boxes_ptr, areas_ptr, num_boxes);
        } else {
            std::transform(boxes_ptr, boxes_ptr + num_boxes, areas_ptr, [](const XYXY<scalar_t> &box) {
                return (box.x2 - box.x1) * (box.y2 - box.y1);
            });
        }
    });

    return output;
}

template<typename T> auto FN_QUAL box_area_backward_(T grad, XYXY<T> box) -> XYXY<T>
{
    XYXY<T> grad_box;
    grad_box.x1 = grad * (box.y1 - box.y2);
    grad_box.y1 = grad * (box.x1 - box.x2);
    grad_box.x2 = grad * (box.y2 - box.y1);
    grad_box.y2 = grad * (box.x2 - box.x1);
    return grad_box;
}

template<typename T>
__global__ void box_area_backward_kernel(const T *__restrict__ grad,
    const XYXY<T> *__restrict__ boxes,
    XYXY<T> *input_grad,
    int64_t num_boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_boxes) { input_grad[idx] = box_area_backward_(grad[idx], boxes[idx]); }
}

auto box_area_backward(const torch::Tensor &grad, const torch::Tensor &boxes) -> torch::Tensor
{
    TORCH_CHECK(grad.is_contiguous(), "Gradient tensor must be contiguous");
    TORCH_CHECK(boxes.is_contiguous(), "Boxes tensor must be contiguous");
    TORCH_CHECK(boxes.size(-1) == 4, "Boxes tensor must have shape (..., 4)");

    auto input_grad = torch::empty_like(boxes);
    TFBO_DISPATCH_BOX_TYPES(boxes.scalar_type(), "box_area_backward", [&] {
        auto grad_ptr = grad.const_data_ptr<scalar_t>();
        auto boxes_ptr = static_cast<const XYXY<scalar_t> *>(boxes.const_data_ptr());
        auto input_grad_ptr = static_cast<XYXY<scalar_t> *>(input_grad.mutable_data_ptr());

        if (boxes.is_cuda()) {
            auto stream = at::cuda::getCurrentCUDAStream();
            const auto num_threads = 256;
            box_area_backward_kernel<scalar_t><<<cuda::ceil_div(grad.numel(), num_threads), num_threads, 0, stream>>>(
                grad_ptr, boxes_ptr, input_grad_ptr, grad.numel());
        } else {
            std::transform(grad_ptr, grad_ptr + grad.numel(), boxes_ptr, input_grad_ptr, box_area_backward_<scalar_t>);
        }
    });

    return input_grad;
}

TORCH_LIBRARY_IMPL(box_ops, CPU, m)
{
    m.impl("iou", &iou_cpu);
    m.impl("box_area", &box_area);
    m.impl("box_area_backward", &box_area_backward);
}

TORCH_LIBRARY_IMPL(box_ops, CUDA, m)
{
    m.impl("box_area", &box_area);
    m.impl("box_area_backward", &box_area_backward);
}
