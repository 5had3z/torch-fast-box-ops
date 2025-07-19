#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "boxes.cuh"

#ifdef __CUDACC__
#define FN_QUAL __host__ __device__
#else
#define FN_QUAL
#endif

template<typename T> FN_QUAL auto box_area_op(const XYXY<T> &box) -> T { return (box.x2 - box.x1) * (box.y2 - box.y1); }

template<typename T> __global__ void box_area_kernel(const XYXY<T> *boxes, T *output, int64_t num_boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_boxes) { output[idx] = box_area_op(boxes[idx]); }
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
            std::transform(boxes_ptr, boxes_ptr + num_boxes, areas_ptr, box_area_op<scalar_t>);
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

template<typename T> FN_QUAL auto box_intersection(const XYXY<T> &box1, const XYXY<T> &box2) -> T
{
    T inter_x1 = std::max(box1.x1, box2.x1);
    T inter_y1 = std::max(box1.y1, box2.y1);
    T inter_x2 = std::min(box1.x2, box2.x2);
    T inter_y2 = std::min(box1.y2, box2.y2);
    T inter_area = std::max(inter_x2 - inter_x1, static_cast<T>(0)) * std::max(inter_y2 - inter_y1, static_cast<T>(0));
    return inter_area;
}

void box_iou_cpu_impl(const torch::Tensor &boxes1, const torch::Tensor &boxes2, torch::Tensor &output)
{
    TFBO_DISPATCH_BOX_TYPES(boxes1.scalar_type(), "box_iou", [&] {
        const auto B = boxes1.size(0);
        const auto N = boxes1.size(1);
        const auto M = boxes2.size(1);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto output_ptr =
            static_cast<std::conditional_t<std::is_integral_v<scalar_t>, float, scalar_t> *>(output.mutable_data_ptr());

        std::vector<scalar_t> areas2(M);

        for (int b = 0; b < B; ++b) {
            std::transform(boxes2_ptr + b * M, boxes2_ptr + (b + 1) * M, areas2.begin(), box_area_op<scalar_t>);
            for (int n = 0; n < N; ++n) {
                const auto area1 = box_area_op(boxes1_ptr[b * N + n]);
                for (int m = 0; m < M; ++m) {
                    const auto intersection = box_intersection(boxes1_ptr[b * N + n], boxes2_ptr[b * M + m]);
                    const auto union_area = area1 + areas2[m] - intersection;
                    if (std::is_integral_v<scalar_t>) {
                        output_ptr[b * N * M + n * M + m] =
                            static_cast<float>(intersection) / static_cast<float>(union_area);
                    } else {
                        output_ptr[b * N * M + n * M + m] = intersection / union_area;
                    }
                }
            }
        }
    });
}

void box_iou_gpu_impl(const torch::Tensor &boxes1, const torch::Tensor &boxes2, torch::Tensor &output) {}

auto box_iou(const torch::Tensor &boxes1, const torch::Tensor &boxes2) -> torch::Tensor
{
    TORCH_CHECK(boxes1.is_contiguous(), "Input tensor boxes1 must be contiguous");
    TORCH_CHECK(boxes2.is_contiguous(), "Input tensor boxes2 must be contiguous");
    TORCH_CHECK(boxes1.size(-1) == 4, "Input tensor boxes1 must have shape (..., 4) for boxes");
    TORCH_CHECK(boxes2.size(-1) == 4, "Input tensor boxes2 must have shape (..., 4) for boxes");
    TORCH_CHECK(boxes1.ndimension() == boxes2.ndimension(),
        "Input tensors boxes1 and boxes2 must have the same number of dimensions");
    TORCH_CHECK(boxes1.ndimension() >= 2, "Input tensors boxes1 and boxes2 must have at least 2 dimensions");

    auto output_shape = boxes1.sizes().vec();
    output_shape.back() = boxes2.size(-2);// Replace '4' with the number of boxes in boxes2

    auto opts = boxes1.options();
    if (opts.dtype() == torch::kInt32) {
        opts = opts.dtype(torch::kFloat32);// Ensure output is float for IoU
    }
    auto output = torch::empty(output_shape, opts);

    // Regularize the shape to Batch x Nboxes x 4
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

    if (boxes1_flat.is_cuda()) {
        box_iou_gpu_impl(boxes1_flat, boxes2_flat, output_flat);
    } else {
        box_iou_cpu_impl(boxes1_flat, boxes2_flat, output_flat);
    }

    return output;
}

TORCH_LIBRARY_IMPL(box_ops, CPU, m)
{
    m.impl("box_iou", &box_iou);
    m.impl("box_area", &box_area);
    m.impl("box_area_backward", &box_area_backward);
}

TORCH_LIBRARY_IMPL(box_ops, CUDA, m)
{
    m.impl("box_iou", &box_iou);
    m.impl("box_area", &box_area);
    m.impl("box_area_backward", &box_area_backward);
}
