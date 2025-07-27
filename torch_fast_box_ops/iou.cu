#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "boxes.cuh"
#include "kernel.cuh"

#ifdef __CUDACC__
#define FN_QUAL __host__ __device__
#else
#define FN_QUAL
#endif

template<typename T> FN_QUAL auto box_area_op(const XYXY<T> &box) -> T { return (box.x2 - box.x1) * (box.y2 - box.y1); }

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
            auto kernel = [=] __device__(int idx) { areas_ptr[idx] = box_area_op(boxes_ptr[idx]); };
            launch_elementwise_kernel(kernel, num_boxes, at::cuda::getCurrentCUDAStream());
        } else {
            std::transform(boxes_ptr, boxes_ptr + num_boxes, areas_ptr, box_area_op<scalar_t>);
        }
    });

    return output;
}

template<typename T> auto FN_QUAL box_area_grad(XYXY<T> box) -> XYXY<T>
{
    XYXY<T> grad_box;
    grad_box.x1 = box.y1 - box.y2;
    grad_box.y1 = box.x1 - box.x2;
    grad_box.x2 = box.y2 - box.y1;
    grad_box.y2 = box.x2 - box.x1;
    return grad_box;
}

template<typename T> auto FN_QUAL box_area_backward_(T grad, XYXY<T> box) -> XYXY<T>
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
    TORCH_CHECK(grad.is_contiguous(), "Gradient tensor must be contiguous");
    TORCH_CHECK(boxes.is_contiguous(), "Boxes tensor must be contiguous");
    TORCH_CHECK(boxes.size(-1) == 4, "Boxes tensor must have shape (..., 4)");

    auto input_grad = torch::empty_like(boxes);
    TFBO_DISPATCH_BOX_TYPES(boxes.scalar_type(), "box_area_backward", [&] {
        auto grad_ptr = grad.const_data_ptr<scalar_t>();
        auto boxes_ptr = static_cast<const XYXY<scalar_t> *>(boxes.const_data_ptr());
        auto input_grad_ptr = static_cast<XYXY<scalar_t> *>(input_grad.mutable_data_ptr());

        if (boxes.is_cuda()) {
            auto kernel = [=] __device__(
                              int idx) { input_grad_ptr[idx] = box_area_backward_(grad_ptr[idx], boxes_ptr[idx]); };
            launch_elementwise_kernel(kernel, grad.numel(), at::cuda::getCurrentCUDAStream());
        } else {
            std::transform(grad_ptr, grad_ptr + grad.numel(), boxes_ptr, input_grad_ptr, box_area_backward_<scalar_t>);
        }
    });

    return input_grad;
}

template<typename T> FN_QUAL auto box_intersection(const XYXY<T> &box1, const XYXY<T> &box2) -> XYXY<T>
{
    XYXY<T> inter_box;
    inter_box.x1 = std::max(box1.x1, box2.x1);
    inter_box.y1 = std::max(box1.y1, box2.y1);
    inter_box.x2 = std::min(box1.x2, box2.x2);
    inter_box.y2 = std::min(box1.y2, box2.y2);
    return inter_box;
}


template<typename T> FN_QUAL auto box_intersection_area(const XYXY<T> &box1, const XYXY<T> &box2) -> T
{
    auto inter_box = box_intersection(box1, box2);
    T inter_area = std::max(inter_box.x2 - inter_box.x1, static_cast<T>(0))
                   * std::max(inter_box.y2 - inter_box.y1, static_cast<T>(0));
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
                    const auto intersection = box_intersection_area(boxes1_ptr[b * N + n], boxes2_ptr[b * M + m]);
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

template<typename T, typename U = std::conditional_t<std::is_integral_v<T>, float, T>>
__global__ void
    box_iou_kernel(const XYXY<T> *__restrict__ boxes1, const XYXY<T> *__restrict__ boxes2, U *output, int N, int M)
{
    int b = blockIdx.x;
    for (int n = threadIdx.y; n < N; n += blockDim.y) {
        const auto &box1 = boxes1[b * N + n];
        const auto area1 = box_area_op(box1);
        for (int m = threadIdx.x; m < M; m += blockDim.x) {
            const auto &box2 = boxes2[b * M + m];
            auto intersection = static_cast<U>(box_intersection_area(box1, box2));
            auto union_area = static_cast<U>(area1 + box_area_op(box2) - intersection);
            output[b * N * M + n * M + m] = intersection / union_area;
        }
    }
}

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

        auto block_dim = dim3(32, std::min(32u, N));
        auto grid_dim = dim3(B);
        box_iou_kernel<<<grid_dim, block_dim, 0, stream>>>(boxes1_ptr, boxes2_ptr, output_ptr, N, M);
    });
}

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

auto loss_inter_union(const torch::Tensor &boxes1, const torch::Tensor &boxes2)
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    TORCH_CHECK(boxes1.is_contiguous() && boxes2.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(boxes1.sizes() == boxes2.sizes(), "Input tensors boxes1 and boxes2 must have the same shape");
    TORCH_CHECK(boxes1.ndimension() == 2 && boxes1.size(-1) == 4, "Input tensors must have shape (N, 4)");

    torch::Tensor intersection = boxes1.new_empty({ boxes1.size(0) });
    torch::Tensor union_area = boxes1.new_empty({ boxes1.size(0) });

    TFBO_DISPATCH_BOX_TYPES(boxes1.scalar_type(), "_loss_inter_union", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto intersection_ptr = static_cast<scalar_t *>(intersection.mutable_data_ptr());
        auto union_area_ptr = static_cast<scalar_t *>(union_area.mutable_data_ptr());

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(int idx) {
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
FN_QUAL auto intersection_grad(const XYXY<T> &box1, const XYXY<T> &box2, const XYXY<T> &inter_box)
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
    grad_box1.x1 = -(x1_gt + subgrad * x1_eq) * inter_height;
    grad_box2.x1 = -(!x1_gt + subgrad * x1_eq) * inter_height;

    bool y1_gt = box1.y1 > box2.y1;
    bool y1_eq = box1.y1 == box2.y1;
    grad_box1.y1 = -(y1_gt + subgrad * y1_eq) * inter_width;
    grad_box2.y1 = -(!y1_gt + subgrad * y1_eq) * inter_width;

    bool x2_gt = box1.x2 > box2.x2;
    bool x2_eq = box1.x2 == box2.x2;
    grad_box1.x2 = (!x2_gt + subgrad * x2_eq) * inter_height;
    grad_box2.x2 = (x2_gt + subgrad * x2_eq) * inter_height;

    bool y2_gt = box1.y2 > box2.y2;
    bool y2_eq = box1.y2 == box2.y2;
    grad_box1.y2 = (!y2_gt + subgrad * y2_eq) * inter_width;
    grad_box2.y2 = (y2_gt + subgrad * y2_eq) * inter_width;

    return { grad_box1, grad_box2 };
}


template<typename T>
FN_QUAL auto inter_union_grad(T grad_inter, T grad_union, const XYXY<T> &box1, const XYXY<T> &box2)
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

    TFBO_DISPATCH_BOX_TYPES(boxes1.scalar_type(), "_loss_inter_union_backward", [&] {
        const auto num_boxes = boxes1.size(0);
        const auto boxes1_ptr = static_cast<const XYXY<scalar_t> *>(boxes1.const_data_ptr());
        const auto boxes2_ptr = static_cast<const XYXY<scalar_t> *>(boxes2.const_data_ptr());
        auto grad_boxes1_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes1.mutable_data_ptr());
        auto grad_boxes2_ptr = static_cast<XYXY<scalar_t> *>(grad_boxes2.mutable_data_ptr());
        const auto grad_inter_ptr = grad_inter.const_data_ptr<scalar_t>();
        const auto grad_union_ptr = grad_union.const_data_ptr<scalar_t>();

        if (boxes1.is_cuda()) {
            auto kernel = [=] __device__(int idx) {
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

TORCH_LIBRARY_IMPL(box_ops, CPU, m)
{
    m.impl("box_iou", &box_iou);
    m.impl("box_area", &box_area);
    m.impl("box_area_backward", &box_area_backward);
    m.impl("_loss_inter_union", &loss_inter_union);
    m.impl("_loss_inter_union_backward", &loss_inter_union_backward);
}

TORCH_LIBRARY_IMPL(box_ops, CUDA, m)
{
    m.impl("box_iou", &box_iou);
    m.impl("box_area", &box_area);
    m.impl("box_area_backward", &box_area_backward);
    m.impl("_loss_inter_union", &loss_inter_union);
    m.impl("_loss_inter_union_backward", &loss_inter_union_backward);
}
