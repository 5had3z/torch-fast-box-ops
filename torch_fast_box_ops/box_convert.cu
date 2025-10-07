#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "boxes.cuh"
#include "kernel.cuh"


template<typename T> TFBO_HOST_DEVICE CXCYWH<T> convert_box(const XYXY<T> box, xyxy_tag, cxcywh_tag)
{
    CXCYWH<T> result;
    result.cx = (box.x1 + box.x2) * 0.5f;
    result.cy = (box.y1 + box.y2) * 0.5f;
    result.w = box.x2 - box.x1;
    result.h = box.y2 - box.y1;
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYXY<T> convert_box_grad(const CXCYWH<T> box, xyxy_tag, cxcywh_tag)
{
    XYXY<T> result;
    result.x1 = 0.5f * box.cx - box.w;
    result.y1 = 0.5f * box.cy - box.h;
    result.x2 = 0.5f * box.cx + box.w;
    result.y2 = 0.5f * box.cy + box.h;
    return result;
}

template<typename T> TFBO_HOST_DEVICE CXCYWH<T> convert_box(const XYWH<T> box, xywh_tag, cxcywh_tag)
{
    CXCYWH<T> result;
    result.cx = box.x + box.w * 0.5f;
    result.cy = box.y + box.h * 0.5f;
    result.w = box.w;
    result.h = box.h;
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYWH<T> convert_box_grad(const CXCYWH<T> box, xywh_tag, cxcywh_tag)
{
    XYWH<T> result;
    result.x = box.cx;
    result.y = box.cy;
    result.w = box.w + 0.5f * box.cx;
    result.h = box.h + 0.5f * box.cy;
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYXY<T> convert_box(const CXCYWH<T> box, cxcywh_tag, xyxy_tag)
{
    XYXY<T> result;
    result.x1 = box.cx - box.w * 0.5f;
    result.y1 = box.cy - box.h * 0.5f;
    result.x2 = box.cx + box.w * 0.5f;
    result.y2 = box.cy + box.h * 0.5f;
    return result;
}

template<typename T> TFBO_HOST_DEVICE CXCYWH<T> convert_box_grad(const XYXY<T> box, cxcywh_tag, xyxy_tag)
{
    CXCYWH<T> result;
    result.cx = box.x1 + box.x2;
    result.cy = box.y1 + box.y2;
    result.w = 0.5f * (box.x2 - box.x1);
    result.h = 0.5f * (box.y2 - box.y1);
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYXY<T> convert_box(const XYWH<T> box, xywh_tag, xyxy_tag)
{
    XYXY<T> result;
    result.x1 = box.x;
    result.y1 = box.y;
    result.x2 = box.x + box.w;
    result.y2 = box.y + box.h;
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYWH<T> convert_box_grad(const XYXY<T> box, xywh_tag, xyxy_tag)
{
    XYWH<T> result;
    result.x = box.x1 + box.x2;
    result.y = box.y1 + box.y2;
    result.w = box.x2;
    result.h = box.y2;
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYWH<T> convert_box(const CXCYWH<T> box, cxcywh_tag, xywh_tag)
{
    XYWH<T> result;
    result.x = box.cx - box.w * 0.5f;
    result.y = box.cy - box.h * 0.5f;
    result.w = box.w;
    result.h = box.h;
    return result;
}

template<typename T> TFBO_HOST_DEVICE CXCYWH<T> convert_box_grad(const XYWH<T> box, cxcywh_tag, xywh_tag)
{
    CXCYWH<T> result;
    result.cx = box.x;
    result.cy = box.y;
    result.w = box.w - 0.5f * box.x;
    result.h = box.h - 0.5f * box.y;
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYWH<T> convert_box(const XYXY<T> box, xyxy_tag, xywh_tag)
{
    XYWH<T> result;
    result.x = box.x1;
    result.y = box.y1;
    result.w = box.x2 - box.x1;
    result.h = box.y2 - box.y1;
    return result;
}

template<typename T> TFBO_HOST_DEVICE XYXY<T> convert_box_grad(const XYWH<T> box, xyxy_tag, xywh_tag)
{
    XYXY<T> result;
    result.x1 = box.x - box.w;
    result.y1 = box.y - box.h;
    result.x2 = box.x + box.w;
    result.y2 = box.y + box.h;
    return result;
}

#undef TFBO_HOST_DEVICE


template<typename T> using BoxConverter = std::function<void(const void *, void *, size_t, bool, cudaStream_t)>;

struct ConversionKey
{
    std::string in_fmt;
    std::string out_fmt;

    bool operator<(const ConversionKey &other) const
    {
        if (in_fmt != other.in_fmt) return in_fmt < other.in_fmt;
        return out_fmt < other.out_fmt;
    }
};

template<typename T, template<typename> typename InBox, template<typename> typename OutBox>
auto make_forward_converter() -> BoxConverter<T>
{
    return [](const void *input, void *output, size_t n, bool is_cuda, cudaStream_t stream = nullptr) {
        const auto input_data = static_cast<const InBox<T> *>(input);
        auto output_data = static_cast<OutBox<T> *>(output);

        using InBoxType = typename box_tag_map<InBox>::type;
        using OutBoxType = typename box_tag_map<OutBox>::type;

        if (is_cuda) {
            auto kernel = [=] __device__(unsigned int idx) {
                output_data[idx] = convert_box(input_data[idx], InBoxType{}, OutBoxType{});
            };
            launch_elementwise_kernel(kernel, n, stream);
        } else {
            std::transform(input_data, input_data + n, output_data, [](const InBox<T> in) {
                return convert_box(in, InBoxType{}, OutBoxType{});
            });
        }
    };
}

auto box_convert_forward(const torch::Tensor &input, const std::string &in_fmt, const std::string &out_fmt)
    -> torch::Tensor
{
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.size(-1) == 4, "Input tensor must have shape (..., 4) for boxes");

    if (in_fmt == out_fmt) {
        return input.clone();// No conversion needed, just return a copy
    }

    auto output = torch::empty_like(input);
    auto numBoxes = input.numel() >> 2;// Assuming input is of shape (..., 4) for boxes
    cudaStream_t stream = nullptr;
    const auto is_cuda = input.is_cuda();
    if (is_cuda) { stream = at::cuda::getCurrentCUDAStream(); }

    TFBO_DISPATCH_BOX_TYPES(input.scalar_type(), "box_convert_forward", [&] {
        static const std::map<ConversionKey, BoxConverter<scalar_t>> converters = {
            { { "xyxy", "cxcywh" }, make_forward_converter<scalar_t, XYXY, CXCYWH>() },
            { { "xywh", "cxcywh" }, make_forward_converter<scalar_t, XYWH, CXCYWH>() },
            { { "cxcywh", "xywh" }, make_forward_converter<scalar_t, CXCYWH, XYWH>() },
            { { "xyxy", "xywh" }, make_forward_converter<scalar_t, XYXY, XYWH>() },
            { { "cxcywh", "xyxy" }, make_forward_converter<scalar_t, CXCYWH, XYXY>() },
            { { "xywh", "xyxy" }, make_forward_converter<scalar_t, XYWH, XYXY>() }
        };

        ConversionKey key{ in_fmt, out_fmt };
        auto it = converters.find(key);
        if (it == converters.end()) { throw std::invalid_argument("Unsupported format conversion"); }

        it->second(input.const_data_ptr(), output.mutable_data_ptr(), numBoxes, is_cuda, stream);
    });

    return output;
}

template<typename T, template<typename> typename InBox, template<typename> typename OutBox>
auto make_backward_converter() -> BoxConverter<T>
{
    return [](const void *input, void *output, size_t n, bool is_cuda, cudaStream_t stream = nullptr) {
        const auto output_grad = static_cast<const OutBox<T> *>(input);
        auto input_grad = static_cast<InBox<T> *>(output);

        using InBoxType = typename box_tag_map<InBox>::type;
        using OutBoxType = typename box_tag_map<OutBox>::type;

        if (is_cuda) {
            auto kernel = [=] __device__(unsigned int idx) {
                input_grad[idx] = convert_box_grad(output_grad[idx], InBoxType{}, OutBoxType{});
            };
            launch_elementwise_kernel(kernel, n, stream);
        } else {
            std::transform(output_grad, output_grad + n, input_grad, [](const OutBox<T> in) {
                return convert_box_grad(in, InBoxType{}, OutBoxType{});
            });
        }
    };
}

auto box_convert_backward(const torch::Tensor &out_grad, const std::string &in_fmt, const std::string &out_fmt)
    -> torch::Tensor
{
    TORCH_CHECK(out_grad.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(out_grad.size(-1) == 4, "Input tensor must have shape (..., 4) for boxes");

    if (in_fmt == out_fmt) {
        return out_grad.clone();// No conversion needed, just return a copy
    }

    auto output = torch::empty_like(out_grad);
    auto numBoxes = out_grad.numel() >> 2;// Assuming input is of shape (..., 4) for boxes
    cudaStream_t stream = nullptr;
    const auto is_cuda = out_grad.is_cuda();
    if (is_cuda) { stream = at::cuda::getCurrentCUDAStream(); }

    TFBO_DISPATCH_BOX_TYPES(out_grad.scalar_type(), "box_convert_backward", [&] {
        static const std::map<ConversionKey, BoxConverter<scalar_t>> converters = {
            { { "cxcywh", "xyxy" }, make_backward_converter<scalar_t, CXCYWH, XYXY>() },
            { { "xywh", "xyxy" }, make_backward_converter<scalar_t, XYWH, XYXY>() },
            { { "cxcywh", "xywh" }, make_backward_converter<scalar_t, CXCYWH, XYWH>() },
            { { "xyxy", "cxcywh" }, make_backward_converter<scalar_t, XYXY, CXCYWH>() },
            { { "xywh", "cxcywh" }, make_backward_converter<scalar_t, XYWH, CXCYWH>() },
            { { "xyxy", "xywh" }, make_backward_converter<scalar_t, XYXY, XYWH>() }
        };

        ConversionKey key{ in_fmt, out_fmt };
        auto it = converters.find(key);
        if (it == converters.end()) { throw std::invalid_argument("Unsupported format conversion"); }

        it->second(out_grad.const_data_ptr(), output.mutable_data_ptr(), numBoxes, is_cuda, stream);
    });

    return output;
}

TORCH_LIBRARY_IMPL(box_ops, CPU, m) { m.impl("box_convert", &box_convert_forward); }
TORCH_LIBRARY_IMPL(box_ops, CUDA, m) { m.impl("box_convert", &box_convert_forward); }

TORCH_LIBRARY_IMPL(box_ops, CPU, m) { m.impl("box_convert_backward", &box_convert_backward); }
TORCH_LIBRARY_IMPL(box_ops, CUDA, m) { m.impl("box_convert_backward", &box_convert_backward); }
