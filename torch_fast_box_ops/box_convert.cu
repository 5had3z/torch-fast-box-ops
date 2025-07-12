#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda/cmath>

#include "boxes.cuh"


template<typename T, template<typename> typename InBox, template<typename> typename OutBox>
__global__ void box_conversion_kernel(const InBox<T> *input, OutBox<T> *output, size_t n)
{
    using InBoxType = typename box_tag_map<InBox>::type;
    using OutBoxType = typename box_tag_map<OutBox>::type;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        output[i] = convert_box(input[i], InBoxType{}, OutBoxType{});
    }
}

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

template<typename T, template<typename> typename InBox, template<typename> typename OutBox> auto make_converter()
{
    return [](const void *input, void *output, size_t n, bool is_cuda, cudaStream_t stream = nullptr) {
        const auto input_data = static_cast<const InBox<T> *>(input);
        auto output_data = static_cast<OutBox<T> *>(output);

        using InBoxType = typename box_tag_map<InBox>::type;
        using OutBoxType = typename box_tag_map<OutBox>::type;

        if (is_cuda) {
            box_conversion_kernel<T, InBox, OutBox><<<cuda::ceil_div(n, 256), 256, 0>>>(input_data, output_data, n);
        } else {
            std::transform(input_data, input_data + n, output_data, [](const InBox<T> in) {
                return convert_box(in, InBoxType{}, OutBoxType{});
            });
        }
    };
}

auto box_convert_impl(const torch::Tensor &input, const std::string &in_fmt, const std::string &out_fmt)
    -> torch::Tensor
{
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.size(-1) == 4, "Input tensor must have shape (..., 4) for boxes");

    if (in_fmt == out_fmt) {
        return input.clone();// No conversion needed, just return a copy
    }

    const auto is_cuda = input.is_cuda();
    auto output = torch::empty_like(input);
    auto numBoxes = input.numel() >> 2;// Assuming input is of shape (..., 4) for boxes
    cudaStream_t stream = nullptr;
    if (is_cuda) { stream = at::cuda::getCurrentCUDAStream(); }

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_impl", [&] {
        static const std::map<ConversionKey, BoxConverter<scalar_t>> converters = {
            { { "xyxy", "cxcywh" }, make_converter<scalar_t, XYXY, CXCYWH>() },
            { { "xywh", "cxcywh" }, make_converter<scalar_t, XYWH, CXCYWH>() },
            { { "cxcywh", "xywh" }, make_converter<scalar_t, CXCYWH, XYWH>() },
            { { "xyxy", "xywh" }, make_converter<scalar_t, XYXY, XYWH>() },
            { { "cxcywh", "xyxy" }, make_converter<scalar_t, CXCYWH, XYXY>() },
            { { "xywh", "xyxy" }, make_converter<scalar_t, XYWH, XYXY>() }
        };

        ConversionKey key{ in_fmt, out_fmt };
        auto it = converters.find(key);
        if (it == converters.end()) { throw std::invalid_argument("Unsupported format conversion"); }

        it->second(input.data_ptr(), output.data_ptr(), numBoxes, is_cuda, stream);
    });

    return output;
}

TORCH_LIBRARY_IMPL(box_ops, CPU, m) { m.impl("box_convert", &box_convert_impl); }
TORCH_LIBRARY_IMPL(box_ops, CUDA, m) { m.impl("box_convert", &box_convert_impl); }
