#include <torch/extension.h>

#include "boxes.hpp"

#include <algorithm>


template<typename T> using BoxConverter = std::function<void(const void *, void *, size_t)>;

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
    return [](const void *input, void *output, size_t n) {
        const auto input_data = static_cast<const InBox<T> *>(input);
        auto output_data = static_cast<OutBox<T> *>(output);

        using InBoxType = typename box_tag_map<InBox>::type;
        using OutBoxType = typename box_tag_map<OutBox>::type;

        std::transform(input_data, input_data + n, output_data, [](const InBox<T> in) {
            return convert_box(in, InBoxType{}, OutBoxType{});
        });
    };
}

auto box_convert_cpu(const torch::Tensor &input, const std::string &in_fmt, const std::string &out_fmt) -> torch::Tensor
{
    if (in_fmt == out_fmt) {
        return input.clone();// No conversion needed, just return a copy
    }

    auto output = torch::empty_like(input);
    auto numBoxes = input.numel() >> 2;// Assuming input is of shape (..., 4) for boxes

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_cpu", [&] {
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

        it->second(input.data_ptr(), output.data_ptr(), numBoxes);
    });

    return output;
}

TORCH_LIBRARY_IMPL(box_ops, CPU, m) { m.impl("box_convert", &box_convert_cpu); }
