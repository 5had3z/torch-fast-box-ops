#include <torch/extension.h>

#include "common.hpp"

#include <algorithm>

TORCH_LIBRARY(box_ops, m) { m.def("box_convert(Tensor input, str in_fmt, str out_fmt) -> Tensor"); }

template<typename T> CXCYWH<T> xyxy_to_cxcywh(const XYXY<T> box)
{
    CXCYWH<T> result;
    result.cx = (box.x1 + box.x2) * 0.5f;
    result.cy = (box.y1 + box.y2) * 0.5f;
    result.w = box.x2 - box.x1;
    result.h = box.y2 - box.y1;
    return result;
}

template<typename T> CXCYWH<T> xywh_to_cxcywh(const XYWH<T> box)
{
    CXCYWH<T> result;
    result.cx = box.x + box.w * 0.5f;
    result.cy = box.y + box.h * 0.5f;
    result.w = box.w;
    result.h = box.h;
    return result;
}

template<typename T> XYXY<T> cxcywh_to_xyxy(const CXCYWH<T> box)
{
    XYXY<T> result;
    result.x1 = box.cx - box.w * 0.5f;
    result.y1 = box.cy - box.h * 0.5f;
    result.x2 = box.cx + box.w * 0.5f;
    result.y2 = box.cy + box.h * 0.5f;
    return result;
}

template<typename T> XYXY<T> xywh_to_xyxy(const XYWH<T> box)
{
    XYXY<T> result;
    result.x1 = box.x;
    result.y1 = box.y;
    result.x2 = box.x + box.w;
    result.y2 = box.y + box.h;
    return result;
}

template<typename T> XYWH<T> cxcywh_to_xywh(const CXCYWH<T> box)
{
    XYWH<T> result;
    result.x = box.cx - box.w * 0.5f;
    result.y = box.cy - box.h * 0.5f;
    result.w = box.w;
    result.h = box.h;
    return result;
}

template<typename T> XYWH<T> xyxy_to_xywh(const XYXY<T> box)
{
    XYWH<T> result;
    result.x = box.x1;
    result.y = box.y1;
    result.w = box.x2 - box.x1;
    result.h = box.y2 - box.y1;
    return result;
}

auto box_convert_cpu(const torch::Tensor &input, const std::string &in_fmt, const std::string &out_fmt) -> torch::Tensor
{
    auto output = torch::empty_like(input);
    auto numBoxes = input.numel() >> 2;// Assuming input is of shape (..., 4) for boxes
    if (in_fmt == out_fmt) {
        output.copy_(input);// No conversion needed, just return a copy
    } else if (in_fmt == "xyxy" && out_fmt == "cxcywh") {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_cpu", [&] {
            const auto input_data = static_cast<const XYXY<scalar_t> *>(input.data_ptr());
            auto output_data = static_cast<CXCYWH<scalar_t> *>(output.data_ptr());
            std::transform(input_data, input_data + numBoxes, output_data, xyxy_to_cxcywh<scalar_t>);
        });
    } else if (in_fmt == "xywh" && out_fmt == "cxcywh") {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_cpu", [&] {
            const auto input_data = static_cast<const XYWH<scalar_t> *>(input.data_ptr());
            auto output_data = static_cast<CXCYWH<scalar_t> *>(output.data_ptr());
            std::transform(input_data, input_data + numBoxes, output_data, xywh_to_cxcywh<scalar_t>);
        });
    } else if (in_fmt == "cxcywh" && out_fmt == "xywh") {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_cpu", [&] {
            const auto input_data = static_cast<const CXCYWH<scalar_t> *>(input.data_ptr());
            auto output_data = static_cast<XYWH<scalar_t> *>(output.data_ptr());
            std::transform(input_data, input_data + numBoxes, output_data, cxcywh_to_xywh<scalar_t>);
        });
    } else if (in_fmt == "xyxy" && out_fmt == "xywh") {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_cpu", [&] {
            const auto input_data = static_cast<const XYXY<scalar_t> *>(input.data_ptr());
            auto output_data = static_cast<XYWH<scalar_t> *>(output.data_ptr());
            std::transform(input_data, input_data + numBoxes, output_data, xyxy_to_xywh<scalar_t>);
        });
    } else if (in_fmt == "cxcywh" && out_fmt == "xyxy") {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_cpu", [&] {
            const auto input_data = static_cast<const CXCYWH<scalar_t> *>(input.data_ptr());
            auto output_data = static_cast<XYXY<scalar_t> *>(output.data_ptr());
            std::transform(input_data, input_data + numBoxes, output_data, cxcywh_to_xyxy<scalar_t>);
        });
    } else if (in_fmt == "xywh" && out_fmt == "xyxy") {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "box_convert_cpu", [&] {
            const auto input_data = static_cast<const XYWH<scalar_t> *>(input.data_ptr());
            auto output_data = static_cast<XYXY<scalar_t> *>(output.data_ptr());
            std::transform(input_data, input_data + numBoxes, output_data, xywh_to_xyxy<scalar_t>);
        });
    } else {
        throw std::invalid_argument("Unsupported format conversion");
    }
    return output;
}

TORCH_LIBRARY_IMPL(box_ops, CPU, m) { m.impl("box_convert", &box_convert_cpu); }
