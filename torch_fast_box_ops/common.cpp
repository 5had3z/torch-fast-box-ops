
#include <torch/extension.h>

TORCH_LIBRARY(box_ops, m)
{
    m.def("iou(Tensor boxes1, Tensor boxes2) -> Tensor");
    m.def("box_convert(Tensor input, str in_fmt, str out_fmt) -> Tensor");
    m.def("box_convert_backward(Tensor grad, str in_fmt, str out_fmt) -> Tensor");
}
